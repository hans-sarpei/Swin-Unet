import os
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from monai.transforms import Resize, Compose, SpatialCrop
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from networks import unet_model, Swin_Unet
from utils import *
#from torch.cuda.amp import GradScaler, autocast -> deprecated
from torch.amp import autocast, GradScaler
import torch
from torch.utils.data import Sampler

from monai.losses import DiceLoss as DiceL
from monai.metrics import compute_dice
import torchvision.transforms.functional as F
from functools import lru_cache
import torch.nn.utils as utils
import sys

class ISLESDataset(Dataset):

    def __init__(self, paths, tabular_data=None, transform=None, type='train'):
        """
        Args:
            paths (list): Liste der Pfade zu den .npz Files
            tabular_data (DataFrame, optional): Tabellarische Daten
            transform (callable, optional): Transformationen für die Bilddaten
        """
        self.paths = paths
        self.tabular_data = tabular_data
        self.transform = transform
        self.cache = {}
        self.type = type

    def __len__(self):
        return len(self.paths)

    @lru_cache(maxsize=1024)
    def load_file(self, idx):
        #generate path for .npy File
        #(1)get slice ID
        with np.load(self.paths[idx]) as data:
            data_copy = {key: data[key].copy() for key in data.files}
        return data_copy


    def __getitem__(self, idx):
        # Check whether slice is already in cache
        if idx in self.cache:
            data = self.cache[idx]
        else:
            data = self.load_file(idx)
            self.cache[idx] = data

        cta = data['image_cta']
        cbf = data['image_cbf']
        cbv = data['image_cbv']
        mtt = data['image_mtt']
        tmax = data['image_tmax']
        label = data['label']

        # Kombiniere die CTP-Parameter in einem Array
        ctp_combined_slice = np.stack([cbf, cbv, mtt, tmax], axis=-1)

        # Erstelle die Eingabe für das neuronale Netzwerk
        # Die Input-Size ist (512, 512, 5)
        # Dabei wird die letzte Dimension die Anzahl der Input-Channels für Netz sein
        # Kanal 0: CTA-Daten
        # Kanäle 1-4: CTP-Parameter
        input_data = np.concatenate((cta[..., np.newaxis], ctp_combined_slice), axis=-1)

        # Konvertiere die Daten in PyTorch-Tensoren
        input_tensor = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1)
        # input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        target_tensor = torch.tensor(label, dtype=torch.float32)

        sample = {
            'image': input_tensor,  # Shape: (5, H, W)
            'mask': target_tensor  # Shape: (H, W)
        }

        return sample


# 2. Dataloader
def get_data_loaders(paths, batch_size=2, batch_size_test=1, transform=None,
                     tabular_csv=None):
    """
    Args:
        paths (dict): Pfade zu den .npz Files pro Typ 'train', 'val'
        batch_size (int): Größe der Chargen
        transform (callable, optional): Transformationen für die Bilddaten
        tabular_csv (str, optional): Pfad zur tabellarischen CSV-Datei
    """


    # Erstellen der Dataset-Objekte
    train_dataset = ISLESDataset(paths['train'], transform=transform, type='train')
    val_dataset = ISLESDataset(paths['val'], transform=transform, type='val')
    #test_dataset = ISLESDataset(paths['test'], transform=transform, type='test')

    # Erstellen der DataLoader-Instanzen
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,  # persistent_workers=True,
                              num_workers=0, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)  # persistent_workers=True
    #test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, sampler=sampler_test)

    return train_loader, val_loader



# 3. Trainings- und Validierungsschleife
def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=1e-4, device='cuda'):

    class_weights = torch.tensor([0.0058, 1.9942]).to(device)
    criterion_dice = DiceL(sigmoid=True, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)#AdamW anstelle von Adam verwenden
    scaler = GradScaler()

    model = model.to(device)

    best_loss = 2.0


    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for i, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            images = batch['image'].to(device)  # Shape: (B, 5, H, W)
            masks = batch['mask'].to(device)  # Shape: (B, H, W)
            masks = masks.unsqueeze(1)  # Shape: (B, 1, H, W)

            if torch.isnan(images).any():
                print(f"Eingabedaten enthalten NaN-Werte. In Epoche:{epoch+1} und Batch: {batch+1}")

            optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)  # Shape: (B, 1, H, W)
                    loss_dice = criterion_dice(outputs, masks)# returns (B,C,1,1) -> torch.squeeze()

                    # Berechne den Mean Loss
                    mean_loss = loss_dice.mean()
                    loss = mean_loss

                if torch.isnan(loss):
                    print(f"Train-Loss hat NaN-Wert! Abbruch. In Epoche:{epoch+1} und Batch: {batch+1}")
                    sys.exit()

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            #loss.backward()
            #optimizer.step()

            train_loss += loss.item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                #save model parameters + plot Input Image + Prediction + Target
                save_model(model, epoch)  # Save the model
                # Prepare to plot images
                with torch.no_grad():
                    preds = torch.sigmoid(outputs)
                    #preds = (preds > 0.5).float()  # Convert predictions to binary

                # Plotting images, predictions, and targets (show all 5 images from batch)
                plot_images(images*255, preds, masks, num_images=2, epoch=epoch, batch=i+1, loss=best_loss)

        train_loss /= len(train_loader)  # get a mean over all iterations in the current epoch

        # Validierung
        model.eval()
        val_loss = 0.0
        dice = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader)):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                masks = masks.unsqueeze(1)

                outputs = model(images)
                loss = criterion_dice(outputs, masks)
                mean_loss = loss.mean()
                val_loss += mean_loss.item()

                dice += 1 - mean_loss.item()

        val_loss /= len(val_loader)
        dice_score_val = (dice / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice Score: {dice_score_val:.4f}")


# 5. Main program
if __name__ == "__main__":
    # Pfad zum Root-Verzeichnis
    #root_dir = os.path.join(os.getcwd(), 'data_', 'derivatives') -> lokaler Pfad
    root_dir = '/storage/ISLES24/ISLES24/preprocessed'


    # Optional: Pfad zur tabellarischen CSV-Datei
    #tabular_csv = "path_to_tabular_data.csv"  # Anpassen oder auf None setzen, wenn nicht vorhanden

    # Transformationen (z.B. könnte man auch Datenaugmentation hinzufügen)
    transform = transforms.Compose([
        # Placeholder: für 2D Transformationen
    ])


    paths_train = glob(os.path.join(root_dir, 'train', '*.npz'))
    paths_val = glob(os.path.join(root_dir, 'val', '*.npz'))
    paths = {'train': paths_train, 'val': paths_val}

    # Daten-Loader erstellen
    train_loader, val_loader= get_data_loaders(
        paths=paths,
        batch_size=2,
        batch_size_test=1,
        transform=None  # Anpassung je nach Bedarf
        #tabular_csv=tabular_csv
    )

    # Modellinitialisierung
    model = unet_model.UNet(n_channels=5, n_classes=1)
    #model = SwinUNet(512,512,1,48, 1)


    # Training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=150,
        learning_rate=1e-4,#1e-4
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )