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




# 1. Datenvorbereitung
class ISLESDataset(Dataset):

    def __init__(self, subjects, root_dir, tabular_data=None, transform=None, target_size=(128, 128, 128)):
        """
        Args:
            subjects (list): Liste der Subjekt-IDs (z.B. ['sub-stroke0001', 'sub-stroke0002'])
            root_dir (str): Pfad zum Root-Verzeichnis (z.B. 'D:\praktikum\swin_unet\Swin-Unet\data_\derivatives')
            tabular_data (DataFrame, optional): Tabellarische Daten
            transform (callable, optional): Transformationen für die Bilddaten
        """
        self.subjects = subjects
        self.root_dir = root_dir
        self.tabular_data = tabular_data
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        #TODO remove this afterwards
        #idx = 5
        subject = self.subjects[idx]
        ses01_dir = os.path.join(self.root_dir, subject, 'ses-01')
        ses02_dir = os.path.join(self.root_dir, subject, 'ses-02')

        # Bildmodalitäten in ses-01
        cta_path = os.path.join(ses01_dir, f"{subject}_ses-01_space-ncct_cta.nii.gz")
        ctp_path = os.path.join(ses01_dir, f"{subject}_ses-01_space-ncct_ctp.nii.gz")

        perfusion_dir = os.path.join(ses01_dir, 'perfusion-maps')
        cbf_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_cbf.nii.gz")
        cbv_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_cbv.nii.gz")
        mtt_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_mtt.nii.gz")
        tmax_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_tmax.nii.gz")

        # Mask in ses-02
        mask_path = os.path.join(ses02_dir, f"{subject}_ses-02_lesion-msk.nii.gz")

        # Laden der Bilddaten
        cta = nib.load(cta_path).get_fdata()
        cbf = nib.load(cbf_path).get_fdata()
        cbv = nib.load(cbv_path).get_fdata()
        mtt = nib.load(mtt_path).get_fdata()
        tmax = nib.load(tmax_path).get_fdata()

        # Laden der Maske
        mask = nib.load(mask_path).get_fdata()
        mask = mask.round().astype(np.uint8)

        #get a randonm slice of the current volume
        #TODO all possible slices or just the ones where mask has at least one Non-Zero value???
        #slices_idx = list(range(cta.shape[-1]))
        slices_idx = np.unique(np.argwhere(mask != 0)[:, -1])
        target_slice_idx = np.random.choice(slices_idx)

        cta_slice = cta[..., target_slice_idx]
        cbf_slice = cbf[..., target_slice_idx]
        cbv_slice = cbv[..., target_slice_idx]
        mtt_slice = mtt[..., target_slice_idx]
        tmax_slice = tmax[..., target_slice_idx]
        mask_slice = mask[..., target_slice_idx]

        #preprocessing starts here (clip, resize, min_max)
        # (1) clip nur cta slice erstmal
        cta_slice = np.clip(cta_slice, a_min=-100, a_max=100)
        cbf_slice = np.clip(cbf_slice, a_min=0, a_max=100)
        cbv_slice = np.clip(cbv_slice, a_min=0, a_max=10)
        mtt_slice = np.clip(mtt_slice, a_min=0, a_max=20)
        tmax_slice = np.clip(tmax_slice, a_min=0, a_max=15)

        # (2) resize (512x512)
        # Resize das Bild mit bilinearer Interpolation
        resized_cta = cv2.resize(cta_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_cbf = cv2.resize(cbf_slice, (512,512), interpolation=cv2.INTER_LINEAR)
        resized_cbv = cv2.resize(cbv_slice, (512,512), interpolation=cv2.INTER_LINEAR)
        resized_mtt = cv2.resize(mtt_slice, (512,512), interpolation=cv2.INTER_LINEAR)
        resized_tmax = cv2.resize(tmax_slice, (512,512), interpolation=cv2.INTER_LINEAR)

        # Resize die Maske mit nearest-neighbor Interpolation
        resized_mask = cv2.resize(mask_slice, (512, 512), interpolation=cv2.INTER_NEAREST)

        # (3) min_max normalisierung
        #cta_slice = min_max_normalization(resized_cta)
        cta_slice = resized_cta
        #cbf_slice = min_max_normalization(cbf_slice)
        #cbv_slice = min_max_normalization(cbv_slice)
        #mtt_slice = min_max_normalization(mtt_slice)
        #tmax_slice = min_max_normalization(tmax_slice)

        # Kombiniere die CTP-Parameter in einem Array
        ctp_combined_slice = np.stack([resized_cbf, resized_cbv, resized_mtt, resized_tmax], axis=-1)

        # Erstelle die Eingabe für das neuronale Netzwerk
        # Die Input-Size ist (512, 512, 5)
        # Dabei wird die letzte Dimension die Anzahl der Input-Channels für Netz sein
        # Kanal 0: CTA-Daten
        # Kanäle 1-4: CTP-Parameter
        input_data = np.concatenate((cta_slice[..., np.newaxis], ctp_combined_slice), axis=-1)
        #input_data = cta_slice

        # Konvertiere die Daten in PyTorch-Tensoren
        input_tensor = torch.tensor(input_data, dtype=torch.float32).permute(2,0,1)
        #input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        target_tensor = torch.tensor(np.round(resized_mask), dtype=torch.float32)

        # Optional: Laden der tabellarischen Daten
        if self.tabular_data is not None:
            tabular_features = self.tabular_data.iloc[idx].values
            tabular_features = torch.tensor(tabular_features, dtype=torch.float32)
        else:
            tabular_features = None

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        # Konvertiere alles in Tensoren
        sample = {
            'image': input_tensor,  # Shape: (5, H, W)
            'mask': target_tensor  # Shape: (H, W)
        }

        if tabular_features is not None:
            sample['tabular'] = tabular_features

        return sample


# 2. Dataloader
def get_data_loaders(root_dir, batch_size=2, batch_size_test=1, transform=None, test_size=0.2, random_state=42,
                     tabular_csv=None):
    """
    Args:
        root_dir (str): Pfad zum Root-Verzeichnis (z.B. 'D:/derivatives')
        batch_size (int): Größe der Chargen
        transform (callable, optional): Transformationen für die Bilddaten
        test_size (float): Anteil der Validierungsdaten
        random_state (int): Zufallsstate für die Aufteilung
        tabular_csv (str, optional): Pfad zur tabellarischen CSV-Datei
    """
    # Liste aller Subjekte, falls
    subjects = os.listdir(root_dir)

    # Laden der tabellarischen Daten, falls vorhanden
    if tabular_csv is not None:
        tabular_data = pd.read_csv(tabular_csv)
        # Annahme: Die CSV hat eine Spalte 'subject' zur Verknüpfung
        tabular_data.set_index('subject', inplace=True)
        tabular_data = tabular_data.loc[subjects].reset_index(drop=True)
        scaler = StandardScaler()
        tabular_features = scaler.fit_transform(tabular_data)
        tabular_df = pd.DataFrame(tabular_features, columns=tabular_data.columns)
    else:
        tabular_df = None

    # Aufteilen in Training und Validierung
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size, random_state=random_state)
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=test_size * 2, random_state=random_state)

    # Erstellen der Dataset-Objekte
    train_dataset = ISLESDataset(train_subjects, root_dir, tabular_data=tabular_df, transform=transform)
    val_dataset = ISLESDataset(val_subjects, root_dir, tabular_data=tabular_df, transform=transform)
    test_dataset = ISLESDataset(test_subjects, root_dir, tabular_data=tabular_df)
    #print(train_dataset[0])
    #print(val_dataset[0])
    #print(test_dataset[0])

    # Erstellen der DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, val_loader, test_loader


# 3. Trainings- und Validierungsschleife
def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=1e-4, device='cuda'):
    criterion = nn.BCEWithLogitsLoss()

    #class_weights = torch.tensor([0.0058, 1.9942]).to(device)
    criterion_dice = DiceLoss(num_classes=2, weights=[0.0058, 1.9942])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    best_loss = 2.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, total=len(train_loader)):
            images = batch['image'].to(device)  # Shape: (B, 5, H, W)
            masks = batch['mask'].to(device)  # Shape: (B, H, W)
            masks = masks.unsqueeze(1)  # Shape: (B, 1, H, W)

            optimizer.zero_grad()
            outputs = model(images)  # Shape: (B, 1, H, W)

            loss_dice = criterion_dice(outputs, masks)

            loss = loss_dice

            # loss_ce = criterion(outputs, masks)
            # loss = 0.4 * loss_ce + 0.6 * loss_dice

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if loss_dice.item() < best_loss:
                best_loss = loss_dice.item()
                #save model parameters + plot Input Image + Prediction + Target
                save_model(model, epoch)  # Save the model
                # Prepare to plot images
                with torch.no_grad():
                    preds = torch.sigmoid(outputs)
                    #preds = (preds > 0.5).float()  # Convert predictions to binary

                # Plotting images, predictions, and targets (show all 5 images from batch)
                plot_images(images*255, preds, masks, num_images=5)

        train_loss /= len(train_loader)

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
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Dice Score
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                intersection = (preds * masks).sum()
                dice += (2. * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)  #(dim=(2,3,4) when using volumes)

        val_loss /= len(val_loader.dataset)
        dice_score_val = (dice / len(val_loader.dataset)).item()

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice Score: {dice_score_val:.4f}")


# 5. Evaluierung (Dice Score)
# Bereits im Trainingsloop integriert

# 6. Main program
if __name__ == "__main__":
    # Pfad zum Root-Verzeichnis
    root_dir = os.path.join(os.getcwd(), 'data_', 'derivatives')

    # Optional: Pfad zur tabellarischen CSV-Datei
    tabular_csv = "path_to_tabular_data.csv"  # Anpassen oder auf None setzen, wenn nicht vorhanden

    # Transformationen (z.B. könnte man auch Datenaugmentation hinzufügen)
    transform = transforms.Compose([
        # Placeholder: für 2D Transformationen
    ])

    # Daten-Loader erstellen
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=5,
        batch_size_test=1,
        transform=None,  # Anpassung je nach Bedarf
        test_size=0.1,
        random_state=42,
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
        learning_rate=1e-3,#1e-4
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
