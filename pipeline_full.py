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




class RepeatVolumeSampler(Sampler):
    def __init__(self, dataset):
        """
        Args:
            dataset (CTVolumeDataset): Das Dataset, das die Slices pro Volumen enthält.
        """
        self.dataset = dataset
        self.current_idx = 0  # Startindex für die Volumen
        self.remaining_slices = dataset.num_slices[self.current_idx]  # Anzahl der Slices im ersten Volumen

    def __iter__(self):
        self.current_idx = 0
        self.remaining_slices = self.dataset.num_slices[self.current_idx] #set it for next epoch to start
        while True:
            # Solange Slices im aktuellen Volumen verbleiben, gib den aktuellen Index aus
            while self.remaining_slices > 0:
                yield self.current_idx
                self.remaining_slices -= 1

            # Wenn alle Slices des aktuellen Volumens verarbeitet wurden, gehe zum nächsten Volumen
            self.current_idx += 1

            # Wenn das Ende des Volumen-Datasets erreicht ist, breche ab
            if self.current_idx >= len(self.dataset):
                break

            # Setze die verbleibenden Slices für das nächste Volumen
            self.remaining_slices = self.dataset.num_slices[self.current_idx]

    def __len__(self):
        # Die Gesamtzahl der Samples ist die Summe aller Slices über alle Volumen
        return sum(self.dataset.num_slices)





# 1. Datenvorbereitung
class ISLESDataset(Dataset):

    def __init__(self, subjects, root_dir, num_slices: list, tabular_data=None, transform=None, target_size=(128, 128, 128)):
        """
        Args:
            subjects (list): Liste der Subjekt-IDs (z.B. ['sub-stroke0001', 'sub-stroke0002'])
            root_dir (str): Pfad zum Root-Verzeichnis (z.B. 'D:\praktikum\swin_unet\Swin-Unet\data_\derivatives')
            tabular_data (DataFrame, optional): Tabellarische Daten
            transform (callable, optional): Transformationen für die Bilddaten
        """
        self.subjects = subjects
        #für jedes Volume bin ich initial bei Slice_index = 0
        self.current_slice_idx = {i: 0 for i in range(len(self.subjects))}
        self.root_dir = root_dir
        self.tabular_data = tabular_data
        self.transform = transform
        self.target_size = target_size
        self.num_slices = num_slices #[num_slice_vol1, num_slice_vol2, ...] für jedes Volumen wie viel Slices es hat
        self.cache = {}

    def __len__(self):
        return len(self.subjects)

    def load_files(self, subject):
        #create paths to load
        ses01_dir = os.path.join(self.root_dir, subject, 'ses-01')
        # ses01_dir_ncct = os.path.join(os.path.split(self.root_dir)[0], 'raw_data', subject, 'ses-01')
        ses02_dir = os.path.join(self.root_dir, subject, 'ses-02')

        # Create paths for all modalities (cta, cbf, cbv, etc.)
        cta_path = os.path.join(ses01_dir, f"{subject}_ses-01_space-ncct_cta.nii.gz")
        perfusion_dir = os.path.join(ses01_dir, 'perfusion-maps')
        cbf_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_cbf.nii.gz")
        cbv_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_cbv.nii.gz")
        mtt_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_mtt.nii.gz")
        tmax_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_tmax.nii.gz")
        # ncct_path = os.path.join(ses01_dir_ncct, f"{subject}_ses-01_space-ncct.nii.gz")
        # Mask in ses-02
        mask_path = os.path.join(ses02_dir, f"{subject}_ses-02_lesion-msk.nii.gz")


        # Laden der CT-Bildmodalitäten
        cta = nib.load(cta_path).get_fdata()
        cbf = nib.load(cbf_path).get_fdata()
        cbv = nib.load(cbv_path).get_fdata()
        mtt = nib.load(mtt_path).get_fdata()
        tmax = nib.load(tmax_path).get_fdata()

        # Laden der Maske
        mask = nib.load(mask_path).get_fdata()

        return [cta, cbf, cbv, mtt, tmax, mask]


    def __getitem__(self, idx):
        subject = self.subjects[idx]
        #get the first slice_index for the current volume 'self.subjects[idx]'
        slice_index = self.current_slice_idx[idx]

        # Update the current slice index for Volume self.subjects[idx]
        self.current_slice_idx[idx] += 1

        # Reset slice index if we reach the end of the volume
        if self.current_slice_idx[idx] >= self.num_slices[idx]:
            self.current_slice_idx[idx] = 0  # Reset to 0 for next epoch



        #Check weather subject volume is already in cache
        if subject in self.cache:
            data = self.cache[subject]
        else:
            data = self.load_files(subject)
            self.cache[subject] = data

        cta, cbf, cbv, mtt, tmax, mask = data

        mask = mask.round().astype(np.uint8)

        #get a random slice idx of the current volume with a size >= median
        #median_nz_count_over_all_slices = np.median(np.unique(np.sum(mask, axis=(0,1)))[1:]) #excluding zero-valued slices
        #slices_idx = np.argwhere(np.sum(mask, axis=(0,1)) >= median_nz_count_over_all_slices)
        #slices_idx = np.unique(np.argwhere(mask != 0)[:, -1])
        target_slice_idx = slice_index
        #target_slice_idx = np.random.choice(slices_idx[...,0])

        cta_slice = cta[..., target_slice_idx]
        cbf_slice = cbf[..., target_slice_idx]
        cbv_slice = cbv[..., target_slice_idx]
        mtt_slice = mtt[..., target_slice_idx]
        tmax_slice = tmax[..., target_slice_idx]
        mask_slice = mask[..., target_slice_idx]

        #put them on the GPU -> didnt work with threading let things on the CPU


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
        resized_cbf = cv2.resize(cbf_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_cbv = cv2.resize(cbv_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_mtt = cv2.resize(mtt_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_tmax = cv2.resize(tmax_slice, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Resize die Maske mit nearest-neighbor Interpolation
        resized_mask = cv2.resize(mask_slice, (512, 512), interpolation=cv2.INTER_NEAREST)

        # (3) min_max normalisierung
        cta_slice = min_max_normalization(resized_cta)
        #cta_slice = resized_cta
        cbf_slice = min_max_normalization(resized_cbf)
        cbv_slice = min_max_normalization(resized_cbv)
        mtt_slice = min_max_normalization(resized_mtt)
        tmax_slice = min_max_normalization(resized_tmax)

        # Kombiniere die CTP-Parameter in einem Array
        ctp_combined_slice = np.stack([cbf_slice, cbv_slice, mtt_slice, tmax_slice], axis=-1)
        #ctp_combined_slice = np.stack([resized_cbf, resized_cbv, resized_mtt, resized_tmax], axis=-1)

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

def calculate_num_slices_per_volume(subject_list: list):
    r_dir = '/storage/ISLES24/ISLES24/raw_data'
    num_slices = []
    for subject in subject_list:
        ses_1_path = os.path.join(r_dir, subject, 'ses-01')
        ncct_path = os.path.join(ses_1_path, f'{subject}_ses-01_ncct.nii.gz')
        ncct_data = nib.load(ncct_path).get_fdata()
        num_slices.append(ncct_data.shape[-1])
    return num_slices



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
        subjects[:7], test_size=test_size, random_state=random_state)
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=test_size * 2, random_state=random_state)

    #Berechne auf train, val und test_subjects die Anzahl an Slices pro Volumen
    num_slices_train = calculate_num_slices_per_volume(train_subjects)
    num_slices_val = calculate_num_slices_per_volume(val_subjects)
    num_slices_test = calculate_num_slices_per_volume(test_subjects)




    # Erstellen der Dataset-Objekte
    train_dataset = ISLESDataset(train_subjects, root_dir, tabular_data=tabular_df, transform=transform, num_slices=num_slices_train)
    val_dataset = ISLESDataset(val_subjects, root_dir, tabular_data=tabular_df, transform=transform, num_slices=num_slices_val)
    test_dataset = ISLESDataset(test_subjects, root_dir, tabular_data=tabular_df, num_slices=num_slices_test)
    #print(train_dataset[0])
    #print(val_dataset[0])
    #print(test_dataset[0])

    #sampler gibt solange den CT-Volume Index zurück bis alle Slices vom Dataloader gezogen wurden
    #dies stellt sicher dass während Epoche jedes Slice vom Volumen einmal gesehen wurde
    sampler_train = RepeatVolumeSampler(train_dataset)
    sampler_val = RepeatVolumeSampler(val_dataset)
    sampler_test = RepeatVolumeSampler(test_dataset)

    # Erstellen der DataLoader shuffle=True -> muss ich entfernen da sonst nicht sichergestellt werden kann,
    # dass alle slices pro volumen gesehen werden


    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, #persistent_workers=True,
                              num_workers=0, sampler=sampler_train, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, sampler=sampler_val, num_workers=0,
                            )#persistent_workers=True
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, sampler=sampler_test)

    return train_loader, val_loader, test_loader


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

            optimizer.zero_grad()

            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)  # Shape: (B, 1, H, W)
                loss_dice = criterion_dice(outputs, masks)# returns (B,C,1,1) -> torch.squeeze()

                # Berechne den Mean Loss
                mean_loss = loss_dice.mean()
                loss = mean_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #loss.backward()
            #optimizer.step()

            train_loss += loss.item() if loss.item() is not float('nan') else print(f'batch: {batch} in epoch {epoch} got calculated a NaN value')

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

        train_loss /= len(train_loader) #get a mean over all iterations in the current epoch

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

                # Dice Score
                #prediction = torch.sigmoid(outputs) #da es bei meiner MONAI Version nicht ein sigmoid=True Flag gibt.

                #dice_scores = compute_dice(prediction, masks, ignore_empty=True) -> mit compute_dice() hab ich immer wieder NANs erhalten
                #wenn es für eine Klasse keinen Overlap auf der Klasse gab

                dice += 1 - mean_loss.item()

        val_loss /= len(val_loader)
        dice_score_val = (dice / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice Score: {dice_score_val:.4f}")


# 5. Evaluierung (Dice Score)
# Bereits im Trainingsloop integriert

# 6. Main program
if __name__ == "__main__":
    # Pfad zum Root-Verzeichnis
    #root_dir = os.path.join(os.getcwd(), 'data_', 'derivatives') -> lokaler Pfad
    root_dir = '/storage/ISLES24/ISLES24/derivatives'


    # Optional: Pfad zur tabellarischen CSV-Datei
    #tabular_csv = "path_to_tabular_data.csv"  # Anpassen oder auf None setzen, wenn nicht vorhanden

    # Transformationen (z.B. könnte man auch Datenaugmentation hinzufügen)
    transform = transforms.Compose([
        # Placeholder: für 2D Transformationen
    ])

    # Daten-Loader erstellen
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=2,
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
        learning_rate=1e-4,#1e-4
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
