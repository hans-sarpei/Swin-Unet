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
import torch.nn.functional as F
from networks import unet_model, Swin_Unet


#paths_scans = glob("./data_/**/*.gz", recursive=True)
#raw_ncct_paths = list(filter(lambda k: 'raw_data' in k and 'perfusion-maps' not in k and 'ncct' in k, paths_scans))
#mask_paths = list(filter(lambda k: 'msk' in k, paths_scans))

def save_model(model, epoch, path='best_model.pth'):
    """Speichert das Modell."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, path)


def plot_images(images, predictions, targets, num_images=5):
    """Plotting function for images, predictions, and targets."""
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    for i in range(num_images):
        axs[i, 0].imshow(images[i][0].cpu().numpy())#.transpose(1, 2, 0))  # nur notwendig wenn Input = (B, 1, H,W)
        axs[i, 0].set_title("Input Image")
        axs[i, 1].imshow(predictions[i].cpu().numpy().squeeze(), cmap='gray')  # Squeeze to remove channel dimension
        axs[i, 1].set_title("Prediction")
        axs[i, 2].imshow(targets[i].cpu().numpy().squeeze(), cmap='gray')  # Squeeze to remove channel dimension
        axs[i, 2].set_title("Target")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def determine_crop_coordinates(mask, target_size):
    """Bestimmt die Startkoordinaten für das Cropping basierend auf der Maske und der Zielgröße."""
    if np.sum(mask) == 0:
        raise ValueError("Die Maske enthält keine Werte.")

    # Finde die Indizes, wo die Maske nicht Null ist
    h_indices, w_indices, s_indices = np.nonzero(mask)

    # Bestimmen der minimalen und maximalen Indizes
    min_h, max_h = h_indices.min(), h_indices.max()
    min_w, max_w = w_indices.min(), w_indices.max()
    min_s, max_s = s_indices.min(), s_indices.max()

    # Berechnen der Mitte der ROI
    center_h = (min_h + max_h) // 2
    center_w = (min_w + max_w) // 2
    center_s = (min_s + max_s) // 2

    # Berechnen der Startkoordinaten für den Crop
    start_coords = (
        max(center_h - target_size[0] // 2, 0),  # Sicherstellen, dass wir nicht außerhalb der Grenzen liegen
        max(center_w - target_size[1] // 2, 0),
        max(center_s - target_size[2] // 2, 0),
    )
    center_coords = (
        min(max_h, center_h),
        min(max_w, center_w),
        min(min_s, center_s),
    )

    return center_coords


def plot_dist(slice):
    # Seaborn für die Darstellung verwenden
    sns.histplot(slice)  # , kde=True)  # kde=True zeigt die Dichtekurve zusätzlich zum Histogramm

    # Titel und Beschriftungen hinzufügen
    plt.title('Verteilung der Werte')
    plt.xlabel('Wert')
    plt.ylabel('Häufigkeit')

    # Anzeige der Grafik
    plt.legend().remove()
    plt.show()


def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


class DiceLoss(nn.Module):
    def __init__(self, weights=None, num_classes=2, size_average=True):
        super(DiceLoss, self).__init__()
        self.weights = weights
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1e-5):

        #get probabilities
        inputs = F.sigmoid(inputs)

        if self.weights is None:
            weight = torch.ones(self.num_classes).to(inputs.device)
        else:
            weight = torch.tensor(self.weights, dtype=torch.float32).to(inputs.device)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection_les = (inputs * targets).sum()
        dice_les = (2. * intersection_les + smooth) / (inputs.sum() + targets.sum() + smooth)


        intersection_non_les = ((inputs == 0) & (targets == 0)).sum()
        dice_non_les = (2. * intersection_non_les + smooth) / ((inputs == 0).sum() + (targets == 0).sum() + smooth)

        dice_all = dice_non_les * weight[0] + dice_les * weight[1]
        weighted_dice = dice_all / self.num_classes

        return 1 - weighted_dice


#dice_loss = DiceLoss(weights=[0.0058, 1.9942], num_classes=2)

#pred = torch.rand(1, 256, 256)
#target = torch.randint(low=0, high=2, size=(1, 256, 256))

#loss = dice_loss(pred, target)


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
        #ctp = nib.load(ctp_path).get_fdata() -> difficult to integrate at the moment
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

        # Normalisiere die Daten (optional, abhängig von der Modellarchitektur) - Z-Normalisierung gut??
        #cta_data = (cta - np.mean(cta)) / np.std(cta)
        #ctp_combined = (ctp_combined - np.mean(ctp_combined)) / np.std(ctp_combined)

        # Erstelle die Eingabe für das neuronale Netzwerk
        # Die Eingabe könnte (512, 512, 74, 5) sein, wenn du CTP-Daten und CTA-Daten kombinierst
        # Dabei wird die letzte Dimension die Anzahl der Kanäle sein
        # Kanal 0: CTA-Daten
        # Kanäle 1-4: CTP-Parameter
        input_data = np.concatenate((cta_slice[..., np.newaxis], ctp_combined_slice), axis=-1)
        #input_data = cta_slice

        # Konvertiere die Daten in PyTorch-Tensoren
        #input_tensor = torch.tensor(input_data, dtype=torch.float32).permute(3, 0, 1, 2) -> 4d permutation on volumes
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
            'image': input_tensor,  # Shape: (5, H, W, D)
            'mask': target_tensor  # Shape: (H, W, D)
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
    #subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
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


# 3. Modellarchitektur (U-Net für Segmentierung)
# siehe networks unet_model.py


# 4. Trainings- und Validierungsschleife
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
            images = batch['image'].to(device)  # Shape: (B, 6, H, W, D)
            masks = batch['mask'].to(device)  # Shape: (B, H, W, D)
            masks = masks.unsqueeze(1)  # Shape: (B, 1, H, W, D)

            optimizer.zero_grad()
            if torch.isnan(images).any():
                continue
            outputs = model(images)  # Shape: (B, 1, H, W)
            #loss_ce = criterion(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            #loss = 0.4 * loss_ce + 0.6 * loss_dice
            loss = loss_dice

            # Anwendung der Gewichtung
            #weighted_loss = loss * class_weights

            # Durchschnitt über alle Elemente
            #final_loss = weighted_loss.mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

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

        train_loss /= len(train_loader.dataset)

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
        # Placeholder: 3D-Transformationen sind komplexer und erfordern spezialisierte Bibliotheken wie TorchIO oder MONAI
        # Für Einfachheit hier nur eine Identity-Transformation
    ])

    # Daten-Loader erstellen
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=5,  #can't use  batch_size bigger than 1 cause subject scans are in  different shape
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
