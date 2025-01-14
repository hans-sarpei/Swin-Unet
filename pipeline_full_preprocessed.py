import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from utils import *
import torch

from monai.losses import DiceLoss as DiceL
from functools import lru_cache
import sys
from monai.networks.nets import UNet
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

# Initialisiere den SummaryWriter (gib einen Log-Ordner an)
out_dir_results = "results/experiment_tmax_ncct"
os.makedirs(out_dir_results, exist_ok=True)
writer = SummaryWriter(log_dir=out_dir_results)

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

        ncct = data['image_ncct']
        tmax = data['image_tmax'][..., np.newaxis]
        label = data['label']

        # Kombiniere die CTP-Parameter in einem Array
        #ct_combined_slice = np.stack([cta, ncct], axis=-1)
        ct_combined_slice = ncct[..., np.newaxis]

        # Erstelle die Eingabe für das neuronale Netzwerk
        # Die Input-Size ist (512, 512, 2)
        # Dabei wird die letzte Dimension die Anzahl der Input-Channels für das Netz sein
        # Kanal 0: NCCT
        # Kanal 1: TMax
        input_data = np.concatenate([ct_combined_slice, tmax], axis=-1)
        #input_data = tmax

        # Konvertiere die Daten in PyTorch-Tensoren
        input_tensor = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1)
        target_tensor = torch.tensor(label, dtype=torch.float32)

        sample = {
            'image': input_tensor,  # Shape: (2, H, W)
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

    # Erstellen der DataLoader-Instanzen
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,  # persistent_workers=True,
                              num_workers=0, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)  # persistent_workers=True

    return train_loader, val_loader



# 3. Trainings- und Validierungsschleife
def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=1e-4, device='cuda'):

    class_weights_train = torch.tensor([0.0093, 1.9906]).to(device)
    class_weights_val = torch.tensor([0.00601, 1.9939]).to(device)
    criterion_dice_train = DiceL(sigmoid=True, weight=class_weights_train)
    criterion_dice_val = DiceL(sigmoid=True, weight=class_weights_val)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)#AdamW anstelle von Adam verwenden

    model = model.to(device)

    best_loss = 2.0


    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        tp_full, tn_full, fp_full, fn_full = (0,0,0,0)
        for i, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            images = batch['image'].to(device)  # Shape: (B, 5, H, W)
            masks = batch['mask'].to(device)  # Shape: (B, H, W)
            masks = masks.unsqueeze(1)  # Shape: (B, 1, H, W)

            if torch.isnan(images).any():
                print(f"Eingabedaten enthalten NaN-Werte. In Epoche:{epoch+1} und Batch: {batch+1}")

            optimizer.zero_grad()

            outputs = model(images)  # Shape: (B, 1, H, W)
            loss_dice = criterion_dice_train(outputs, masks)# returns (B,C,1,1) -> torch.squeeze()

            preds = torch.sigmoid(outputs).round()
            pred = preds.detach().cpu().numpy()
            mask = masks.cpu().numpy()
            tn, fp, fn, tp = confusion_matrix(list(mask.flatten()), list(pred.flatten()), labels=[0, 1]).ravel()
            tn_full += tn
            fp_full += fp
            tp_full += tp
            fn_full += fn

            # Berechne den Mean Loss
            mean_loss = loss_dice.mean()
            loss = mean_loss


            #gradient clipping um zu verhindern dass Gradienten zu groß werden können
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                '''save model parameters + plot Input Image + Prediction + Target'''
                save_model(model, epoch)  # Save the model
                # Prepare to plot images
                with torch.no_grad():
                    preds = torch.sigmoid(outputs)
                    #preds = (preds > 0.5).float()  # Convert predictions to binary

                # Plotting images, predictions, and targets (show all 2 images from batch)
                plot_images(images*255, preds, masks, num_images=2, epoch=epoch, batch=i+1, loss=best_loss)

        train_loss /= len(train_loader)  # get a mean over all iterations in the current epoch
        train_recall = tp_full/(fn_full+tp_full)
        train_specificity = tn_full/(tn_full+fp_full)
        train_precision = tp_full/(fp_full+tp_full)
        train_conf_matrix = torch.tensor([[tp_full, fp_full], [fn_full, tn_full]])


        # Validierung
        model.eval()
        val_loss = 0.0
        dice = 0.0
        tp_full_val, tn_full_val, fp_full_val, fn_full_val = (0, 0, 0, 0)
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader)):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                masks = masks.unsqueeze(1)

                out_val = model(images)
                loss = criterion_dice_val(out_val, masks)

                preds_val = torch.sigmoid(out_val).round()
                pred_val = preds_val.detach().cpu().numpy()
                mask = masks.cpu().numpy()
                tn_val, fp_val, fn_val, tp_val = confusion_matrix(list(mask.flatten()), list(pred_val.flatten()), labels=[0, 1]).ravel()
                tn_full_val += tn_val
                fp_full_val += fp_val
                tp_full_val += tp_val
                fn_full_val += fn_val


                mean_loss = loss.mean()
                val_loss += mean_loss.item()

                dice += 1 - mean_loss.item()

        val_loss /= len(val_loader)
        dice_score_val = (dice / len(val_loader))

        val_recall = tp_full_val / (fn_full_val + tp_full_val)
        val_specificity = tn_full_val / (tn_full_val + fp_full_val)
        val_precision = tp_full_val / (fp_full_val + tp_full_val)

        val_conf_matrix = torch.tensor([[tp_full_val, fp_full_val], [fn_full_val, tn_full_val]])

        # Werte speichern
        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Validation", val_loss, epoch+1)
        writer.add_scalar("Metric/Dice_Score", dice_score_val, epoch+1)
        writer.add_scalar("Metric/Train_recall", train_recall, epoch+1)
        writer.add_scalar("Metric/Train_specificity", train_specificity, epoch+1)
        writer.add_scalar("Metric/Train_precision", train_precision, epoch+1)
        writer.add_scalar("Metric/val_recall", val_recall, epoch + 1)
        writer.add_scalar("Metric/val_specificity", val_specificity, epoch + 1)
        writer.add_scalar("Metric/val_precision", val_precision, epoch + 1)
        writer.add_tensor("Tensor/Train-Confusion_Matrix", train_conf_matrix, global_step=epoch+1)
        writer.add_tensor("Tensor/Val-Confusion_Matrix", val_conf_matrix, global_step=epoch + 1)


        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice Score: {dice_score_val:.4f}")

    #schließen des writers
    writer.close()


# 5. Main program
if __name__ == "__main__":
    # Pfad zum Root-Verzeichnis
    root_dir = '/storage/ISLES24/ISLES24/preprocessed'


    paths_train = glob(os.path.join(root_dir, 'train', '*.npz'))
    paths_val = glob(os.path.join(root_dir, 'val', '*.npz'))
    paths = {'train': paths_train, 'val': paths_val}

    # Daten-Loader erstellen
    train_loader, val_loader= get_data_loaders(
        paths=paths,
        batch_size=2,
        batch_size_test=1,
        transform=None  # Anpassung je nach Bedarf

    )

    # Modellinitialisierung
    model = UNet(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        channels=(64,128,256,512),
        strides=(2,2,2),
    )



    # Starte Training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-4,#1e-4
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )