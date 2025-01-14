import os

import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from glob import glob
from collections import Counter
from sklearn.model_selection import train_test_split
import nibabel as nib

def save_model(model, epoch, path='best_model.pth'):
    """Speichert das Modell."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, path)


def plot_images(images, predictions, targets, epoch, batch, loss,num_images=5):
    """Plotting function for images, predictions, and targets.
    images shape = (1, 5, 512, 512)
    """
    fig, axs = plt.subplots(num_images, 4, figsize=(15, 5 * num_images))
    fig.text(0.01, 0.5, f"Epoch:{epoch}, Batch:{batch}, Loss:{loss}", va='center', ha='center', rotation='vertical', fontsize=12)
    for i in range(num_images):
        # Eingabebild: ncct und tmax Bild
        axs[i, 0].imshow(images[i][0].cpu().numpy().squeeze(), cmap='gray')  # ncct Bild
        axs[i, 0].set_title("NCCT Input")
        axs[i, 1].imshow(images[i][1].cpu().numpy().squeeze(), cmap='gray')  # tmax Bild
        #axs[i, 1].imshow(images[i][0].cpu().numpy().squeeze(), cmap='gray')  # tmax Bild
        axs[i, 1].set_title("TMax Input")
        axs[i, 2].imshow(predictions[i].cpu().numpy().squeeze(), cmap='gray')  # prediction map
        axs[i, 2].set_title("Prediction")
        axs[i, 3].imshow(targets[i].cpu().numpy().squeeze(), cmap='gray')  # target map
        axs[i, 3].set_title("Target")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()

    # Speichere die Abbildung statt sie anzuzeigen
    # Erstelle einen dynamischen Dateinamen basierend auf Epoch und Batch
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"graph_epoch_{epoch}_batch_{batch}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # dpi kann für die Auflösung angepasst werden

    # Optional: Schließe die Abbildung, um Speicher freizugeben
    plt.close(fig)




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


def min_max_normalization(data, clip_min, clip_max):
    min_val = clip_min #np.min(data)
    max_val = clip_max #np.max(data)
    if min_val == max_val:
        normalized_data = (data - min_val + 1e-8) / (max_val - min_val + 1e-8)
    else:
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


def compute_ranges_of_volumes(vol_paths):
    min_x = 5000
    min_y = 5000
    min_z = 5000
    max_x = 0
    max_y = 0
    max_z = 0
    for vol_path in vol_paths:
        cur_vol = nib.load(vol_path).get_fdata()
        min_x = cur_vol.shape[0] if cur_vol.shape[0] < min_x else min_x
        min_y = cur_vol.shape[1] if cur_vol.shape[1] < min_y else min_y
        min_z = cur_vol.shape[2] if cur_vol.shape[2] < min_z else min_z
        max_x = cur_vol.shape[0] if cur_vol.shape[0] > max_x else max_x
        max_y = cur_vol.shape[1] if cur_vol.shape[1] > max_y else max_y
        max_z = cur_vol.shape[2] if cur_vol.shape[2] > max_z else max_z

    return (min_x, min_y, min_z), (max_x, max_y, max_z)

#remote paths liegen nicht in ./data_/ ->
all_paths = glob("/storage/ISLES24/**/*.gz", recursive=True)
msk_paths = list(filter(lambda k: 'msk' in k, all_paths))
#min_coords, max_coords = compute_ranges_of_volumes(msk_paths)


def compute_class_weights(targets, num_classes):
    """
    Berechnet die Gewichte für jede Klasse basierend auf der inversen Häufigkeit.

    Args:
        targets (List[torch.Tensor]): Liste von Ground-Truth-Masken (B x H x W).
        num_classes (int): Anzahl der Klassen.

    Returns:
        torch.Tensor: Tensor mit den Gewichten für jede Klasse.
    """
    class_counts = Counter()

    for target in targets:
        # Flatten das Ziel, um die Häufigkeit zu zählen
        class_counts.update(target.flatten().tolist())

    # Berechnung der Frequenz jeder Klasse
    class_freq = np.array([class_counts.get(i) for i in range(num_classes)], dtype=np.float32)

    # Vermeidung von Division durch Null - nur zur Sicherheit
    class_freq = np.where(class_freq == 0, 1, class_freq)

    # Inverse Häufigkeit als Gewicht
    class_weights = 1.0 / class_freq

    # Normalisierung der Gewichte (optional) -> um zu verhindern dass es zu große class_weights Unterschiede auf Skala gibt
    class_weights = class_weights / class_weights.sum() * num_classes

    return torch.tensor(class_weights)


def build_a_full_target_dataset_tensor(mask_paths):
    target_list = []
    for mask_path in mask_paths:
        mask = nib.load(mask_path).get_fdata()
        mask = torch.from_numpy(np.round(mask))
        target_list.append(mask)
    return target_list


root_dir = '/storage/ISLES24/ISLES24/derivatives'
# Liste aller Subjekte, falls
subjects = os.listdir(root_dir)

train_subjects, test_subjects = train_test_split(subjects, test_size=0.1, random_state=42)
train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.2, random_state=42)

#local paths            paths_scans = glob("./data_/**/*.gz", recursive=True)
paths_scans = glob("/storage/ISLES24/**/*.gz", recursive=True) #remote paths
#mask_paths = list(filter(lambda k: 'msk' in k and any(name in k for name in train_subjects), paths_scans)) #val_subjects
mask_paths = list(filter(lambda k: 'msk' in k , paths_scans))

target_list = build_a_full_target_dataset_tensor(mask_paths)


num_classes = 2
class_weights = compute_class_weights(target_list, num_classes)
print(f'Klassengewichte: {class_weights}') #Klassengewichte: tensor([0.0058, 1.9942])