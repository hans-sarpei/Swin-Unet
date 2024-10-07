import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F



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
