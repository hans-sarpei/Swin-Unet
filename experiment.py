import torch
from collections import Counter
import numpy as np
from glob import glob
import nibabel as nib

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

    # Gesamtzahl der Pixel
    total_pixels = sum(class_counts.values())

    # Berechnung der Frequenz jeder Klasse
    class_freq = np.array([class_counts.get(i) for i in range(num_classes)], dtype=np.float32)

    # Vermeidung von Division durch Null
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
        mask = torch.from_numpy(mask)
        target_list.append(mask)
    return target_list



paths_scans = glob("./data_/**/*.gz", recursive=True)
mask_paths = list(filter(lambda k: 'msk' in k, paths_scans))

target_list = build_a_full_target_dataset_tensor(mask_paths)


num_classes = 2
class_weights = compute_class_weights(target_list, num_classes)
print(f'Klassengewichte: {class_weights}') #Klassengewichte: tensor([0.0058, 1.9942])
