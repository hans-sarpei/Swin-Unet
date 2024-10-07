from glob import glob
import os
import nibabel as nib
from tqdm import tqdm
import numpy as np

root_dir = os.path.join(os.getcwd(), 'data_', 'derivatives')

subjects = os.listdir(root_dir)
all_voxel_sizes = []

for subject in tqdm(subjects, total=len(subjects)):

    ses01_dir = os.path.join(root_dir, subject, 'ses-01')

    # Bildmodalitäten in ses-01
    cta_path = os.path.join(ses01_dir, f"{subject}_ses-01_space-ncct_cta.nii.gz")

    #get physical voxel_sizes for each subject
    (cta_h, cta_w, cta_d) = nib.load(cta_path).header.get_zooms() #in mm


    voxel_sizes_cta = (cta_h, cta_w, cta_d)

    all_voxel_sizes.append(voxel_sizes_cta)



    with open('voxel_sizes.txt', 'a') as f:
        f.write(f"Subject {subject} hat Voxel-Groeße (in mm): {voxel_sizes_cta}\n")

all_voxel_sizes = np.array(all_voxel_sizes)
min_h = np.min(all_voxel_sizes[:,0])
max_h = np.max(all_voxel_sizes[:,0])
min_w = np.min(all_voxel_sizes[:,1])
max_w = np.max(all_voxel_sizes[:,1])
min_d = np.min(all_voxel_sizes[:,2])
max_d = np.max(all_voxel_sizes[:,2])

boundaries_h = (min_h, max_h)
boundaries_w = (min_w, max_w)
boundaries_d = (min_d, max_d)

with open('voxel_sizes.txt', 'a') as f:
    f.write(f'Bounderies for h-dimension: {boundaries_h}\n')
    f.write(f'Bounderies for w-dimension: {boundaries_w}\n')
    f.write(f'Bounderies for d-dimension: {boundaries_d}\n')


