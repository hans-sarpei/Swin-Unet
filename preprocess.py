import os
from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np
import cv2
from utils import *
from tqdm import tqdm


def generate_np_slices(subject, root_dir, type='train'):
    # create paths to load
    ses01_dir = os.path.join(root_dir, subject, 'ses-01')
    ses02_dir = os.path.join(root_dir, subject, 'ses-02')

    # Create paths for all modalities (cta, cbf, cbv, etc.)
    cta_path = os.path.join(ses01_dir, f"{subject}_ses-01_space-ncct_cta.nii.gz")
    perfusion_dir = os.path.join(ses01_dir, 'perfusion-maps')
    cbf_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_cbf.nii.gz")
    cbv_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_cbv.nii.gz")
    mtt_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_mtt.nii.gz")
    tmax_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_tmax.nii.gz")
    mask_path = os.path.join(ses02_dir, f"{subject}_ses-02_lesion-msk.nii.gz")

    # Laden der CT-Bildmodalitäten
    cta = nib.load(cta_path).get_fdata()
    cbf = nib.load(cbf_path).get_fdata()
    cbv = nib.load(cbv_path).get_fdata()
    mtt = nib.load(mtt_path).get_fdata()
    tmax = nib.load(tmax_path).get_fdata()

    # Laden der Maske
    mask = nib.load(mask_path).get_fdata()

    '''preprocessing starts here (lip, resize, min_max)'''

    for i in range(cta.shape[-1]):
        current_cta_slice = cta[..., i]
        current_cbf_slice = cbf[..., i]
        current_cbv_slice = cbv[..., i]
        current_mtt_slice = mtt[..., i]
        current_tmax_slice = tmax[..., i]
        current_mask_slice = mask[..., i]

        # (1) clipping
        cta_slice = np.clip(current_cta_slice, a_min=-100, a_max=100)
        cbf_slice = np.clip(current_cbf_slice, a_min=0, a_max=100)
        cbv_slice = np.clip(current_cbv_slice, a_min=0, a_max=10)
        mtt_slice = np.clip(current_mtt_slice, a_min=0, a_max=20)
        tmax_slice = np.clip(current_tmax_slice, a_min=0, a_max=15)

        # (2) resize (512x512)
        # Resize das Bild mit bilinearer Interpolation
        resized_cta = cv2.resize(cta_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_cbf = cv2.resize(cbf_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_cbv = cv2.resize(cbv_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_mtt = cv2.resize(mtt_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_tmax = cv2.resize(tmax_slice, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Resize die Maske mit nearest-neighbor Interpolation
        resized_mask = cv2.resize(current_mask_slice, (512, 512), interpolation=cv2.INTER_NEAREST)

        # (3) min_max normalisierung
        cta_slice = min_max_normalization(resized_cta)
        cbf_slice = min_max_normalization(resized_cbf)
        cbv_slice = min_max_normalization(resized_cbv)
        mtt_slice = min_max_normalization(resized_mtt)
        tmax_slice = min_max_normalization(resized_tmax)

        '''save numpy arrays to disk'''


        save_folder = f'/storage/ISLES24/ISLES24/preprocessed/{type}'
        os.makedirs(save_folder, exist_ok=True)
        out_file = save_folder + f'/{subject}_slice_{i+1}.nii.gz'

        np.savez(out_file,
                 image_cta=cta_slice,
                 image_cbf=cbf_slice,
                 image_cbv=cbv_slice,
                 image_mtt=mtt_slice,
                 image_tmax=tmax_slice,
                 label=resized_mask,
                 sample_name=out_file)







root_dir = '/storage/ISLES24/ISLES24/derivatives'
# Liste aller Subjekte, falls
subjects = os.listdir(root_dir)

train_subjects, test_subjects = train_test_split(subjects, test_size=0.1, random_state=42)
train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.2, random_state=42)

#for train_subject in train_subjects:
#    generate_np_slices(train_subject, root_dir, 'train')
for val_subject in tqdm(val_subjects, total=len(val_subjects)):
    generate_np_slices(val_subject, root_dir, 'val')
