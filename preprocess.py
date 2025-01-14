import cv2
from utils import *
from tqdm import tqdm



def generate_np_slices(subject, root_dir, type='train'):
    # create paths to load
    ses01_dir = os.path.join(root_dir, subject, 'ses-01')
    ses02_dir = os.path.join(root_dir, subject, 'ses-02')
    ses01_dir_ncct = os.path.join(os.path.split(root_dir)[0], 'raw_data', subject, 'ses-01')

    # Create paths for all modalities (cta, cbf, cbv, etc.)
    ncct_path = os.path.join(ses01_dir_ncct, f"{subject}_ses-01_ncct.nii.gz")
    perfusion_dir = os.path.join(ses01_dir, 'perfusion-maps')
    tmax_path = os.path.join(perfusion_dir, f"{subject}_ses-01_space-ncct_tmax.nii.gz")
    mask_path = os.path.join(ses02_dir, f"{subject}_ses-02_lesion-msk.nii.gz")

    # Laden der CT-Bildmodalitäten
    ncct = nib.load(ncct_path).get_fdata()
    tmax = nib.load(tmax_path).get_fdata()

    # Laden der Maske
    mask = nib.load(mask_path).get_fdata()
    mask = np.round(mask)

    '''preprocessing starts here (clip, resize, min_max)'''

    for i in range(ncct.shape[-1]):
        current_ncct_slice = ncct[..., i]
        current_tmax_slice = tmax[..., i]
        current_mask_slice = mask[..., i]


        #skip only-zero slices or slices with only negative time values
        if (current_tmax_slice.min() < 0 and current_tmax_slice.max() < 0) or not (current_mask_slice > 0).any():
            #don't save
            continue


        # (1) clipping as explored in ITK-SNAP
        ncct_slice = np.clip(current_ncct_slice, a_min=0, a_max=110)
        tmax_slice = np.clip(current_tmax_slice, a_min=0, a_max=20)

        # (2) resize (512x512)
        # Resize das Input-Bild mit bilinearer Interpolation
        resized_ncct = cv2.resize(ncct_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_tmax = cv2.resize(tmax_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Resize die Maske mit nearest-neighbor Interpolation
        resized_mask = cv2.resize(current_mask_slice, (512, 512), interpolation=cv2.INTER_NEAREST)

        # (3) min_max normalisierung
        ncct_slice = min_max_normalization(resized_ncct, clip_min=0, clip_max=110)
        tmax_slice = min_max_normalization(resized_tmax, clip_min=0, clip_max=20)

        '''save numpy arrays to disk'''
        save_folder = f'/storage/ISLES24/ISLES24/preprocessed/{type}'
        os.makedirs(save_folder, exist_ok=True)
        out_file = save_folder + f'/{subject}_slice_{i+1}.nii.gz'

        np.savez(out_file,
                 image_ncct=ncct_slice,
                 image_tmax=tmax_slice,
                 label=resized_mask,
                 sample_name=out_file)



root_dir = '/storage/ISLES24/ISLES24/derivatives'
# Liste aller Subjekte, falls
subjects = os.listdir(root_dir)

train_subjects, test_subjects = train_test_split(subjects, test_size=0.1, random_state=42)
train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.2, random_state=42)

#for train_subject in tqdm(train_subjects, total=len(train_subjects)):
#    generate_np_slices(train_subject, root_dir, 'train')
for val_subject in tqdm(val_subjects, total=len(val_subjects)):
    generate_np_slices(val_subject, root_dir, 'val')
