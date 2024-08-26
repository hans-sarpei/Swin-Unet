from glob import glob
import os
from dotenv import load_dotenv
import nibabel as nib
import json
import numpy as np
import h5py
from tqdm import tqdm
from monai import transforms


paths = glob("./data_/**/*.gz", recursive=True)
#print(nib.load(paths[0]).get_fdata())

def generate_np_slices(paths, vol=False):
    raw_ncct_paths = list(filter(lambda k: 'raw_data' in k and 'perfusion-maps' not in k and 'ncct' in k, paths))
    mask_paths = list(filter(lambda k: 'msk' in k, paths))

    for i in tqdm(range(len(raw_ncct_paths)), total=len(raw_ncct_paths)):
        ncct_vol = nib.load(raw_ncct_paths[i]).get_fdata()
        mask_vol = nib.load(mask_paths[i]).get_fdata()
        if not vol:
            for j in range(ncct_vol.shape[-1]):
                ncct_slice = ncct_vol[:, :, j]
                mask_slice = mask_vol[:, :, j]

                original_slice = np.copy(ncct_slice)
                #preprocessing clipping + normalization (0,1) -> can't use clipping cause there are  2d slices with only negative values
                ncct_slice = np.clip(ncct_slice, a_min=0, a_max=80) #or use a_min=20HU, and a_max=40HU to focus on lesion  characteristics ->potentiell werden dadurch noch weitere Slices weggeworfen

                # Step 1: Check if all pixel values are the same
                if np.all(ncct_slice == ncct_slice[0,0]):
                    continue
                    # Handle the case where all values are the same
                    #ncct_slice_norm = np.full_like(ncct_slice, 0.5)  # or np.ones_like(x) or np.zero_like(x)
                else:
                    min_val_slice = np.min(ncct_slice)

                    # Step 2: Shift the values to make the minimum value zero
                    x_shifted = ncct_slice - min_val_slice

                    max_val_shifted = np.max(x_shifted)
                    # Step 3: Normalize to the range (0, 1)
                    ncct_slice_norm = x_shifted / max_val_shifted


                out_filename = os.path.split(raw_ncct_paths[i])[1].replace("_ses-01_ncct.nii.gz", "")
                main_dir = "./data_preprocessed/Isles/train_npz/"
                out_file = f"{main_dir}{out_filename}_slice_{j}"


                if not os.path.exists(main_dir):
                    os.makedirs(main_dir)
                np.savez(out_file, image=ncct_slice_norm, label=mask_slice, original_slice=original_slice, sample_name=out_file)
        elif vol:
            mins_per_slice = np.min(ncct_vol, axis=(0, 1), keepdims=True)
            x_shifted = ncct_vol - mins_per_slice
            maxs_per_x_shifted = np.max(x_shifted, axis=(0, 1), keepdims=True)
            slices_to_fill_narray = np.argwhere(maxs_per_x_shifted == 0)[:, 2]
            #neutralize dividing for same-valued slices by setting it to 1
            max_per_x_shifted = np.where(maxs_per_x_shifted == 0, np.full_like(maxs_per_x_shifted, 1), maxs_per_x_shifted)


            ncct_vol_norm = x_shifted / max_per_x_shifted

            # setting values to 0.5 for same-valued slices
            ncct_vol_norm[:, :, slices_to_fill_narray] = 0.5

            out_filename = os.path.split(raw_ncct_paths[i])[1].replace("_ses-01_ncct.nii.gz", "")
            main_dir = "./data_preprocessed/Isles/test_vol_h5/"
            out_file = f"{main_dir}{out_filename}.npy.h5"

            if not os.path.exists(main_dir):
                os.makedirs(main_dir)
            # Save the arrays in an HDF5 file with key-value pairs
            with h5py.File(out_file, 'w') as h5file:
                h5file.create_dataset('image', data=ncct_vol_norm)
                h5file.create_dataset('label', data=mask_vol)



def save_np_slices_names(paths, vol=False):

    file_names = [os.path.split(path)[-1].replace(".npz", "") for path in paths]
    file_names = sorted(file_names, key=lambda x: (int(x.split('_')[0][-4:]), int(x.split('_')[-1])))

    if not vol:
        txt_file = "train.txt"
    else:
        txt_file = "test_vol.txt"
    # write file_names to list
    with open(f'./lists/lists_Isles/{txt_file}', 'w') as f:
        for file_name in file_names:
            f.write(f"{file_name}\n")





paths_test= glob("./data_test/**/*.gz", recursive=True)
paths_reduced = glob("./data_preprocessed/Isles/train_npz/*.npz", recursive=True)

#generate_np_slices(paths, vol=False)
generate_np_slices(paths_test, vol=True)
#save_np_slices_names(paths_test, vol=True)
#save_np_slices_names(paths_reduced)

def find_class_weights(paths):
    class_weights = []
    mask_paths = list(filter(lambda k: 'msk' in k, paths))

    samples = {'non_les': np.int64(0), 'les': np.int64(0),'all': np.int64(0)}
    for i in tqdm(range(len(mask_paths)), total=len(mask_paths)):
        mask_vol = nib.load(mask_paths[i]).get_fdata()



        non_les_class = np.int64((np.round(mask_vol) == 0).sum())
        assert non_les_class >= 0

        les_class = np.int64((np.round(mask_vol) == 1).sum())
        assert les_class >= 0

        all_samples = np.int64(mask_vol.size)
        assert all_samples >= 0

        assert non_les_class + les_class == all_samples

        samples['non_les'] += non_les_class
        samples['les'] += les_class
        samples['all'] += all_samples

    # Calculate class weights
    assert samples['non_les'] >=0 & samples['les'] >= 0 & samples['all'] >= 0

    total_samples = samples['all']
    weight_non_les = total_samples / (2 * samples['non_les'])
    weight_les = total_samples / (2 * samples['les'])
    return (weight_non_les, weight_les)

#w0,w1 = find_class_weights(paths)
#print(w0, w1)

