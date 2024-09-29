import os
import numpy as np
from glob import glob

def check_shape(self, npy_file):
        if npy_file.shape != (1259, 300, 300):
            npy_file = npy_file.T
        return npy_file

def slice_and_save(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    npy_files = sorted(glob(f'{data_dir}/*.npy'))

    for npy_file in npy_files:
        volume = np.load(npy_file)
        if volume.shape != (1259, 300, 300):
            print(f"Skipping {npy_file} due to incorrect shape {volume.shape}.")
            continue
        base_filename = os.path.basename(npy_file).replace('.npy', '')
        for i in range(volume.shape[0]):
            slice_2d = volume[i, :, :]
            slice_filename = f'{base_filename}_SLICE_{i}.npy'
            np.save(os.path.join(save_dir, slice_filename), slice_2d)

        print(f"Finished slicing and saving {npy_file}.")


data_dir = 'data/training_data'
save_dir = 'data/slice_training_data'