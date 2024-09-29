import os
import numpy as np
from glob import glob
from tqdm.auto import tqdm

def check_shape(npy_file):
        if npy_file.shape != (1259, 300, 300):
            npy_file = npy_file.T
        return npy_file

def get_slices(filepath):
    file = np.load(filepath, allow_pickle=True)
    file = check_shape(file)
    num_slices = file.shape[0]
    folder = filepath.split('/')
    folder = '/'.join(folder[:-1])
    filename = filepath.split('/')[-1].split('.npy')[0]
    for i in range(num_slices):
        slice = file[i]
        save_path = f'{folder}/{filename}_{i}.npy'
        np.save(save_path, slice)
    os.remove(filepath)
    


files = glob('data/training_data/*/*.npy')
for file in tqdm(files):
    get_slices(file)