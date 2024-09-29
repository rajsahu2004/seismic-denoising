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
    for i in tqdm(range(file.shape[0]),total=file.shape[0],desc=f'Slicing {filepath.split("/")[-1]}'):
        slice_name = os.path.join(filepath.split('.npy')[0])
        slice_name = slice_name + '_slice_' + str(i+1) + '.npy'
        np.save(slice_name, file[i])
    os.remove(filepath)


files = glob('data/training_data/*/*.npy')
for file in files:
    get_slices(file)