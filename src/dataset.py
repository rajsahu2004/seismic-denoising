import os
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from sklearn.preprocessing import MinMaxScaler

class SeismicDataset(Dataset):
    def __init__(self, data_dir='data', train=True, transform=None):
        self.train = train
        self.dir = os.path.join(data_dir, 'test_data' if not train else 'training_data')
        self.noise_paths = glob(f'{self.dir}/*/seismic_w_noise_vol_*.npy')
        self.stack_paths = None
        if self.train:
            self.stack_paths = glob(f'{self.dir}/*/seismicCubes_RFC_fullstack_*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.noise_paths)

    def check_shape(self, file):
        shape = file.shape
        if shape != (1259, 300, 300):
            return file.T
        return file
    
    def rescale(self, file, low=2, high=98):
        p_low, p_high = np.percentile(file, (low, high))
        return np.clip(file, p_low, p_high)
    
    def __getitem__(self, index):
        noise_slice = np.load(self.noise_paths[index], allow_pickle=True)
        noise_slice = self.check_shape(noise_slice)
        scale = MinMaxScaler()
        noise_slice = scale.fit_transform(noise_slice.reshape(-1, 1)).reshape(noise_slice.shape)
        noise_slice = self.rescale(noise_slice)
        if self.train:
            # Derive the corresponding full-stack (clean) file path based on the slice number
            slice_number = self.noise_paths[index].split('_')[-1].split('.')[0]  # Extract slice number
            stack_file_path = [path for path in self.stack_paths if f'{slice_number}.npy' in path][0]
            
            # Load the clean seismic data
            stack_slice = np.load(stack_file_path, allow_pickle=True)
            stack_slice = self.check_shape(stack_slice)
            scale = MinMaxScaler()
            stack_slice = scale.fit_transform(stack_slice.reshape(-1, 1)).reshape(stack_slice.shape)
            stack_slice = self.rescale(stack_slice)
            # Apply any transforms (if provided)
            if self.transform:
                noise_slice = self.transform(noise_slice)
                stack_slice = self.transform(stack_slice)

            return noise_slice, stack_slice

        # Apply transforms for test data if needed
        if self.transform:
            noise_slice = self.transform(noise_slice)

        return noise_slice