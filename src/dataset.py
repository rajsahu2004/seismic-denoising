import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms

class SeismicDataset(Dataset):
    def __init__(self, data_dir='data', train=True, transform=None):
        self.train = train
        self.dir = os.path.join(data_dir, 'test_data' if not train else 'training_data')
        self.noise_paths = glob(f'{self.dir}/*/seismic_w_noise_vol_*.npy')  # Noisy files
        self.stack_paths = None
        if self.train:
            self.stack_paths = glob(f'{self.dir}/*/seismicCubes_RFC_fullstack_*.npy')  # Clean files
        self.transform = transform

    def __len__(self):
        # Return the number of noisy slices
        return len(self.noise_paths)

    def __getitem__(self, index):
        # Load the noisy seismic data
        noise_slice = np.load(self.noise_paths[index], allow_pickle=True)
        noise_slice = self.check_shape(noise_slice)
        print(noise_slice.shape)
        if self.train:
            # Derive the corresponding full-stack (clean) file path based on the slice number
            slice_number = self.noise_paths[index].split('_')[-1].split('.')[0]  # Extract slice number
            stack_file_path = [path for path in self.stack_paths if f'slice_{slice_number}.npy' in path][0]
            
            # Load the clean seismic data
            stack_slice = np.load(stack_file_path, allow_pickle=True)
            stack_slice = self.check_shape(stack_slice)
            print(stack_slice.shape)
            # Apply any transforms (if provided)
            if self.transform:
                noise_slice = self.transform(noise_slice)
                stack_slice = self.transform(stack_slice)

            return noise_slice, stack_slice

        # Apply transforms for test data if needed
        if self.transform:
            noise_slice = self.transform(noise_slice)

        return noise_slice

# Example usage
seismic_dataset = SeismicDataset()
print(len(seismic_dataset))  # Check dataset length
print(len(seismic_dataset[0]))  # Retrieve the first pair of noisy and clean seismic slices
plt.subplot(121)
plt.imshow(seismic_dataset[0][0])
plt.subplot(122)
plt.imshow(seismic_dataset[0][1])
plt.savefig('a.png')