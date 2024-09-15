import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob

class SeismicDataset(Dataset):
    def __init__(self, data_dir='data', train=True, transform=None, low=2, high=98):
        self.train = train
        self.dir = os.path.join(data_dir, 'test_data')
        self.img = glob(f'{self.dir}/*/*.npy')
        if self.train:
            self.dir = os.path.join(data_dir, 'training_data')
            self.label = glob(f'{self.dir}/*/*stack*.npy')
            self.img = glob(f'{self.dir}/*/*noise*.npy')
        self.transform = transform
        self.low = low
        self.high = high
        
    def __len__(self):
        return len(self.img)
    
    def check_shape(self, npy_file):
        if npy_file.shape != (1259, 300, 300):
            npy_file = npy_file.T
        return npy_file
    
    def rescale_volume(self, seismic, low, high):
        minval = np.percentile(seismic, low)
        maxval = np.percentile(seismic, high)
        seismic = np.clip(seismic, minval, maxval)
        seismic = ((seismic - minval) / (maxval - minval)) * 255
        return seismic
    
    def __getitem__(self, index):
        assert index < len(self.img), f'Index {index} out of bounds'
        self.x = np.load(self.img[index], allow_pickle=True, mmap_mode='r+')
        self.x = self.check_shape(self.x)
        self.x = self.rescale_volume(self.x, self.low, self.high)
        
        if self.train:
            self.y = np.load(self.label[index], allow_pickle=True, mmap_mode='r+')
            self.y = self.check_shape(self.y)
            self.y = self.rescale_volume(self.y, self.low, self.high)
            
            if self.transform:
                self.x = self.transform(self.x)
                self.y = self.transform(self.y)
            return torch.tensor(self.x).float(), torch.tensor(self.y).float()
        
        if self.transform:
            self.x = self.transform(self.x)
        return torch.tensor(self.x).float()