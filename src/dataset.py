import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision import transforms

class SeismicDataset(Dataset):
    def __init__(self, data_dir='data', train=True, transform=None, time=False):
        self.train = train
        self.dir = os.path.join(data_dir, 'test_data' if not train else 'training_data')
        self.img_paths = sorted(glob(f'{self.dir}/*/*noise*.npy'))
        self.label_paths = None
        if self.train:
            self.label_paths = sorted(glob(f'{os.path.join(data_dir, "training_data")}/*/*stack*.npy'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def check_shape(self, npy_file):
        if npy_file.shape != (1259, 300, 300):
            npy_file = npy_file.T
        return npy_file



    def __getitem__(self, index):
        assert index < len(self.img_paths), f'Index {index} out of bounds'
        self.x = np.load(self.img_paths[index], allow_pickle=True, mmap_mode='r+')
        self.x = self.check_shape(self.x)

        if self.train:
            self.y = np.load(self.label_paths[index], allow_pickle=True, mmap_mode='r+')
            self.y = self.check_shape(self.y)
        self.x = Rescale()(self.x)
        if self.train:
            self.y = Rescale()(self.y)
        if self.transform:
            self.x = self.transform(self.x)
            if self.train:
                self.y = self.transform(self.y)

        return (self.x, self.y) if self.train else self.x
    
    def plot_data_train(self, img, label, title='Seismic Data', x_slice='all', y_slice='all'):
        if img.shape != (1259, 300, 300):
            img = torch.squeeze(img)
        if label.shape != (1259, 300, 300):
            label = torch.squeeze(label)

        if x_slice == 'all' and y_slice == 'all':
            print('Plotting entire volume is not possible, select a slice')
            return
        elif x_slice == 'all' and y_slice != 'all':
            img = img[:, :, y_slice]
            label = label[:, :, y_slice]
        elif x_slice != 'all' and y_slice == 'all':
            img = img[:, x_slice, :]
            label = label[:, x_slice, :]
        if label is not None:
            fig, axs = plt.subplots(1, 2, figsize=(5, 9))
            axs[0].imshow(img, cmap='seismic')
            axs[0].set_title('Seismic Image')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Depth (ms)')

            axs[1].imshow(label, cmap='seismic')
            axs[1].set_title('Label')
            axs[1].set_xlabel('X')
            plt.suptitle(f'{title} (X Slice: {x_slice}, Y Slice: {y_slice})')
            plt.tight_layout()
            plt.savefig(f'images/{title}_{x_slice}_{y_slice}.png')
            plt.close()

    def plot_data_test(self, img, title='Seismic Data', x_slice='all', y_slice='all'):
        if img.shape != (1259, 300, 300):
            img = torch.squeeze(img)

        if x_slice == 'all' and y_slice == 'all':
            print('Plotting entire volume is not possible, select a slice')
            return
        elif x_slice == 'all' and y_slice != 'all':
            img = img[:, :, y_slice]
        elif x_slice != 'all' and y_slice == 'all':
            img = img[:, x_slice, :]
        fig, ax = plt.subplots(1, 1, figsize=(3, 9))
        ax.imshow(img, cmap='seismic')
        ax.set_title('Seismic Image')
        ax.set_xlabel('X')
        ax.set_ylabel('Depth (ms)')
        plt.suptitle(f'{title} (X Slice: {x_slice}, Y Slice: {y_slice})')
        plt.tight_layout()
        plt.savefig(f'images/{title}_{x_slice}_{y_slice}.png')
        plt.close()

class Rescale(transforms.Lambda):
    def __init__(self, low=1, high=99):
        super().__init__(self.rescale)
        self.low = low
        self.high = high

    def rescale(self, seismic):
        minval = np.percentile(seismic, self.low)
        maxval = np.percentile(seismic, self.high)
        seismic = np.clip(seismic, minval, maxval)
        seismic = ((seismic - minval) / (maxval - minval)) * 255

        return seismic