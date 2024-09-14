import os
import numpy as np
import matplotlib.pyplot as plt


def check_shape(npy_file):
    if npy_file.shape != (1259,300,300):
        npy_file = npy_file.T
    return npy_file

def plot_slices(img, noise, noise_only, slice_num, folder, output_path='output.png'):
    fig, axs = plt.subplots(1, 3, figsize=(8,10), sharey=True)
    axs[0].imshow(img[:, :, slice_num], cmap='seismic')
    axs[0].set_title('Image')
    axs[0].set_ylabel('Depth (m)')
    axs[1].imshow(noise[:, :, slice_num], cmap='seismic')
    axs[1].set_title('Noisy Image')
    axs[2].imshow(noise_only[:, :, slice_num], cmap='seismic')
    axs[2].set_title('Noise')
    plt.suptitle(f'Folder: {folder}', fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def rescale_volume(seismic, low=0, high=100):
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    return seismic

def load_and_preprocess_data(folder_path, low=2, high=98):
    for file in os.listdir(folder_path):
        if 'noise' in file:
            noise = np.load(os.path.join(folder_path, file), allow_pickle=True, mmap_mode='r+')
            noise = check_shape(noise)
            noise = rescale_volume(noise, low, high)
        else:
            img = np.load(os.path.join(folder_path, file), allow_pickle=True, mmap_mode='r+')
            img = check_shape(img)
            img = rescale_volume(img, low, high)
    return img, noise