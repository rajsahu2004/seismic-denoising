import os
import numpy as np
import matplotlib.pyplot as plt


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
