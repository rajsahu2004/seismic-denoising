import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(17)

def plot_img_and_noise(folder=42655965, base_dir='training_data'):
    folder_path = os.path.join(base_dir, str(folder))
    for file in os.listdir(folder_path):
        if 'noise' in file:
            noise = np.load(os.path.join(folder_path, file), allow_pickle=True, mmap_mode='r+')
        else:
            img = np.load(os.path.join(folder_path, file), allow_pickle=True, mmap_mode='r+')
    if img is not None and noise is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 9), sharey=True)
        section_no = np.random.randint(0, img.shape[1])    
        ax1.imshow(img[:, :, section_no], cmap='seismic')
        ax1.set_title('Image')
        ax2.imshow(noise[:, :, section_no], cmap='seismic')
        ax2.set_title('Noisy Image')
        plt.suptitle(f'Folder: {folder}, Section: {section_no}')
        plt.tight_layout()
        name = 'output.png'
        plt.savefig(name, dpi=100)
        print(f"Image saved as {name}")
    else:
        print("Image or noise not found in the folder.")