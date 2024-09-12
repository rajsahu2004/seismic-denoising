import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def check_shape(npy_file):
    if npy_file.shape != (1259,300,300):
        npy_file = npy_file.T
    return npy_file

training_path = 'data/training_data'
train = os.listdir(training_path)
folder = np.random.choice(train)
for file in os.listdir(os.path.join(training_path, folder)):
    if 'noise' in file:
        noise = np.load(os.path.join(training_path, folder, file), allow_pickle=True, mmap_mode='r+')
        noise = check_shape(noise)
    else:
        img = np.load(os.path.join(training_path, folder, file), allow_pickle=True, mmap_mode='r+')
        img = check_shape(img)

noise_only = noise - img

fig, axs = plt.subplots(1, 3, figsize=(8, 10), sharey=True)

# Plot the original image
axs[0].imshow(img[:, :, 0], cmap='seismic')
axs[0].set_title('Image')
axs[0].set_ylabel('Depth (m)')

# Plot the noisy image
axs[1].imshow(noise[:, :, 0], cmap='seismic')
axs[1].set_title('Noisy Image')

# Plot the noise
axs[2].imshow(noise_only[:, :, 0], cmap='seismic')
axs[2].set_title('Noise')

# Add an overall title and adjust the layout
plt.suptitle(f'Folder: {folder}', fontsize=18)
plt.tight_layout()

# Save the figure
plt.savefig('output.png')