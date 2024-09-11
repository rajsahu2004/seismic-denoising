import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def check_shape(npy_file):
    if npy_file.shape != (1259,300,300):
        npy_file = npy_file.T
    return npy_file
training_path = 'training_data'
train = os.listdir(training_path)
folder = np.random.choice(train)
for file in os.listdir(os.path.join(training_path, folder)):
    if 'noise' in file:
        noise = np.load(os.path.join(training_path, folder, file), allow_pickle=True, mmap_mode='r+')
        noise = check_shape(noise)
    else:
        img = np.load(os.path.join(training_path, folder, file), allow_pickle=True, mmap_mode='r+')
        img = check_shape(img)

fig = plt.figure(figsize=(5, 10), dpi=100)
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,0], cmap='seismic')
plt.title('Image')
plt.subplot(1, 2, 2)
plt.imshow(noise[:,:,0], cmap='seismic')
plt.title('Noisy Image')
plt.suptitle(f'Folder: {folder}')
plt.tight_layout()
plt.savefig('output.png')