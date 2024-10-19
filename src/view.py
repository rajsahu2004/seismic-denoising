from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import SeismicDataset
import numpy as np

dataset = SeismicDataset(data_dir='data', train=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
slice_num = np.random.randint(0, 300)

for i, (x, y) in enumerate(tqdm(dataloader)):
    x = x.squeeze(0)
    y = y.squeeze(0)
    print(x[:,slice_num,:][0,:15])
    print(y[:,slice_num,:][0,:15])
    plt.figure(figsize=(5, 8))
    plt.subplot(121)
    plt.imshow(x[:,slice_num,:], cmap='gray')
    plt.subplot(122)
    plt.imshow(y[:,slice_num,:], cmap='gray')
    plt.suptitle(f'[{i+1}/{len(dataloader)}] Slice number: {slice_num}')
    plt.tight_layout()
    plt.savefig('img.png')
    plt.show()