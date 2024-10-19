import os
from dataset import SeismicDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

dataset = SeismicDataset(data_dir='data', train=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for (x, y) in tqdm(dataloader):
    x = x.squeeze(0)
    y = y.squeeze(0)
    print(x.shape, y.shape)
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(x[:,0,:])
    plt.subplot(122)
    plt.imshow(y[:,0,:])
    plt.savefig('sample.png')