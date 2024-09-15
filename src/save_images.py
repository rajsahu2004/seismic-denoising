from torch.utils.data import DataLoader
import os
from dataset import SeismicDataset
from tqdm.auto import tqdm

# Create a DataLoader instance
dataset = SeismicDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count())

os.makedirs('images', exist_ok=True)
n = len(dataset)
for i, (x,y) in tqdm(enumerate(dataloader), total=n):
    print(x.shape, y.shape)
    dataset.plot_data_train(x, y, x_slice=68, title=f'Train Image {i}')

datatest = SeismicDataset(train=False)
dataloader = DataLoader(datatest, batch_size=1, shuffle=True, num_workers=os.cpu_count())
m = len(datatest)
for i, x in tqdm(enumerate(dataloader), total=m):
    print(x.shape)
    datatest.plot_data_test(x, x_slice=68, title=f'Test Image {i}')