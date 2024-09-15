from dataset import SeismicDataset
from glob import glob
from utils import *
folder = glob('data/training_data/*')[40]
data = SeismicDataset(data_dir='data', train=True)[40]
img = data[0].numpy()
noise = data[1].numpy()
noise_only = noise - img

plot_slices(img, noise, noise_only, 100, folder.split('/')[-1], 'output.png')
