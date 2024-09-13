import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from data import SeismicDataset

training_path = 'data/training_data'
train = [os.path.join(training_path, path) for path in os.listdir(training_path)]
folder = np.random.choice(train)
