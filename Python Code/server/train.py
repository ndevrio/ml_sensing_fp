import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sys
import os
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader, random_split


np.set_printoptions(precision=4)

types = ['standing', 'swaying', 'random walking', 'sitting']
classes = {
    'front_left_pocket': 0,
    'front_right_pocket': 1,
    'back_left_pocket': 2,
    'back_right_pocket': 3
}
class_names = ['front_left_pocket', 'front_right_pocket', 'back_left_pocket', 'back_right_pocket']
data_in = [[], [], [], []]

target_dir = sys.argv[1]

# Read in IMU data files from all folders
for d in os.listdir('./data/' + target_dir):
    if('.' in d):
        continue
    for f in os.listdir('./data/' + target_dir + '/' + d):
        if(f[-4:] == ".npy"):
            data_in[classes[d]].append(np.load('./data/' + target_dir + '/' + d + '/' + f))


print(len(data_in[1]))