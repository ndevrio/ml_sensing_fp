import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sys
import os
import pandas as pd
from tqdm import tqdm

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
samp_rate = 60
num_win_samples = 128
window_size = int(num_win_samples * (samp_rate/60))
if((window_size % 2) == 1):
    window_size += 1

target_dir = sys.argv[1]

# Read in IMU data files from all folders
for d in os.listdir('./data/' + target_dir):
    if('.' in d):
        continue
    for f in os.listdir('./data/' + target_dir + '/' + d):
        if(f[-4:] == ".npy"):
            z = np.load('./data/' + target_dir + '/' + d + '/' + f)
            z = z[~np.isnan(z).any(axis=1)] # remove rows that contain NaN
            data_in[classes[d]].append(z)


raw_data = []
num_windows = 0
overlap = 0.5
for i in range(len(classes)):
    if len(data_in[i]) > 0:
        t = np.vstack(data_in[i])[:, 2:]
        num_windows += int(t.shape[0] * (1/overlap) / window_size)-1
        raw_data.append(t)


processed_data = np.zeros((num_windows, window_size, 9))
processed_labels = np.zeros((num_windows,))
def preprocessing():
    global processed_data, processed_labels
    p_idx = 0

    for i in range(len(raw_data)):
        d = np.zeros((raw_data[i].shape[0], 9))
        d = raw_data[i][:, :-4]
        """for j in range(6):        
            # [1] Noise filtering
            # Median filter
            z = d[:, j]
            z = signal.medfilt(z, kernel_size=5)
            # 3rd order lowpass butterworth filter to remove noise
            sos = signal.butter(3, 20, btype='lowpass', output='sos', fs=100)
            z = signal.sosfilt(sos, z)
            if(j < 3):
                # [2] Separate accelerometer data into graviational (total) and body motion components
                #     Use a 3rd order highpass butterworth filter to remove the gravity component
                sos = signal.butter(3, 0.3, btype='highpass', output='sos', fs=100)
                sos_grav = signal.butter(3, 0.3, btype='lowpass', output='sos', fs=100)
                z_nograv = signal.sosfilt(sos, z)
                z_grav = signal.sosfilt(sos_grav, z)
                d[:, j] = z_nograv
                d[:, j+6] = z_grav"""

        # [3] Split continuous data into windows with 50% overlap
        processed_d = np.zeros((int(d.shape[0]*(1/overlap)/window_size)-int((1/overlap)-1), window_size, 9))
        for k in range(len(processed_d)):
            processed_d[k] = d[int(k*overlap*window_size):int(k*overlap*window_size)+window_size] 
        
        # [4] Reformat processed array data to input into the model
        processed_data[p_idx:p_idx+processed_d.shape[0]] = processed_d.copy()
        processed_labels[p_idx:p_idx+processed_d.shape[0]] = i
        p_idx += processed_d.shape[0]


class Motion1DCNN(nn.Module):
    def __init__(self):
        super(Motion1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32*window_size, 100)
        self.fc2 = nn.Linear(100, len(class_names))
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)

        x = self.fc1(x.view(x.size(0),-1))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def main():
    ########################################
    ###  Process and featurize the data  ###
    ########################################
    preprocessing()

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    lst_accu_stratified = np.zeros((n_splits,))
    cmat = np.zeros((n_splits, len(class_names), len(class_names)))
    s = 0

    torch.set_printoptions(sci_mode=False)

    for train_index, test_index in skf.split(processed_data, processed_labels):
        print("============== Split " + str(s) + " ==============")
        tensor_data = torch.Tensor(np.moveaxis(processed_data, 1, -1))
        tensor_labels = torch.Tensor(processed_labels).type(torch.LongTensor)

        x_train_fold, x_test_fold = tensor_data[train_index], tensor_data[test_index]
        y_train_fold, y_test_fold = tensor_labels[train_index], tensor_labels[test_index]

        # Set of datasets and dataloaders
        train_dataset = TensorDataset(x_train_fold, y_train_fold)
        test_dataset = TensorDataset(x_test_fold, y_test_fold)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                            shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                            shuffle=False, num_workers=2)

        # Set up model and torch properties
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Motion1DCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        num_epoch = 20

        ########################
        ###  Model training  ###
        ########################
        print("Start training...")
        model.train() # Set the model to training mode
        for i in range(num_epoch):
            running_loss = []
            for batch, label in tqdm(train_loader):
                batch = batch.to(device)
                label = label.to(device)
                optimizer.zero_grad() # Clear gradients from the previous iteration
                pred = model(batch) # This will call Network.forward() that you implement
                loss = criterion(pred, label) # Calculate the loss
                running_loss.append(loss.item())
                loss.backward() # Backprop gradients to all tensors in the network
                optimizer.step() # Update trainable weights
            print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
        print("Done!")

        ##########################
        ###  Model evaluation  ###
        ##########################
        model.eval() # Set the model to evaluation mode
        correct = 0
        class_correct = np.zeros((len(class_names),len(class_names)))
        num_each_class = np.zeros((len(class_names),))
        with torch.no_grad(): # Do not calculate grident to speed up computation
            for batch, label in tqdm(test_loader):
                batch = batch.to(device)
                label = label.to(device)
                pred = F.softmax(model(batch), dim=1)
                pred = torch.argmax(pred,dim=1)
                correct += (pred==label).sum().item()

                for i in range(len(class_names)):
                    samp = (label == i)
                    for j in range(len(class_names)):
                        class_correct[i, j] += (pred[samp]==j).sum().item()
                    num_each_class[i] += len(label[samp])

        acc = correct/len(test_loader.dataset)
        lst_accu_stratified[s] = acc
        print("Evaluation accuracy: {}".format(acc))
        for i in range(len(class_names)):
            class_correct[i] = class_correct[i] / num_each_class[i]
        cmat[s] = class_correct

        s += 1

        break

    print("============== Final average ==============")
    print(lst_accu_stratified)
    print("============== Confusion matrix ==============")
    print(np.mean(cmat, axis=0))

    print("============== Saving model ==============")
    torch.save(model.state_dict(), 'models/best_' + target_dir)
    print("Saved.")


if __name__ == "__main__":
    main()