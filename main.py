import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as FT

import argparse
import numpy as np
import os
from scipy.sparse import dok_matrix

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

parser = argparse.ArgumentParser(description='MFModel')
parser.add_argument('-i', type=str, dest="dataPath")
parser.add_argument('-m', type=str, dest="method")
parser.add_argument('-MT', type=str, dest="MFPath")
parser.add_argument('-s', type=str, dest="modelPath")
argvs = parser.parse_args()

""" Set hyper-parameters """
file_path = argvs.dataPath
method = argvs.method
model_path = argvs.modelPath
batch_size = 4096
n_epoch = 50
lr = 0.001
weight_decay = 0.01 if method == 'BPR' else 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

""" Load the dataset """
print(f'loading data from {file_path}...')
from loadData import loadData, BPRDataset, BCEDataset
train_data, train_count, val_data, val_count, MF, total_negative = loadData(file_path, argvs.MFPath, method=method)
# print(len(train_data), len(val_data), MF.shape, len(total_negative))

train_set = BPRDataset(train_data, train_count, MF) if method == 'BPR' else BCEDataset(train_data, train_count, MF)
val_set = BPRDataset(val_data, val_count, MF) if method == 'BPR' else BCEDataset(val_data, val_count, MF)

train_set.sampleNegative(total_negative)
val_set.sampleNegative(total_negative)

train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size, shuffle=False, drop_last=False)


""" Test sampling negative """
train_loader.dataset.sampleNegative(total_negative)
val_loader.dataset.sampleNegative(total_negative)
print(len(train_loader.dataset.data), len(train_loader.dataset.positive))
print('Completed processing data.')


""" Prepare the models """
from model import BPRModel, BCEModel
model = BPRModel(MF.shape[0], MF.shape[1]).to(device) if 'BPR' else BCEModel(MF.shape[0], MF.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# -(attr * FT.logsigmoid(prediction) + (1 - attr) * torch.log(1 - torch.sigmoid(prediction))).sum()
# -FT.logsigmoid(pos_prediction - neg_prediction)
""" Start training """
from tqdm import tqdm
min_loss = 1000
if method == 'BPR':
    for epoch in range(n_epoch):
        """ Train one epoch """
        train_loader.dataset.sampleNegative(total_negative)
        val_loader.dataset.sampleNegative(total_negative)
        model.train()
        train_loss = 0
        for user, pos_item, neg_item in tqdm(train_loader):
            optimizer.zero_grad()
            user, pos_item, neg_item = user.to(device), pos_item.to(device), neg_item.to(device)
            pos_prediction, neg_prediction = model(user, pos_item, neg_item)
            loss = -FT.logsigmoid(pos_prediction - neg_prediction).sum()
            train_loss += loss.item() / len(user)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Train Avg Loss = {train_loss / len(train_loader)}')

        """ Validate the MF """
        print('Validating...')
        val_loss = 0
        model.eval()
        for user, pos_item, neg_item in tqdm(val_loader):
            user, pos_item, neg_item = user.to(device), pos_item.to(device), neg_item.to(device)
            pos_prediction, neg_prediction = model(user, pos_item, neg_item)
            loss = -FT.logsigmoid(pos_prediction - neg_prediction).sum()
            val_loss += loss.item() / len(user)
        print(f'Validation Avg Loss = {val_loss / len(val_loader)}')
        if val_loss < min_loss:
            min_loss = val_loss
            print(f'Saving model to {model_path}')
            torch.save(model.state_dict(), model_path)
#os.system(f'python predict.py')