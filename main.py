import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as FT

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


""" Set hyper-parameters """
file_path = os.sys.argv[1]
method = os.sys.argv[2]
model_path = os.sys.argv[3]
batch_size = 1024
n_epoch = 30
lr = 0.001
weight_decay = 0.01 if method == 'BPR' else 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


""" Load the dataset """
print(f'loading data from {file_path}...')
from loadData import loadData, BPRDataset, BCEDataset
train_data, train_count, val_data, val_count, MF, total_negative = loadData(file_path, method=method)
# print(len(train_data), len(val_data), MF.shape, len(total_negative))

train_set = BPRDataset(train_data, train_count, MF) if method == 'BPR' else BCEDataset(train_data, train_count, MF)
val_set = BPRDataset(val_data, val_count, MF) if method == 'BPR' else BCEDataset(val_data, val_count, MF)

train_set.sampleNegative(total_negative)
val_set.sampleNegative(total_negative)

train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size, shuffle=False, drop_last=False)


""" Test sampling negative """
train_loader.dataset.sampleNegative(total_negative)
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
if method == 'BPR':
    min_loss = 1000
    for epoch in range(n_epoch):
        """ Train one epoch """
        model.train()
        train_loss = 0
        for pos_pairs, neg_pairs in tqdm(train_loader):
            optimizer.zero_grad()
            pos_prediction, neg_prediction = model(pos_pairs, neg_pairs)
            loss = -FT.logsigmoid(pos_prediction - neg_prediction).sum()
            train_loss += loss.item() / len(pos_pairs)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Train Avg Loss = {train_loss}')

        """ Validate the MF """
        print('Validating...')
        val_loss = 0
        model.eval()
        for pos_pairs, neg_pairs in tqdm(val_loader):
            pos_prediction, neg_prediction = model(pos_pairs, neg_pairs)
            loss = -FT.logsigmoid(pos_prediction - neg_prediction).sum()
            val_loss += loss.item() / len(pos_pairs)
        print(f'Validation Avg Loss = {val_loss}')
        if val_loss < min_loss:
            min_loss = val_loss
            print(f'Saving model to {model_path}')
            torch.save(model.state_dict(), model_path)

os.system(f'python predict.py')