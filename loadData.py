from torch.utils.data import Dataset, DataLoader
from scipy.sparse import dok_matrix, save_npz
from random import sample
import numpy as np
import os

class BCEDataset(Dataset):
    def __init__(self, data, item_count, MF, neg=5):
        self.positive = data
        self.n_positive = len(data)
        self.item_count = item_count
        self.MF = MF
        self.neg = neg
        self.data = list()
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    def sampleNegative(self, total_negative):
        self.data = self.positive.copy()
        for i in range(self.MF.shape[0]):
            neg_samples = sample(total_negative[i], min(int(self.neg * self.item_count[i]), self.MF.shape[1]))
            self.data.extend(neg_samples)

class BPRDataset(Dataset):
    def __init__(self, data, item_count, MF, neg=5):
        self.positive = data
        self.n_positive = len(data)
        self.item_count = item_count
        self.MF = MF
        self.neg = neg
        self.data = list()
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    def sampleNegative(self, total_negative):
        self.data = list()
        for pair in self.positive:
            neg_samples = sample(total_negative[pair[0]], self.neg)
            for i in range(self.neg):
                self.data.append((pair[0], pair[1], neg_samples[i][1]))

def loadData(file_path, MFPath, method='BPR', cut=10, nug_num=5):
    raw_data = list()
    user_num = 0
    item_num = 0
    with open(file_path) as f:
        f.readline()
        for line in f:
            user_idx, item_idxs = line.strip().split(',')
            user_idx = int(user_idx)
            user_num = max(user_num, user_idx)

            item_idxs = [int(item_idx.strip()) for item_idx in item_idxs.split()]
            item_num = max(item_num, *tuple(item_idxs)) 

            raw_data.append((user_idx, item_idxs))
            
    MF = dok_matrix((user_num + 1, item_num + 1))
    total_negative = list()

    train_data = list()
    train_count = np.zeros(user_num + 1)

    val_data = list()
    val_count = np.zeros(user_num + 1)

    print('Computing MF...')
    for user, items in raw_data:
        for idx, item in enumerate(items):
            if idx % cut != 0:
                train_count[user] += 1
                train_data.append((user, item, 1))
            else:
                val_count[user] += 1
                val_data.append((user, item, 1))
        neg_list = list(set(range(item_num + 1)) - set(items))
        total_negative.append(list())
        for neg_index in neg_list:
            total_negative[user].append((user, neg_index, 0))
        MF[user, items] = 1
    if not os.path.exists(f'{MFPath}.npz'):
        print(f'Saving matrix to {MFPath}.npz')
        save_npz(MFPath, MF.tocoo())
    return train_data, train_count, val_data, val_count, MF, total_negative