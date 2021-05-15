import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import BPRModel, BCEModel
from scipy.sparse import dok_matrix, save_npz, load_npz
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='MFModel')
parser.add_argument('-o', type=str, dest="outPath")
parser.add_argument('-m', type=str, dest="method")
parser.add_argument('-MT', type=str, dest="MFPath")
parser.add_argument('-s1', type=str, dest="modelPath1")
parser.add_argument('-s2', type=str, dest="modelPath2")
argvs = parser.parse_args()

MF = load_npz(argvs.MFPath).todok()

model1 = BPRModel(MF.shape[0], MF.shape[1]) if argvs.method == 'BPR' else BCEModel(MF.shape[0], MF.shape[1])
ckpt1 = torch.load(argvs.modelPath1, map_location=torch.device('cpu'))
model1.load_state_dict(ckpt1)

model2 = BPRModel(MF.shape[0], MF.shape[1]) if argvs.method == 'BPR' else BCEModel(MF.shape[0], MF.shape[1])
ckpt2 = torch.load(argvs.modelPath2, map_location=torch.device('cpu'))
model2.load_state_dict(ckpt2)

model1.eval()
model2.eval()
with open(argvs.outPath, 'w') as out_file:
    print('UserId,ItemId', file=out_file)
    prediction = ((model1.getPrediction() + model2.getPrediction()) / 2).sort(descending=True)[1]
    print(prediction.size())
    keys = MF.keys()
    for user in tqdm(range(MF.shape[0])):
        print(f'{user},', file=out_file, end='')
        cnt = 0
        for item in prediction[user]:
            if (user, item.item()) not in keys:
                print(item.item(), end=' ', file=out_file)
                cnt += 1
            if cnt == 50:
                print('', file=out_file)
                break
