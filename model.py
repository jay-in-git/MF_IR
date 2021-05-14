import torch
import torch.nn as nn

class BCEModel(nn.Module):
    def __init__(self, n_user, n_item, n_latent=512):
        super().__init__()
        self.user_embedding = nn.Embedding(n_user, n_latent)
        self.item_embedding = nn.Embedding(n_item, n_latent)
        nn.init.orthogonal_(self.user_embedding.weight)
        nn.init.orthogonal_(self.item_embedding.weight)
    def forward(self, user, item, attr):
        return (self.user_embedding.weight[user] * self.item_embedding.weight[item]).sum(dim=1)
    def getPrediction(self):
        return torch.matmul(self.user_embedding.weight, self.item_embedding.weight.T)

class BPRModel(nn.Module):
    def __init__(self, n_user, n_item, n_latent=512):
        super().__init__()
        self.user_embedding = nn.Embedding(n_user, n_latent)
        self.item_embedding = nn.Embedding(n_item, n_latent)
        nn.init.orthogonal_(self.user_embedding.weight)
        nn.init.orthogonal_(self.item_embedding.weight)
    def forward(self, user, pos_item, neg_item ):
        pos_prediction = (self.user_embedding.weight[user] * self.item_embedding.weight[pos_item]).sum(dim=1)
        neg_prediction = (self.user_embedding.weight[user] * self.item_embedding.weight[neg_item]).sum(dim=1)
        return pos_prediction, neg_prediction
    def getPrediction(self):
        return torch.matmul(self.user_embedding.weight, self.item_embedding.weight.T)
