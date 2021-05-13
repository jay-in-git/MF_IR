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

class BPRModel(nn.Module):
    def __init__(self, n_user, n_item, n_latent=512):
        super().__init__()
        self.user_embedding = nn.Embedding(n_user, n_latent)
        self.item_embedding = nn.Embedding(n_item, n_latent)
        nn.init.orthogonal_(self.user_embedding.weight)
        nn.init.orthogonal_(self.item_embedding.weight)
    def forward(self, pos_pair, neg_pair):
        pos_prediction = (self.user_embedding.weight[pos_pair[0]] * self.item_embedding.weight[pos_pair[1]]).sum(dim=1)
        neg_prediction = (self.user_embedding.weight[neg_pair[0]] * self.item_embedding.weight[neg_pair[1]]).sum(dim=1)
        return pos_prediction, neg_prediction
