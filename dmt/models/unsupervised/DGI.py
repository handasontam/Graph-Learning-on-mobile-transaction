"""
Deep Graph Infomax in DGL
References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import torch
import torch.nn as nn
import math
from .gcn import GCN
from ..GAT import GAT
import logging

class Encoder(nn.Module):
    def __init__(self, g, conv_model, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        if conv_model.upper() == 'GCN':
            self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        elif conv_model.upper() == 'GAT':
            num_heads = 4
            heads=([num_heads] * n_layers) + [num_heads]
            self.conv = GAT(g=g, 
                            num_layers=n_layers, 
                            in_dim=in_feats, 
                            num_hidden=n_hidden, 
                            num_classes=n_hidden, 
                            heads=heads,
                            activation=activation,
                            feat_drop=dropout,
                            attn_drop=dropout, 
                            alpha=0.2,
                            residual=True)
        else:
            logging.info('The encoder model - {} is not implemented in DGI'.format(conv_model))

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, g, conv_model, in_feats, n_hidden, n_layers, activation, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder(g, conv_model, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        summary = positive.mean(dim=0)

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)