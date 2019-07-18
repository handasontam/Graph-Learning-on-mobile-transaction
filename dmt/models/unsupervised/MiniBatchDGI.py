"""
Deep Graph Infomax in DGL
References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..MiniBatchEdgePropNS import MiniBatchEdgeProp, MiniBatchEdgePropInfer
from ..MiniBatchGCN import MiniBatchGCNInfer, MiniBatchGCNSampling
import logging
from dgl.contrib.sampling import NeighborSampler


class Encoder(nn.Module):
    def __init__(self, g, conv_model, in_feats, edge_in_feats, n_hidden, n_layers, activation, dropout, infer, cuda):
        super(Encoder, self).__init__()
        self.g = g
        self.infer = infer
        if conv_model.upper() == 'EDGEPROP':
            if not infer:
                self.conv = MiniBatchEdgeProp(
                    g,
                    n_layers,
                    in_feats,
                    edge_in_feats,
                    n_hidden,
                    n_hidden,
                    F.elu,
                    dropout,
                    cuda)
            else:
                self.conv = MiniBatchEdgePropInfer(
                    g,
                    n_layers,
                    in_feats,
                    edge_in_feats,
                    n_hidden,
                    n_hidden,
                    F.elu,
                    cuda)
        elif conv_model.upper() == 'GCN':
            if not infer:
                self.conv = MiniBatchGCNSampling(
                    in_feats=in_feats,
                    n_hidden=n_hidden,
                    n_classes=n_hidden,
                    n_layers=n_layers,
                    activation=F.relu, 
                    dropout=dropout
                )
            else:
                self.conv = MiniBatchGCNInfer(
                    in_feats=in_feats, 
                    n_hidden=n_hidden,
                    n_classes=n_hidden,
                    n_layers=n_layers,
                    activation=F.relu
                )
        else:
            logging.info(
                'The encoder model - {} is not implemented in DGI'.format(conv_model))

    def forward(self, nf, corrupt=False):
        if corrupt:
            # shuffle node features
            perm = torch.randperm(nf.layer_size(0))
            nf.layers[0].data['node_features'] = nf.layers[0].data['node_features'][perm]
            for i in range(nf.num_layers-1):
                # shuffle edge features
                perm = torch.randperm(nf.block_size(i))
                nf.blocks[i].data['edge_features'] = nf.blocks[i].data['edge_features'][perm]
        if self.infer:
            features, _ = self.conv(nf)
        else:
            features = self.conv(nf)
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
    def __init__(self, g, conv_model, in_feats, edge_in_feats, n_hidden, n_layers, activation, dropout, cuda):
        super(DGI, self).__init__()
        self.encoder = Encoder(g, conv_model, in_feats, edge_in_feats,
                               n_hidden, n_layers, activation, dropout, False, cuda)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, nf):
        positive = self.encoder(nf, corrupt=False)
        negative = self.encoder(nf, corrupt=True)
        summary = positive.mean(dim=0)
        # summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


class DGIInfer(nn.Module):
    def __init__(self, g, conv_model, in_feats, edge_in_feats, n_hidden, n_layers, activation, dropout, cuda):
        super(DGIInfer, self).__init__()
        self.encoder = Encoder(g, conv_model, in_feats, edge_in_feats,
                               n_hidden, n_layers, activation, dropout, True, cuda)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, nf):
        positive = self.encoder(nf, corrupt=False)
        negative = self.encoder(nf, corrupt=True)
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
