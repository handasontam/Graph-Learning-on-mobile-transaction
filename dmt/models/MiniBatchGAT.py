import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            h = h * node.data['norm']
        h = self.linear(h)
        # skip connection
        if self.concat:
            h = torch.cat((h, self.activation(h)), dim=1)
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}


class MiniBatchGATSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(MiniBatchGATSampling, self).__init__()
        self.n_layers = n_layers
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        # input layer
        skip_start = (0 == n_layers-1)
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
        # output layer
        # self.output_layers = NodeUpdate(2*n_hidden, n_classes)
        self.fc = nn.Linear(2*n_hidden, n_classes, bias=True)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             lambda node : {'h': node.mailbox['m'].mean(dim=1)},
                             layer)
        embedding = nf.layers[-1].data.pop('activation')
        h = self.fc(embedding)
        return h


class MiniBatchGATInfer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(MiniBatchGATInfer, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        skip_start = (0 == n_layers-1)
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, test=True, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, test=True, concat=skip_start))
        # output layer
        # self.layers.append(NodeUpdate(2*n_hidden, n_classes, test=True))
        # self.output_layers = NodeUpdate(2*n_hidden, n_classes)
        self.fc = nn.Linear(2*n_hidden, n_classes, bias=True)

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)

        embedding = nf.layers[-1].data.pop('activation')
        h = self.fc(embedding)
        return h, embedding


# def main(args):
#     # load and preprocess dataset
#     data = load_data(args)

#     if args.self_loop and not args.dataset.startswith('reddit'):
#         data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

#     train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
#     test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)

#     features = torch.FloatTensor(data.features)
#     labels = torch.LongTensor(data.labels)
#     train_mask = torch.ByteTensor(data.train_mask)
#     val_mask = torch.ByteTensor(data.val_mask)
#     test_mask = torch.ByteTensor(data.test_mask)
#     in_feats = features.shape[1]
#     n_classes = data.num_labels
#     n_edges = data.graph.number_of_edges()

#     n_train_samples = train_mask.sum().item()
#     n_val_samples = val_mask.sum().item()
#     n_test_samples = test_mask.sum().item()

#     print("""----Data statistics------'
#       #Edges %d
#       #Classes %d
#       #Train samples %d
#       #Val samples %d
#       #Test samples %d""" %
#           (n_edges, n_classes,
#               n_train_samples,
#               n_val_samples,
#               n_test_samples))

#     # create GCN model
#     g = DGLGraph(data.graph, readonly=True)
#     norm = 1. / g.in_degrees().float().unsqueeze(1)

#     if args.gpu < 0:
#         cuda = False
#     else:
#         cuda = True
#         torch.cuda.set_device(args.gpu)
#         features = features.cuda()
#         labels = labels.cuda()
#         train_mask = train_mask.cuda()
#         val_mask = val_mask.cuda()
#         test_mask = test_mask.cuda()
#         norm = norm.cuda()

#     g.ndata['features'] = features

#     num_neighbors = args.num_neighbors

#     g.ndata['norm'] = norm

#     model = GCNSampling(in_feats,
#                         args.n_hidden,
#                         n_classes,
#                         args.n_layers,
#                         F.relu,
#                         args.dropout)

#     if cuda:
#         model.cuda()

#     loss_fcn = nn.CrossEntropyLoss()

#     infer_model = GCNInfer(in_feats,
#                            args.n_hidden,
#                            n_classes,
#                            args.n_layers,
#                            F.relu)

    # if cuda:
    #     infer_model.cuda()

    # # use optimizer
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.lr,
    #                              weight_decay=args.weight_decay)

    # # initialize graph
    # dur = []
    # for epoch in range(args.n_epochs):
    #     for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
    #                                                    args.num_neighbors,
    #                                                    neighbor_type='in',
    #                                                    shuffle=True,
    #                                                    num_hops=args.n_layers+1,
    #                                                    seed_nodes=train_nid):
    #         nf.copy_from_parent()
    #         model.train()
    #         # forward
    #         pred = model(nf)
    #         batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
    #         batch_labels = labels[batch_nids]
    #         loss = loss_fcn(pred, batch_labels)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     for infer_param, param in zip(infer_model.parameters(), model.parameters()):
    #         infer_param.data.copy_(param.data)

    #     num_acc = 0.

    #     for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
    #                                                    g.number_of_nodes(),
    #                                                    neighbor_type='in',
    #                                                    num_hops=args.n_layers+1,
    #                                                    seed_nodes=test_nid):
    #         nf.copy_from_parent()
    #         infer_model.eval()
    #         with torch.no_grad():
    #             pred = infer_model(nf)
    #             batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
    #             batch_labels = labels[batch_nids]
    #             num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

    #     print("Test Accuracy {:.4f}". format(num_acc/n_test_samples))
