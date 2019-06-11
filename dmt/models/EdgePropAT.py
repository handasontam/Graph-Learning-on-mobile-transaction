"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch.softmax import EdgeSoftmax

def div(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b) # prevent division by zero
    return a / b
class GraphAttention(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 edge_in_dim, 
                 out_dim, 
                 num_heads,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual=False,
                 use_batch_norm=False):
        super(GraphAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=True)
        self.fc_e = nn.Linear(edge_in_dim, num_heads * out_dim, bias=True)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_e = nn.Parameter(torch.Tensor(size=(edge_in_dim, num_heads))) # (K X H)
        nn.init.xavier_normal_(self.fc.weight.data, gain=0.1)
        nn.init.xavier_normal_(self.attn_l.data, gain=0.1)
        nn.init.xavier_normal_(self.attn_r.data, gain=0.1)
        nn.init.xavier_normal_(self.attn_e.data ,gain=0.1)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = EdgeSoftmax()
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=0.1)
            else:
                self.res_fc = None
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(self.out_dim * self.num_heads)

    def forward(self, inputs, edge_inputs):
        # prepare
        h = self.feat_drop(inputs)  # NxD
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
        self.g.edata['eft'] = edge_inputs
        # e = self.feat_drop(edge_inputs)  # EXK (K=num edge features)
        head_ft = ft.transpose(0, 1)  # HxNxD'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)  # NxHx1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)  # NxHx1
        a3 = torch.mm(edge_inputs, self.attn_e)  # EXH
        a3 = torch.unsqueeze(a3, -1)  # EXHX1
        # edge attention
        self.g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        self.g.edata['a3'] = a3
        # 1. compute edge attention
        self.g.apply_edges(self.edge_attention)
        self.g.apply_edges(self.edge_transform)
        # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
        self.edge_softmax()
        # 2. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        self.g.update_all(fn.copy_edge(edge='eft', out='sum_eft'), fn.sum('sum_eft', 'sum_eft')) # aggregate edge features
        # 3. apply normalizer
        ret = self.g.ndata['ft'] / self.g.ndata['z']  # NxHxD'
        ret = ret + self.g.ndata['sum_eft'].reshape((self.g.ndata['sum_eft'].shape[0], self.num_heads, -1))  # EXHXK'  # merge edge features with aggregate node features
        # 4. residual
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                resval = torch.unsqueeze(h, 1)  # Nx1xD'
            ret = resval + ret
        # 5. flatten
        ret = ret.flatten(1) # NXD'
        # 6. Batch normalization
        if self.use_batch_norm:
            ret = self.bn(ret)
        return ret

    def edge_transform(self, edges):
        eft = self.fc_e(edges.data['eft'])
        # head_eft = e.transpose(0, 1)  # HXEXK'
        # a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)  # NxHx1
        return {'eft': eft}

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'] + edges.data['a3'])
        return {'a' : a}

    def edge_softmax(self):
        scores, normalizer = self.softmax(self.g.edata['a'], self.g)
        # Save normalizer
        self.g.ndata['z'] = normalizer
        # Dropout attention scores and save them
        self.g.edata['a_drop'] = self.attn_drop(scores)

class EdgePropGAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 edge_in_dim, 
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual, 
                 use_batch_norm):
        super(EdgePropGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GraphAttention(
            g, in_dim, edge_in_dim, num_hidden, heads[0], feat_drop, attn_drop, alpha, 
            False, use_batch_norm))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttention(
                g, num_hidden * heads[l-1], num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, alpha, residual, use_batch_norm))
        # output projection
        self.fc = nn.Linear(num_hidden * heads[-1], num_classes, bias=True)
        nn.init.xavier_normal_(self.fc.weight.data, gain=0.1)
        # self.gat_layers.append(GraphAttention(
        #     g, num_hidden * heads[-2], num_hidden * heads[-2], num_classes, heads[-1],  # only nodes features have multi-head
        #     feat_drop, attn_drop, alpha, residual, False))


    def forward(self, inputs):
        h = inputs
        # e = edge_inputs
        for l in range(self.num_layers):
            if l == 0:
                e = self.g.edata['edge_features']
            else:
                e = self.g.edata['eft']
            h = self.gat_layers[l](h, e)
            h = self.activation(h)
            self.g.apply_edges(self.edge_nonlinearity)
        # output projection
        # logits = self.gat_layers[-1](h, self.g.edata['eft'])
        logits = self.fc(h)
        return logits

    def edge_nonlinearity(self, edges):
        eft = self.activation(edges.data['eft'])
        return {'eft': eft}
