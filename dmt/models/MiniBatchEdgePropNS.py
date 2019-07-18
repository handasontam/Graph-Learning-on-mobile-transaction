import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import EdgeSoftmax
# from ..utils.randomwalk import random_walk_nodeflow
from dgl.contrib.sampling.sampler import NeighborSampler
import dgl

def div(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b) # prevent division by zero
    return a / b

def get_embeddings(h, nodeset):
    return h[nodeset]

def put_embeddings(h, nodeset, new_embeddings):
    n_nodes = nodeset.shape[0]
    n_features = h.shape[1]
    return h.scatter(0, nodeset[:, None].expand(n_nodes, n_features), new_embeddings)

class OneLayerNN(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 last=False,
                 **kwargs):
        super(OneLayerNN, self).__init__(**kwargs)
        self.last = last
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, h):
        h = self.fc(h)
        h = F.relu(h)
        h = self.fc2(h)
        return h

class NodeUpdate(nn.Module):
    def __init__(self, layer_id, in_dim, out_dim, feat_drop,
                 test=False, last=False, name=''):
        super(NodeUpdate, self).__init__()
        self.layer_id = layer_id
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        self.test = test
        self.last = last
        self.layer = OneLayerNN(in_dim=in_dim, 
                                out_dim=out_dim, 
                                last=last)
        self.name = name

    def forward(self, node):
        h = node.data['h']  # sum of previous layer's delta_h
        # activation from previous layer of myself
        self_h = node.data['self_h']

        if self.test:
            # average
            norm = node.data['norm']
            h = (h - self_h) * norm
        else:
            # normalization constant
            subg_norm = node.data['subg_norm']
            h = (h - self_h) * subg_norm
        # graphsage
        h = torch.cat((h, self_h), 1)
        h = self.feat_drop(h)

        h = self.layer(h)

        # return {'activation_{}'.format(self.name): h}
        return {'activation': h}

class MiniBatchEdgeProp(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 edge_in_dim, 
                 num_hidden,
                 num_classes,
                 activation,
                 feat_drop,
                 cuda):
        super(MiniBatchEdgeProp, self).__init__()
        self.cuda = cuda
        self.num_layers = num_layers
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        self.in_dim = in_dim
        self.edge_in_dim = edge_in_dim
        self.num_hidden = num_hidden


        self.input_layer_e = OneLayerNN(in_dim=edge_in_dim, 
                                      out_dim=num_hidden)
        self.node_layers = nn.ModuleList()
        self.node_layers.append(NodeUpdate(layer_id=0, 
                                        in_dim=2*num_hidden, 
                                        out_dim=num_hidden, 
                                        feat_drop=feat_drop, 
                                        name='node', 
                                        test=False))

        self.phi = nn.ModuleList()
        self.phi.append(OneLayerNN(in_dim=in_dim+num_hidden,
                                   out_dim=num_hidden))
        for i in range(1, num_layers):
            self.node_layers.append(NodeUpdate(layer_id=i, 
                                        in_dim=2*num_hidden, 
                                        out_dim=num_hidden, 
                                        feat_drop=feat_drop, 
                                        name='node', 
                                        test=False))
            self.phi.append(OneLayerNN(in_dim=2*num_hidden,
                                       out_dim=num_hidden))

        
        # output projection
        self.fc = nn.Linear(num_hidden, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc.weight.data, gain=0.1)


    def forward(self, nodeflow):
        '''
        Given a complete embedding matrix h and a list of node IDs, return
        the output embeddings of these node IDs.

        nodeflow: NodeFlow Object
        return: new node embeddings (num_nodes, out_features)
        '''
        nf = nodeflow
        h = nf.layers[0].data['node_features']
        # h = self.feat_drop(h)
        # h = self.input_layer(h)  # ((#nodes in layer_i) X D)

        for i, (node_layer, phi_layer) in enumerate(zip(self.node_layers, self.phi)):
            # compute edge embeddings
            e = nf.blocks[i].data['edge_features']
            e = self.feat_drop(e)
            e = self.input_layer_e(e)
            nf.blocks[i].data['e'] = e

            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            if self.cuda:
                self_h = torch.cat((torch.zeros(len(layer_nid), self.num_hidden).cuda(), h[layer_nid]), 1)
            else:
                self_h = torch.cat((torch.zeros(len(layer_nid), self.num_hidden), h[layer_nid]), 1)
            self_h = phi_layer(self_h)
            nf.layers[i+1].data['self_h'] = self_h # ((#nodes in layer_i+1) X D)
            nf.layers[i].data['h'] = h

            def message_func(edges):
                temp = torch.cat((edges.data['e'],edges.src['h']), 1)
                temp = phi_layer(temp)
                return {'m': temp}
    
            nf.block_compute(i,
                            message_func,
                            fn.sum(msg='m', out='h'), 
                            node_layer)
            h = nf.layers[i+1].data.pop('activation')

        h = self.fc(h)
        return h

    def edge_nonlinearity(self, edges):
        eft = self.activation(edges.data['eft'])
        return {'eft': eft}


class MiniBatchEdgePropInfer(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 edge_in_dim, 
                 num_hidden,
                 num_classes,
                 activation,
                 cuda):
        super(MiniBatchEdgePropInfer, self).__init__()
        self.cuda = cuda
        self.num_layers = num_layers
        self.activation = activation
        self.in_dim = in_dim
        self.edge_in_dim = edge_in_dim
        self.num_hidden = num_hidden
        # self.input_layer = OneLayerNN(in_dim=in_dim, 
        #                               out_dim=num_hidden)


        # self.phi = OneLayerNN(in_dim=2*num_hidden,
        #                       out_dim=num_hidden)

        self.input_layer_e = OneLayerNN(in_dim=edge_in_dim, 
                                      out_dim=num_hidden)
        self.node_layers = nn.ModuleList()
        self.node_layers.append(NodeUpdate(layer_id=0, 
                                        in_dim=2*num_hidden, 
                                        out_dim=num_hidden, 
                                        feat_drop=0, 
                                        name='node', 
                                        test=True))

        self.phi = nn.ModuleList()
        self.phi.append(OneLayerNN(in_dim=in_dim+num_hidden,
                                   out_dim=num_hidden))
        for i in range(1, num_layers):
            self.node_layers.append(NodeUpdate(layer_id=i, 
                                        in_dim=2*num_hidden, 
                                        out_dim=num_hidden, 
                                        feat_drop=0, 
                                        name='node', 
                                        test=True))
            self.phi.append(OneLayerNN(in_dim=2*num_hidden,
                                       out_dim=num_hidden))
        
        # output projection
        self.fc = nn.Linear(num_hidden, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc.weight.data, gain=0.1)


    def forward(self, nodeflow):
        '''
        Given a complete embedding matrix h and a list of node IDs, return
        the output embeddings of these node IDs.

        nodeflow: NodeFlow Object
        return: new node embeddings (num_nodes, out_features)
        '''
        nf = nodeflow
        h = nf.layers[0].data['node_features']

        for i, (node_layer, phi_layer) in enumerate(zip(self.node_layers, self.phi)):
            # compute edge embeddings
            e = nf.blocks[i].data['edge_features']
            e = self.input_layer_e(e)
            nf.blocks[i].data['e'] = e
            
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid, remap_local=True)
            if self.cuda:
                self_h = torch.cat((torch.zeros(len(layer_nid), self.num_hidden).cuda(), h[layer_nid]), 1)
            else:
                self_h = torch.cat((torch.zeros(len(layer_nid), self.num_hidden), h[layer_nid]), 1)
            self_h = phi_layer(self_h)
            nf.layers[i+1].data['self_h'] = self_h # ((#nodes in layer_i+1) X D)

            nf.layers[i].data['h'] = h
            def message_func(edges):
                temp = torch.cat((edges.data['e'],edges.src['h']), 1)
                temp = phi_layer(temp)
                # temp = self.activation(temp)
                return {'m': temp}

            nf.block_compute(i,
                            message_func,
                            fn.sum(msg='m', out='h'), 
                            node_layer)
            h = nf.layers[i+1].data.pop('activation')
        embedding = h
        h = self.fc(embedding)
        return h, embedding

    def edge_nonlinearity(self, edges):
        eft = self.activation(edges.data['eft'])
        return {'eft': eft}
