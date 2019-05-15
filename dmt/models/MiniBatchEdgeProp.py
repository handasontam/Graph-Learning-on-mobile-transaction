import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import EdgeSoftmax
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
                 hidden,
                 out_dim,
                 last=False,
                 **kwargs):
        super(OneLayerNN, self).__init__(**kwargs)
        self.last = last
        self.fc = nn.Linear(in_dim, hidden, bias=True)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=hidden)
        if not self.last:
            self.layer_norm2 = nn.LayerNorm(normalized_shape=out_dim)

    def forward(self, h):
        h = self.fc(h)
        h = self.layer_norm1(h)
        h = F.relu(h)
        # if not self.last:
        #     h = self.layer_norm2(h)
        #     h = F.relu(h)
        return h

class TwoLayerNN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden,
                 out_dim,
                 feat_drop,
                 last=False,
                 **kwargs):
        super(TwoLayerNN, self).__init__(**kwargs)
        self.last = last
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        self.fc = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, out_dim, bias=True)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=hidden)
        if not self.last:
            self.layer_norm2 = nn.LayerNorm(normalized_shape=out_dim)

    def forward(self, h):
        h = self.fc(h)
        h = self.layer_norm1(h)
        h = F.relu(h)
        h = self.feat_drop(h)
        h = self.fc2(h)
        if not self.last:
            h = self.layer_norm2(h)
            h = F.relu(h)
        return h


class NodeUpdate(nn.Module):
    def __init__(self, layer_id, in_dim, out_dim, hidden, feat_drop,
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
                                hidden=hidden, 
                                out_dim=out_dim, 
                                last=last)
        # self.layer = TwoLayerNN(in_dim=in_dim, 
        #                         hidden=hidden, 
        #                         out_dim=out_dim, 
        #                         feat_drop=feat_drop, 
        #                         last=last)
        self.name = name

    def forward(self, node):
        if self.name == 'node':
            h = node.data['h']  # sum of previous layer's delta_h
        elif self.name == 'edge':
            h = node.data['e']
        else:
            print('name must be node/ edge')
            raise ValueError
        norm = node.data['norm']
        # activation from previous layer of myself
        self_h = node.data['self_h']

        if self.test:
            # average
            h = (h - self_h) * norm
            # graphsage
            h = torch.cat((h, self_h), 1)
        else:
            agg_history_str = 'agg_history_{}'.format(self.layer_id)
            agg_history = node.data[agg_history_str]
            # normalization constant
            subg_norm = node.data['subg_norm']
            # delta_h (h - history) from previous layer of myself
            self_delta_h = node.data['self_delta_h']
            # control variate for variance reduction
            # h-self_delta_h because:
            # {1234} -> {34}
            # we only want sum of delta_h for {1,2}
            h = (h - self_delta_h) * subg_norm + agg_history * norm
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
                 residual, 
                 use_batch_norm):
        super(MiniBatchEdgeProp, self).__init__()
        # self.h = create_embeddings(num_nodes, self.in_features)
        self.num_layers = num_layers
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        # self.input_layer = TwoLayerNN(in_dim=in_dim, 
        #                               hidden=num_hidden, 
        #                               out_dim=num_hidden, 
        #                               feat_drop=feat_drop)

        self.input_layer = OneLayerNN(in_dim=in_dim, 
                                      hidden=num_hidden, 
                                      out_dim=num_hidden)



        # edge embeddigns
        # self.input_layer_e = TwoLayerNN(in_dim=edge_in_dim, 
        #                               hidden=num_hidden, 
        #                               out_dim=num_hidden, 
        #                               feat_drop=feat_drop)
        self.input_layer_e = OneLayerNN(in_dim=edge_in_dim, 
                                      hidden=num_hidden, 
                                      out_dim=num_hidden)
        self.node_layers = nn.ModuleList()
        # hidden layers
        for i in range(num_layers):
                self.node_layers.append(NodeUpdate(layer_id=i, 
                                            in_dim=2*num_hidden, 
                                            out_dim=num_hidden, 
                                            hidden=num_hidden, 
                                            feat_drop=feat_drop, 
                                            name='node', 
                                            test=True))

        
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
        h = nf.layers[0].data['features']
        h = self.feat_drop(h)
        h = self.input_layer(h)  # ((#nodes in layer_i) X D)

        for i, node_layer in enumerate(self.node_layers):
            # compute edge embeddings
            e = nf.blocks[i].data['edge_features']
            e = self.feat_drop(e)
            e = self.input_layer_e(e)
            nf.blocks[i].data['e'] = e
            

            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid)
            self_h = h[layer_nid]  # ((#nodes in layer_i+1) X D)
            nf.layers[i+1].data['self_h'] = self_h
            new_history = h.detach()
            history_str = 'history_{}'.format(i)
            history = nf.layers[i].data[history_str]  # ((#nodes in layer_i) X D)

            # delta_h used in control variate
            delta_h = h - history  # ((#nodes in layer_i) X D)
            # delta_h from previous layer of the nodes in (i+1)-th layer, used in control variate
            nf.layers[i+1].data['self_delta_h'] = delta_h[layer_nid]

            nf.layers[i].data['h'] = delta_h

            nf.block_compute(i,
                            fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'), 
                            node_layer)
            h = nf.layers[i+1].data.pop('activation')
            nf.block_compute(i,
                            fn.copy_edge(edge='e', out='m'),
                            fn.sum(msg='m', out='e'), 
                            lambda node: {'e': node.data['e'] * node.data['subg_norm']})
            h = h + nf.layers[i+1].data.pop('e')

            # update history
            if i < nf.num_layers-1:
                nf.layers[i].data[history_str] = new_history
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
                 feat_drop,
                 residual, 
                 use_batch_norm, 
                 ):
        super(MiniBatchEdgePropInfer, self).__init__()
        # self.h = create_embeddings(num_nodes, self.in_features)
        self.num_layers = num_layers
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        # self.input_layer = TwoLayerNN(in_dim=in_dim, 
        #                               hidden=num_hidden, 
        #                               out_dim=num_hidden, 
        #                               feat_drop=feat_drop)

        self.input_layer = OneLayerNN(in_dim=in_dim, 
                                      hidden=num_hidden, 
                                      out_dim=num_hidden)



        # edge embeddigns
        # self.input_layer_e = TwoLayerNN(in_dim=edge_in_dim, 
        #                               hidden=num_hidden, 
        #                               out_dim=num_hidden, 
        #                               feat_drop=feat_drop)
        self.input_layer_e = OneLayerNN(in_dim=edge_in_dim, 
                                      hidden=num_hidden, 
                                      out_dim=num_hidden)
        self.node_layers = nn.ModuleList()
        # hidden layers
        for i in range(num_layers):
                self.node_layers.append(NodeUpdate(layer_id=i, 
                                            in_dim=2*num_hidden, 
                                            out_dim=num_hidden, 
                                            hidden=num_hidden, 
                                            feat_drop=feat_drop, 
                                            name='node', 
                                            test=True))
        
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
        h = nf.layers[0].data['features']
        h = self.feat_drop(h)
        h = self.input_layer(h)  # ((#nodes in layer_i) X D)

        for i, node_layer in enumerate(self.node_layers):
            # compute edge embeddings
            e = nf.blocks[i].data['edge_features']
            e = self.feat_drop(e)
            e = self.input_layer_e(e)
            nf.blocks[i].data['e'] = e
            
            nf.layers[i].data['h'] = h
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid)
            self_h = h[layer_nid]  # ((#nodes in layer_i+1) X D)
            nf.layers[i+1].data['self_h'] = self_h
            nf.block_compute(i,
                            fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'), 
                            node_layer)
            h = nf.layers[i+1].data.pop('activation')
            nf.block_compute(i,
                            fn.copy_edge(edge='e', out='m'),
                            fn.sum(msg='m', out='e'), 
                            lambda node: {'e': node.data['e'] * node.data['norm']}
                            )
            embeddings = h + nf.layers[i+1].data.pop('e')

        h = self.fc(embeddings)
        return h, embeddings

    def edge_nonlinearity(self, edges):
        eft = self.activation(edges.data['eft'])
        return {'eft': eft}
