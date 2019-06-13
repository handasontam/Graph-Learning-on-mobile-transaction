import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import EdgeSoftmax
from dgl.contrib.sampling.sampler import NeighborSampler
import dgl

class SingleLayerNN(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 **kwargs):
        super(SingleLayerNN, self).__init__(**kwargs)
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        return x

class NodeUpdate(nn.Module):
    def __init__(self, layer_id, in_dim, out_dim, feat_drop,
                 test=False, name=''):
        super(NodeUpdate, self).__init__()
        self.layer_id = layer_id
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        self.test = test
        self.output_layer = SingleLayerNN(in_dim=in_dim, 
                                out_dim=out_dim)
        self.name = name

    def forward(self, node):
        
        delta_nb = node.data['delta_nb']  # sum of previous layer's delta_h        
    
        norm = node.data['norm']
        # activation from previous layer of myself
        self_h = node.data['self_h']
        

        if self.test:
            # average
            new_h = (delta_nb - self_h) * norm
        else:
            # normalization constant
            subg_norm = node.data['subg_norm']
            if self.layer_id == 0:
                new_h = (delta_nb - self_h) * subg_norm
            else:
                self_delta_h = node.data['self_delta_h']
                agg_history_str = 'agg_history_{}'.format(self.layer_id)
                agg_history = node.data[agg_history_str]
                # delta_h (h - history) from previous layer of myself
                
                # control variate for variance reduction
                # h-self_delta_h because:
                # {1234} -> {34}
                # we only want sum of delta_h for {1,2}
                delta_nb = (delta_nb - self_delta_h) * subg_norm
                new_h = delta_nb + agg_history * norm
        # graphsage
        new_h = torch.cat((new_h, self_h), 1)
        new_h = self.feat_drop(new_h)

        new_h = self.output_layer(new_h)

        return {'activation': new_h}


class EdgeSeq(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(EdgeSeq, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)

    def forward(self, input, hidden):
        #print('in edge seq')
        #print(input.size())
        #print(hidden.size())
        #input.view()
        #hidden = torch.zeros(1, input.size()[0], self.edge_hidden_dim)
        #hidden = hidden.reshape(-1, 1, hidden.size()[1])
        output, hidden = self.gru(input, hidden)
        return output, hidden


class MiniBatchEdgePropPlus(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 node_in_dim,
                 edge_in_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 fc_hidden_dim,
                 num_classes,
                 activation,
                 feat_drop,
                 residual, 
                 use_batch_norm):
        super(MiniBatchEdgePropPlus, self).__init__()
        
        self.num_layers = num_layers
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.node_hidden_dim = node_hidden_dim

        self.edge_hidden_dim = edge_hidden_dim  

        # self.input_layer_e = OneLayerNN(node_in_dim=edge_in_dim, 
        #                               out_dim=num_hidden)
        self.edgeseq_layers = nn.ModuleList()
        self.edgeseq_layers.append(EdgeSeq(in_dim=edge_in_dim, 
                                            hidden_dim=self.edge_hidden_dim))


        self.node_layers = nn.ModuleList()
        self.node_layers.append(NodeUpdate(layer_id=0, 
                                        in_dim=2*node_hidden_dim, 
                                        out_dim=node_hidden_dim, 
                                        feat_drop=feat_drop, 
                                        name='node', 
                                        test=False))

        self.phi = nn.ModuleList()
        self.phi.append(SingleLayerNN(in_dim=node_in_dim + edge_hidden_dim,
                                   out_dim=node_hidden_dim))
        # self.phi.append(OneLayerNN(node_in_dim=2 * num_hidden,
        #                            out_dim=num_hidden))
        for i in range(1, num_layers):
            self.edgeseq_layers.append(EdgeSeq(in_dim=self.edge_hidden_dim, 
                                            hidden_dim=self.edge_hidden_dim))

            self.node_layers.append(NodeUpdate(layer_id=i, 
                                        in_dim=2*node_hidden_dim, 
                                        out_dim=node_hidden_dim, 
                                        feat_drop=feat_drop, 
                                        name='node', 
                                        test=False))
            self.phi.append(SingleLayerNN(in_dim=node_hidden_dim + edge_hidden_dim,
                                       out_dim=node_hidden_dim))

        
        # output projection
        self.fc1 = nn.Linear(node_hidden_dim, fc_hidden_dim, bias=True)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc1.weight.data, gain=0.1)
        nn.init.xavier_normal_(self.fc2.weight.data, gain=0.1)

    def forward(self, nodeflow):
        '''
        Given a complete embedding matrix h and a list of node IDs, return
        the output embeddings of these node IDs.

        nodeflow: NodeFlow Object
        return: new node embeddings (num_nodes, out_features)
        '''
        nf = nodeflow
        h = nf.layers[0].data['node_features'] 
        # h = self.input_layer(h)  # ((#nodes in layer_i) X D)

        for i, (edgeseq_layer, node_layer, phi_layer) in enumerate(zip(self.edgeseq_layers, self.node_layers, self.phi)):
            # compute node embedding for i+1 layer

            # compute edge embeddings
            e = nf.blocks[i].data['edge_features']
            
            e = self.feat_drop(e)
            #e = self.input_layer_e(e)
            nf.blocks[i].data['e'] = e
            #print('assign edges!!!!!!!!!!!!!!!!!!!', e.size())

            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))   # get i+1 layer nodes from parent graph
            self_layer_nid = nf.map_from_parent_nid(i, parent_nid)          # get the nodeflow id for the self edge node
            self_h = torch.cat((torch.zeros(len(self_layer_nid), self.node_hidden_dim), h[self_layer_nid]), 1)
            self_h = phi_layer(self_h)
            nf.layers[i+1].data['self_h'] = self_h # ((#nodes in layer_i+1) X D)
            if i == 0:
                nf.layers[i].data['h'] = h              
                #print('assign h', h.size())
                nf.layers[i+1].data['self_delta_h'] = self_h
            else:
                new_history = h.detach()
                history_str = 'history_{}'.format(i)
                history = nf.layers[i].data[history_str]  # ((#nodes in layer_i) X D)

                # delta_h used in control variate
                #delta_h = h - history  # ((#nodes in layer_i) X D)
                # delta_h from previous layer of the nodes in (i+1)-th layer, used in control variate
                nf.layers[i+1].data['self_delta_h'] = self_h - history[self_layer_nid]

                #nf.layers[i].data['h'] = delta_h
                

            def message_func(edges):
                #print(edges.dst.keys())
                #print('hidden!!!!!!!!!!', edges.src['h'].size())
                #hidden = torch.cat((edges.src['h'], edges.src.['h']), 1)
                edges_seq = edges.data['e']

                hidden = torch.zeros(1, edges_seq.size()[0], self.edge_hidden_dim)                
                
                output, _ = edgeseq_layer(edges_seq, hidden)
                #print(output[-1].size())
                edge_emb = torch.mean(output, dim=1)
                #print('!!!!!!!!!!!!!!!!!!!!!', output.size())

                nb = torch.cat((edges.src['h'], edge_emb), 1)
                nb = phi_layer(nb)
                history = edges.src['history_{}'.format(i)]
                delta_nb = nb - history
                delta_nb = self.activation(delta_nb)

                return {'m': delta_nb}
    
            nf.block_compute(i,
                            message_func,
                            fn.sum(msg='m', out='delta_nb'), 
                            node_layer)
            h = nf.layers[i+1].data.pop('activation')              # input node embedding for next iteration 

            # update history
            if (i < nf.num_layers-1) and (i!=0):
                nf.layers[i].data[history_str] = new_history
        h = self.fc1(h)
        logit = self.fc2(h)
        return logit

class MiniBatchEdgePropPlusInfer(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 node_in_dim,
                 edge_in_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 fc_hidden_dim,
                 num_classes,
                 activation,
                 feat_drop,
                 residual, 
                 use_batch_norm):
        super(MiniBatchEdgePropPlusInfer, self).__init__()
        
        self.num_layers = num_layers
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.node_hidden_dim = node_hidden_dim

        self.edge_hidden_dim = edge_hidden_dim  

        # self.input_layer_e = OneLayerNN(node_in_dim=edge_in_dim, 
        #                               out_dim=num_hidden)
        self.edgeseq_layers = nn.ModuleList()
        self.edgeseq_layers.append(EdgeSeq(in_dim=edge_in_dim, 
                                            hidden_dim=self.edge_hidden_dim))


        self.node_layers = nn.ModuleList()
        self.node_layers.append(NodeUpdate(layer_id=0, 
                                        in_dim=2*node_hidden_dim, 
                                        out_dim=node_hidden_dim, 
                                        feat_drop=feat_drop, 
                                        name='node', 
                                        test=True))

        self.phi = nn.ModuleList()
        self.phi.append(SingleLayerNN(in_dim=node_in_dim + edge_hidden_dim,
                                   out_dim=node_hidden_dim))
        # self.phi.append(OneLayerNN(node_in_dim=2 * num_hidden,
        #                            out_dim=num_hidden))
        for i in range(1, num_layers):
            self.edgeseq_layers.append(EdgeSeq(in_dim=self.edge_hidden_dim, 
                                            hidden_dim=self.edge_hidden_dim))

            self.node_layers.append(NodeUpdate(layer_id=i, 
                                        in_dim=2*node_hidden_dim, 
                                        out_dim=node_hidden_dim, 
                                        feat_drop=feat_drop, 
                                        name='node', 
                                        test=True))
            self.phi.append(SingleLayerNN(in_dim=node_hidden_dim + edge_hidden_dim,
                                       out_dim=node_hidden_dim))

        
        # output projection
        self.fc1 = nn.Linear(node_hidden_dim, fc_hidden_dim, bias=True)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc1.weight.data, gain=0.1)
        nn.init.xavier_normal_(self.fc2.weight.data, gain=0.1)

    def forward(self, nodeflow):
        '''
        Given a complete embedding matrix h and a list of node IDs, return
        the output embeddings of these node IDs.

        nodeflow: NodeFlow Object
        return: new node embeddings (num_nodes, out_features)
        '''
        nf = nodeflow
        h = nf.layers[0].data['node_features'] 
        # h = self.input_layer(h)  # ((#nodes in layer_i) X D)

        for i, (edgeseq_layer, node_layer, phi_layer) in enumerate(zip(self.edgeseq_layers, self.node_layers, self.phi)):
            # compute node embedding for i+1 layer

            # compute edge embeddings
            e = nf.blocks[i].data['edge_features']
            
            e = self.feat_drop(e)
            #e = self.input_layer_e(e)
            nf.blocks[i].data['e'] = e
            #print('assign edges!!!!!!!!!!!!!!!!!!!', e.size())

            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))   # get i+1 layer nodes from parent graph
            self_layer_nid = nf.map_from_parent_nid(i, parent_nid)          # get the nodeflow id for the self edge node
            self_h = torch.cat((torch.zeros(len(self_layer_nid), self.node_hidden_dim), h[self_layer_nid]), 1)
            self_h = phi_layer(self_h)
            nf.layers[i+1].data['self_h'] = self_h # ((#nodes in layer_i+1) X D)
            nf.layers[i].data['h'] = h                  

            def message_func(edges):   
                edges_seq = edges.data['e']   
                hidden = torch.zeros(1, edges_seq.size()[0], self.edge_hidden_dim)                
                
                output, _ = edgeseq_layer(edges_seq, hidden)
                #print(output[-1].size())
                edge_emb = torch.mean(output, dim=1)
                #print('!!!!!!!!!!!!!!!!!!!!!', output.size())

                nb = torch.cat((edges.src['h'], edge_emb), 1)
                nb = phi_layer(nb)
                
                delta_nb = nb
                delta_nb = self.activation(delta_nb)

                return {'m': delta_nb}
    
            nf.block_compute(i,
                            message_func,
                            fn.sum(msg='m', out='delta_nb'), 
                            node_layer)
            h = nf.layers[i+1].data.pop('activation')              # input node embedding for next iteration 

            # update history
        embeddings = h
        h = self.fc1(h)
        logit = self.fc2(h)
        return logit, embeddings 
