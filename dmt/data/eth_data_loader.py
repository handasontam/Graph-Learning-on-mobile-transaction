import numpy as np
import pandas as pd
import dgl
import os
import scipy.sparse as sp
from sklearn import preprocessing
import networkx as nx
import torch

class EthDataset(object):
    def __init__(self):
        eth_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eth')
        self.node_features_path = os.path.join(eth_data_path, 'features.csv')
        self.edges_dir = os.path.join(eth_data_path, 'adj.csv' )
        self.label_path = os.path.join(eth_data_path, 'label.csv')
        self.vertex_map_path = os.path.join(eth_data_path, 'node_id_map.txt' )
        self.load()

    def load(self):
        print('loading data')
        # Edge features
        edge_attr_name = []

        g = nx.readwrite.edgelist.read_edgelist(self.edges_dir, 
                                            delimiter=',', 
                                            data=[
                                                    ('sum_value', float), 
                                                    ('avg_value', float), 
                                                    ('var_value', float), 
                                                    ('count_trans', float), 
                                                    ('count_call', float), 
                                                    ('count_mining', float), 
                                                    ('count_other', float)
                                                ], 
                                            comments='#', 
                                            create_using=nx.DiGraph)

        v_map = pd.read_csv(self.vertex_map_path, delimiter=' ', header=None)
        mapping = pd.Series(v_map[1].values,index=v_map[0]).to_dict()
        g = nx.relabel.relabel_nodes(g, mapping)

        print('number of connected components: ', nx.algorithms.components.number_weakly_connected_components(g))
        # Node Features
        if self.node_features_path is None:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!! No node features is given, use dummy featuers!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            features = np.ones((g.number_of_nodes(), 10))
        else:
            features = pd.read_csv(self.node_features_path, delimiter=',').values
            # Log transform
            # features = np.log(features)
            print('variance of the features: ', np.var(features, axis=0))
            # cubic root transform
            # features = np.cbrt(features)
            features = np.power(features, 1/3)
            print('variance of the features after transformation: ', np.var(features, axis=0))

        # Ground Truth label
        labels = pd.read_csv(self.label_path, delimiter=',')
        
        # convert label to one-hot format
        one_hot_labels = pd.get_dummies(data=labels, dummy_na=True, columns=['label']).set_index('id') # N X (#edge attr)  # one hot 
        # print(labels.columns)
        one_hot_labels = one_hot_labels.drop(['label_nan'], axis=1)

        size = features.shape[0]

        train_id = set()
        test_id = set()
        train_mask = np.zeros((size,)).astype(bool)
        val_mask = np.zeros((size,)).astype(bool)
        test_mask = np.zeros((size,)).astype(bool)

        train_ratio = 0.8
        np.random.seed(1)
        for column in one_hot_labels.columns:
            set_of_key = set(one_hot_labels[(one_hot_labels[column] == 1)].index)
            train_key_set = set(np.random.choice(list(set_of_key), size=int(len(set_of_key)*train_ratio), replace=False))
            test_key_set = set_of_key - train_key_set
            train_id = train_id.union(train_key_set)
            test_id = test_id.union(test_key_set)
        train_mask[list(train_id)] = 1
        val_mask[list(test_id)] = 1
        test_mask[list(test_id)] = 1

        # one_hot_labels = one_hot_labels.values[:,:-1]  # convert to numpy format and remove the nan column
        y = np.zeros(size)
        y[one_hot_labels.index] = np.argmax(one_hot_labels.values, 1)

        y_train = np.zeros((size, one_hot_labels.shape[1]))  # one hot format
        y_val = np.zeros((size, one_hot_labels.shape[1]))
        y_test = np.zeros((size, one_hot_labels.shape[1]))
        y_train[train_mask, :] = one_hot_labels.loc[sorted(train_id)]
        y_val[val_mask, :] = one_hot_labels.loc[sorted(test_id)]
        y_test[test_mask, :] = one_hot_labels.loc[sorted(test_id)]



        # # standardize node features and convert it to sparse matrix
        # # print(features)
        # # print(features[train_mask])
        # scaler = preprocessing.StandardScaler().fit(features)
        # print(scaler.mean_)
        # print(scaler.var_)
        # # scaler = preprocessing.MinMaxScaler(feature_range=(-2, 2), copy=True).fit(features)
        # features = scaler.transform(features)
        # # features = sp.csr_matrix(features)

        # print('adjs length: ', len(adjs))
        print('features shape: ', features.shape)
        print('y_train shape: ', y_train.shape)
        print('y_val shape: ', y_val.shape)
        print('y_test shape: ', y_test.shape)
        print('train_mask shape: ', train_mask.shape)
        print('val_mask shape: ', val_mask.shape)
        print('test_mask shape: ', test_mask.shape)

        # self.adj = adjs[0]
        self.graph = dgl.DGLGraph()
        self.graph.from_networkx(nx_graph=g, edge_attrs=['avg_value', 'count_call', 'count_mining', 
                                                # 'count_other', 
                                                'count_trans', 'sum_value', 'var_value'])
        self.num_edge_feats = len(self.graph.edge_attr_schemes())
        # standardize edge attrs
        for attr in self.graph.edge_attr_schemes().keys():
            # self.graph.edata[attr] = (self.graph.edata[attr] - torch.mean(self.graph.edata[attr])) / torch.var(self.graph.edata[attr])
            self.graph.edata[attr] = torch.pow(self.graph.edata[attr], 1/3)
        # concatenate edge attrs
        self.graph.edata['edge_features'] = torch.cat([self.graph.edata[attr][:,None] for attr in self.graph.edge_attr_schemes().keys()], dim=1)
        print(self.graph.edge_attr_schemes())
        # self.graph.from_scipy_sparse_matrix(spmat=self.adj)
        self.labels = y
        self.num_labels = one_hot_labels.shape[1]
        # self.edge_attr_adjs = adjs[1:]
        self.features = features
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_mask = train_mask.astype(int)
        self.val_mask = val_mask.astype(int)
        self.test_mask = test_mask.astype(int)
        self.edge_attr_name = edge_attr_name