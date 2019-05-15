import numpy as np
import pandas as pd
import dgl
import os
import scipy.sparse as sp
from sklearn import preprocessing
import networkx as nx
import torch
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp


class SimpleDataset(object):
    def __init__(self):
        simple_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'simple_data')
        self.node_features_path = os.path.join(simple_data_path, 'features.csv')
        self.edges_dir = os.path.join(simple_data_path, 'adj.csv' )
        self.label_path = os.path.join(simple_data_path, 'label.csv')
        self.vertex_map_path = os.path.join(simple_data_path, 'node_id_map.txt' )
        self.load()

    def load(self):
        logger.info('loading data')
        # Edge features
        edge_attr_name = []

        g = nx.readwrite.edgelist.read_edgelist(self.edges_dir, 
                                            delimiter=',', 
                                            data=[
                                                    ('weight', float), 
                                                ], 
                                            comments='#', 
                                            create_using=nx.DiGraph())

        v_map = pd.read_csv(self.vertex_map_path, delimiter=',', header=None)
        mapping = pd.Series(v_map[1].values,index=v_map[0]).to_dict()
        g = nx.relabel.relabel_nodes(g, mapping)

        logger.info('number of connected components: %d'%nx.algorithms.components.number_weakly_connected_components(g))
        # Node Features
        features = pd.read_csv(self.node_features_path, delimiter=',', header=None).values

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

        y = np.zeros(size)
        y[one_hot_labels.index] = np.argmax(one_hot_labels.values, 1)

        y_train = np.zeros((size, one_hot_labels.shape[1]))  # one hot format
        y_val = np.zeros((size, one_hot_labels.shape[1]))
        y_test = np.zeros((size, one_hot_labels.shape[1]))
        y_train[train_mask, :] = one_hot_labels.loc[sorted(train_id)]
        y_val[val_mask, :] = one_hot_labels.loc[sorted(test_id)]
        y_test[test_mask, :] = one_hot_labels.loc[sorted(test_id)]
        print(train_mask, val_mask)

        logger.info('features shape: %s', str(features.shape))
        logger.info('y_train shape: %s', str(y_train.shape))
        logger.info('y_val shape: %s', str(y_val.shape))
        logger.info('y_test shape: %s', str(y_test.shape))
        logger.info('train_mask shape%s: ', str(train_mask.shape))
        logger.info('val_mask shape: %s', str(val_mask.shape))
        logger.info('test_mask shape: %s', str(test_mask.shape))

        # self.adj = adjs[0]
        self.graph = dgl.DGLGraph()
        self.graph.from_networkx(nx_graph=g, edge_attrs=['weight'])
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