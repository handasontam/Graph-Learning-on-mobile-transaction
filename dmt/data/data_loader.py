import numpy as np
import pandas as pd
import dgl
import os
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import networkx as nx
import torch
import glob
import csv
from ..utils import graph_utils
from .nx_utils import get_graph_from_data
import pickle
import logging

class Dataset(object):
    def __init__(self, data_path, preprocess, directed):
        self.data_path = data_path
        self.node_features_path = os.path.join(data_path, 'features.txt')
        # self.node_features_dir = os.path.join(data_path, 'features.txt')
        # self.node_features_files = glob.glob(os.path.join(self.node_features_dir, '*'))
        self.edges_dir = os.path.join(data_path, 'adj.txt')
        with open(self.edges_dir) as f:
            self.num_edge_feats = len(f.readline().strip().split(','))-2
        self.directed = directed
        if directed:
            self.num_edge_feats = self.num_edge_feats * 2
        self.label_path = os.path.join(data_path, 'label.txt')
        self.vertex_map_path = os.path.join(data_path, 'node_id_map.csv' )
        self.train_val_test_mask = os.path.join(data_path, 'mask.txt')
        self.dgl_pickle_path = os.path.join(data_path, 'dgl_graph.pkl')
        self.preprocess = preprocess
        self.load()
    
    def load_graph(self):
        # Graph and Edge Features
        if self.preprocess:
            logging.info(f'Reading {self.edges_dir} into dgl graph')
            with open(self.edges_dir, 'r') as f:
                data = f.readlines()
                data = np.array([line.strip().split(',') for line in data]).astype(np.float32)
                logging.info(f'Number of edges found in {self.edges_dir}: {data.shape[0]}')
                data = data[np.in1d(data[:,0], list(self.feat_graph_intersec_set))]
                data = data[np.in1d(data[:,1], list(self.feat_graph_intersec_set))]
                logging.info(f'*** Number of edges after filtering : {data.shape[0]}')
                edge_from_id = data[:,0].astype(int)
                edge_to_id = data[:,1].astype(int)
                # Map vertex id to consecutive integers
                edge_from_id = np.vectorize(self.v_mapping.get)(edge_from_id)
                edge_to_id = np.vectorize(self.v_mapping.get)(edge_to_id)
                edge_features = data[:,2:]
                # Create DGL Graph
                self.g = dgl.DGLGraph()
                self.g.add_nodes(self.number_of_nodes)
                self.g.add_edges(u=edge_from_id, 
                            v=edge_to_id, 
                            data={'edge_features': torch.from_numpy(edge_features)})
                logging.info(self.g.edge_attr_schemes())

            means = self.g.edata['edge_features'].mean(dim=1, keepdim=True)
            stds = self.g.edata['edge_features'].std(dim=1, keepdim=True)
            self.g.edata['edge_features'] = (self.g.edata['edge_features'] - means) / stds
            self.g.edata['edge_features'] = self.g.edata['edge_features'].to(dtype=torch.float32)
            logging.info('Adding self-loop')
            self.g.add_edges(self.g.nodes(), self.g.nodes(), 
                    data={'edge_features': torch.zeros((self.g.number_of_nodes(), self.num_edge_feats))})

            with open(self.dgl_pickle_path, 'wb') as f:
                pickle.dump(self.g, f)
        else:
            logging.info(f'Reading dgl graph directly from {self.dgl_pickle_path}')
            with open(self.dgl_pickle_path, 'rb') as f:
                self.g= pickle.load(f)
            logging.info(f'dgl graph loaded successfully from {self.dgl_pickle_path}')
    
    def load_node_features(self):
        # Node Features
        logging.info('Loading Node Features...')
        if self.node_features_path is None:
            logging.info("\033[1;31;40mNo node features is given, use dummy featuers\033[0;37;40m ")
            features = np.ones((self.g.number_of_nodes(), 1))
        else:
            features = pd.read_csv(self.node_features_path, 
                                   delimiter=',', 
                                   header=None)
            features = features.set_index(0)
            self.features = features
    
    def load_labels(self):
        # Ground Truth Labels
        logging.info('Loading Labels...')
        self.labels = pd.read_csv(self.label_path, 
                            delimiter=',', 
                            header=None, 
                            names=['id', 'label'])

    def vertex_id_map(self):
        # Vertex id map
        logging.info('Mapping vertex id to consecutive integers')
        if self.preprocess:
            nodes_set = set()
            with open(self.edges_dir, 'r') as f:
                for line in f.readlines():
                    s = line.strip().split(',')
                    nodes_set = nodes_set.union({int(s[0])}).union({int(s[1])})
            features_set = set(self.features.index)
            self.labels_set = set(self.labels['id'])
            self.feat_graph_intersec_set = features_set.intersection(nodes_set)
            logging.info(f'Number of node features in features.txt: {len(features_set)}')
            logging.info(f'Number of nodes in adj.txt: {len(nodes_set)}')
            logging.info(f'Number of node in the intersection: {len(features_set)}')
            
            self.number_of_nodes = len(self.feat_graph_intersec_set)
            self.v_mapping = dict(zip(list(self.feat_graph_intersec_set), range(self.number_of_nodes)))  # key is vertex id, value is new vertex id

            logging.info('Node id mapping created')
            logging.info(f'Save the mapping to {self.vertex_map_path}')
            with open(self.vertex_map_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in self.v_mapping.items():
                    writer.writerow([key, value])
            logging.info(f'\033[1;32;40mVertex id mapping is sucessfully saved to {self.vertex_map_path}\033[0;37;40m ')
        else:
            logging.info(f'Loading vertex id mapping from {self.vertex_map_path}')
            v_map = pd.read_csv(self.vertex_map_path, delimiter=',', header=None)
            self.v_mapping = pd.Series(v_map[1].values,index=v_map[0]).to_dict()
            logging.info(f'\033[1;32;40mLoad vertex id mapping success')

    def preprocess_node_features(self):
        logging.info('Preprocessing node features')
        logging.info('Filtering nodes')
        self.features = self.features.loc[list(self.feat_graph_intersec_set)]
        logging.info('Mean imputation on the missing value')
        self.features = self.features.fillna(self.features.mean())
        # self.features = self.features.fillna(0)
        self.features.index = self.features.index.map(lambda x: self.v_mapping[x])
        self.features.sort_index(inplace=True)
        self.features = self.features.values

        # standardize node features and convert it to sparse matrix
        scaler = preprocessing.StandardScaler().fit(self.features)
        large_variance_column_index = np.where(scaler.var_ > 100)
        self.features[:, large_variance_column_index] = np.cbrt(self.features[:, large_variance_column_index])
        scaler = preprocessing.StandardScaler().fit(self.features)
        self.features = scaler.transform(self.features)
        logging.info(f'features shape: {self.features.shape}')
    
    def preprocess_labels(self):
        logging.info('filtering unused nodes in the label')
        self.feat_graph_label_intersec_set = self.feat_graph_intersec_set.intersection(self.labels_set)
        self.labels = self.labels.set_index('id')
        self.labels = self.labels.loc[list(self.feat_graph_label_intersec_set)]
        print(self.labels)
        logging.info('mapping labels node id to new node id')
        self.labels.index = self.labels.index.map(lambda x: self.v_mapping[x])
        self.labels = self.labels.dropna(axis='rows')
        self.labels.index = self.labels.index.astype(int)
        # convert label to one-hot format
        logging.info('convert label to one-hot format')
        one_hot_labels = pd.get_dummies(data=self.labels, dummy_na=True, columns=['label']) # N X (#edge attr)  # one hot 
        one_hot_labels = one_hot_labels.drop(['label_nan'], axis=1)
        logging.info('Train, validation, test split')
        # train, val, test split
        if os.path.exists(self.train_val_test_mask):
            logging.info(f'The mask file: {self.train_val_test_mask} exists! Reading train, val, test mask from the file')
            train_val_test_label = pd.read_csv(self.train_val_test_mask, delimiter=',', header=None, names=['id', 'mode'])
            train_val_test_label = train_val_test_label.set_index('id')
            train_val_test_label = train_val_test_label.loc[list(self.feat_graph_label_intersec_set)]
            train_val_test_label.index = train_val_test_label.index.map(lambda x: self.v_mapping[x])
            train_val_test_label = train_val_test_label.dropna(axis='rows')
            train_val_test_label.index = train_val_test_label.index.astype(int)
            train_id = set(train_val_test_label[train_val_test_label['mode'] == 'train'].index.values)
            val_id = set(train_val_test_label[train_val_test_label['mode'] == 'val'].index.values)
            test_id = set(train_val_test_label[train_val_test_label['mode'] == 'test'].index.values)
        else:
            logging.info(f'The mask file: {self.train_val_test_mask} doest not exist. Performing train, val, test split')
            train_id, test_id, y_train, y_test = train_test_split(self.labels.index, self.labels['label'], 
                                               test_size=0.2)
            train_id, val_id, y_train, y_val = train_test_split(train_id, y_train, 
                                              test_size=0.2)
        self.train_id = train_id
        self.val_id = val_id
        self.test_id = test_id
            
            
        self.train_mask = np.zeros((self.number_of_nodes,)).astype(int)
        self.val_mask = np.zeros((self.number_of_nodes,)).astype(int)
        self.test_mask = np.zeros((self.number_of_nodes,)).astype(int)

        # train_ratio = 0.8
        np.random.seed(1)
        # for column in one_hot_labels.columns:
        #     set_of_key = set(one_hot_labels[(one_hot_labels[column] == 1)].index)
        #     train_key_set = set(np.random.choice(list(set_of_key), size=int(len(set_of_key)*train_ratio), replace=False))
        #     test_key_set = set_of_key - train_key_set
        #     train_id = train_id.union(train_key_set)
        #     test_id = test_id.union(test_key_set)
        self.train_mask[list(train_id)] = 1
        self.val_mask[list(val_id)] = 1
        self.test_mask[list(test_id)] = 1

        # one_hot_labels = one_hot_labels.values[:,:-1]  # convert to numpy format and remove the nan column
        y = np.zeros(self.number_of_nodes)
        y[one_hot_labels.index] = np.argmax(one_hot_labels.values, 1)

        # y_train = np.zeros((self.number_of_nodes, one_hot_labels.shape[1]))  # one hot format
        # y_val = np.zeros((self.number_of_nodes, one_hot_labels.shape[1]))
        # y_test = np.zeros((self.number_of_nodes, one_hot_labels.shape[1]))
        # y_train[train_mask, :] = one_hot_labels.loc[sorted(train_id)]
        # y_val[val_mask, :] = one_hot_labels.loc[sorted(val_id)]
        # y_test[test_mask, :] = one_hot_labels.loc[sorted(test_id)]

        # logging.info(f'y_train shape: {y_train.shape}')
        # logging.info(f'y_val shape: {y_val.shape}')
        # logging.info(f'y_test shape: {y_test.shape}')
        logging.info(f'train_mask shape: {self.train_mask.shape}')
        logging.info(f'val_mask shape: {self.val_mask.shape}')
        logging.info(f'test_mask shape: {self.test_mask.shape}')

    def load(self):
        logging.info('loading data...')
        # Load Graph and Edge features
        # edge_attr_name = []
        # g = get_graph_from_data(self.edges_dir, True, 4)
        # logging.info(f'number of weakly connected components: {nx.algorithms.components.number_weakly_connected_components(g)}')

        # Node Features
        self.load_node_features()
        # Labels
        self.load_labels()
        # Map vertex id to consecutive integers
        self.vertex_id_map()
        # Preprocess node features
        self.preprocess_node_features()
        # Preprocess ground truth label
        self.preprocess_labels()
        # Load Graph
        self.load_graph()
        print(self.features)
        print(self.labels)
        print(self.v_mapping)

        self.num_classes = len(np.unique(self.labels))
        # import sys
        # sys.exit()
        # self.y_train = y_train
        # self.y_val = y_val
        # self.y_test = y_test
        # self.train_mask = train_mask.astype(int)
        # self.val_mask = val_mask.astype(int)
        # self.test_mask = test_mask.astype(int)
        # self.edge_attr_name = edge_attr_name
        # print(self.graph.edata)
        # print(self.labels)
        # print(self.features)
        # print(self.y_train)
        # print(self.y_val)
        # print(self.y_test)
        # print(self.edge_attr_name)