import dgl
import os
from sklearn import preprocessing
import networkx as nx
import pickle 
import logging
import os
import torch
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

from sklearn.model_selection import StratifiedShuffleSplit
class TestDataset(object):
    def __init__(self):

    
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
        with open(os.path.join(data_path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        graph = data['graph']
        n_feat = data['n_feat']
        e_feat = data['e_feat']
        labels = data['labels']

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=10)

        for train_mask, test_mask in sss.split(n_feat, labels):
            y_train = labels[train_mask]
            y_test = labels[test_mask]
            break

        print(y_train)
        print(y_test)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=10)

        for val_mask, test_mask in sss.split(n_feat[test_mask], y_test):
            y_val = y_test[val_mask]
            y_test = y_test[test_mask]
            break


        logger.info('features shape: %s', str(n_feat.shape))
        logger.info('y_train shape: %s', str(y_train.shape))
        logger.info('y_val shape: %s', str(y_val.shape))
        logger.info('y_test shape: %s', str(y_test.shape))
        logger.info('train_mask shape%s: ', str(train_mask.shape))
        logger.info('val_mask shape: %s', str(val_mask.shape))
        logger.info('test_mask shape: %s', str(test_mask.shape))



        #return graph, n_feat, e_feat, labels, y_test, y_val, y_test, train_mask, val_mask, test_mask



        self.graph = dgl.DGLGraph()
        self.graph.from_networkx(nx_graph=graph)
       
        # concatenate edge attrs
        #self.graph.edata['edge_features'] = torch.cat([self.graph.edata[attr][:,None] for attr in self.graph.edge_attr_schemes().keys()], dim=1)
        #print(self.graph.edge_attr_schemes())
        # self.graph.from_scipy_sparse_matrix(spmat=self.adj)
        self.labels = labels
        self.num_labels = 2
        # self.edge_attr_adjs = adjs[1:]
        self.features = n_feat
        self.graph.edata['edge_features'] = torch.Tensor(e_feat) 
        self.graph.edata['edge_features'] = self.graph.edata['edge_features'].to(dtype=torch.float32)

        self.graph.add_edges(self.graph.nodes(), graph.nodes(), data={'edge_features': torch.zeros((self.graph.number_of_nodes(), e_feat.shape[1], e_feat.shape[2]))})
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_mask = np.zeros(len(labels), dtype=int)
        self.train_mask[train_mask.astype(int)] = 1
        self.val_mask = np.zeros(len(labels), dtype=int)
        self.val_mask[val_mask.astype(int)] = 1
        self.test_mask = np.zeros(len(labels), dtype=int)
        self.test_mask[test_mask.astype(int)] = 1
        self.num_edge_feats = e_feat.shape[1]
        #self.edge_attr_name = edge_attr_name

