from .utils.metrics import accuracy
from .utils.torch_utils import EarlyStopping
import torch
try:
    from tensorboardX import SummaryWriter
    use_tensorboardx = True
except:
    use_tensorboardx = False
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import os
from dgl.contrib.sampling import NeighborSampler
import dgl.function as fn

class MiniBatchTrainer(object):
    def __init__(self, g, model, model_infer, loss_fn, optimizer, epochs, features, labels, train_mask, val_mask, test_mask, fast_mode, n_edges, patience, batch_size, test_batch_size, num_neighbors, n_layers, model_dir='./'):
        self.g = g
        self.model = model
        self.model_infer = model_infer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # self.sched_lambda = {
        #         'none': lambda epoch: 1,
        #         'decay': lambda epoch: max(0.98 ** epoch, 1e-4),
        #         }
        # self.sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
        #                                         self.sched_lambda['none'])
        # print(train_mask.shape)
        self.train_id = train_mask.nonzero().view(-1).to(torch.int64)
        self.val_id = val_mask.nonzero().view(-1).to(torch.int64)
        self.test_id = test_mask.nonzero().view(-1).to(torch.int64)
        self.epochs = epochs
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        if use_tensorboardx:
            self.writer = SummaryWriter('/tmp/tensorboardx')
        self.fast_mode = fast_mode
        self.n_edges = n_edges
        self.patience = patience
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_neighbors = num_neighbors
        self.n_layers = n_layers
        self.model_dir = model_dir
        
        # initialize early stopping object
        self.early_stopping = EarlyStopping(patience=patience, log_dir=model_dir, verbose=True)

    def evaluate(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            return accuracy(logits, labels)

    def train(self):
        # initialize
        dur = []
        train_losses = []  # per mini-batch
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(self.epochs):
            train_losses_temp = []
            train_accuracies_temp = []
            val_losses_temp = []
            val_accuracies_temp = []
            if use_tensorboardx:
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    self.writer.add_histogram(name, param, epoch)
            # minibatch train
            train_num_correct = 0  # number of correct prediction in validation set
            train_total_losses = 0  # total cross entropy loss
            for nf in NeighborSampler(self.g, 
                                        batch_size=self.batch_size,
                                        expand_factor=self.num_neighbors,
                                        neighbor_type='in',
                                        shuffle=True,
                                        num_hops=self.n_layers,
                                        add_self_loop=False,
                                        seed_nodes=self.train_id):
                # update the aggregate history of all nodes in each layer
                for i in range(self.n_layers):
                    agg_history_str = 'agg_history_{}'.format(i)
                    self.g.pull(nf.layer_parent_nid(i+1), fn.copy_src(src='history_{}'.format(i), out='m'),
                        fn.sum(msg='m', out=agg_history_str))

                # Copy the features from the original graph to the nodeflow graph
                node_embed_names = [['features', 'history_0']]
                for i in range(1, self.n_layers):
                    node_embed_names.append(['history_{}'.format(i), 'agg_history_{}'.format(i-1), 'subg_norm', 'norm'])
                node_embed_names.append(['agg_history_{}'.format(self.n_layers-1), 'subg_norm', 'norm'])
                edge_embed_names = [['edge_features']]
                nf.copy_from_parent(node_embed_names=node_embed_names, 
                                    edge_embed_names=edge_embed_names)

                # Forward Pass, Calculate Loss and Accuracy
                self.model.train() # set to train mode
                if epoch >= 2:
                    t0 = time.time()
                logits = self.model(nf)
                batch_node_ids = nf.layer_parent_nid(-1)
                batch_size = len(batch_node_ids)
                batch_labels = self.labels[batch_node_ids]
                mini_batch_accuracy = accuracy(logits, batch_labels)
                train_num_correct += mini_batch_accuracy * batch_size
                train_loss = self.loss_fn(logits, batch_labels)
                train_total_losses += (train_loss.item() * batch_size)

                # Train
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                node_embed_names = [['history_{}'.format(i)] for i in range(self.n_layers)]
                node_embed_names.append([])

                # Copy the udpated features from the nodeflow graph to the original graph
                nf.copy_to_parent(node_embed_names=node_embed_names)

            # loss and accuracy of this epoch
            train_average_loss = train_total_losses / len(self.train_id)
            train_losses.append(train_average_loss)
            train_accuracy = train_num_correct / len(self.train_id)
            train_accuracies.append(train_accuracy)

            # copy parameter to the inference model
            if epoch >= 2:
                dur.append(time.time() - t0)

            # Validation
            val_num_correct = 0  # number of correct prediction in validation set
            val_total_losses = 0  # total cross entropy loss
            for nf in NeighborSampler(self.g, 
                                        batch_size=len(self.val_id),
                                        expand_factor=self.g.number_of_nodes(),
                                        neighbor_type='in',
                                        num_hops=self.n_layers,
                                        seed_nodes=self.val_id,
                                        add_self_loop=False):
                # in testing/validation, no need to update the history
                node_embed_names = [['features']]
                edge_embed_names = [['edge_features']]
                for i in range(self.n_layers):
                    node_embed_names.append(['norm', 'subg_norm'])
                nf.copy_from_parent(node_embed_names=node_embed_names, 
                                    edge_embed_names=edge_embed_names)
                self.model_infer.load_state_dict(self.model.state_dict())
                logits = self.model_infer(nf)
                batch_node_ids = nf.layer_parent_nid(-1)
                batch_size = len(batch_node_ids)
                batch_labels = self.labels[batch_node_ids]
                mini_batch_accuracy = accuracy(logits, batch_labels)
                val_num_correct += mini_batch_accuracy * batch_size
                mini_batch_val_loss = self.loss_fn(logits, batch_labels)
                val_total_losses += (mini_batch_val_loss.item() * batch_size)

            # loss and accuracy of this epoch
            val_average_loss = val_total_losses / len(self.val_id)
            val_losses.append(val_average_loss)
            val_accuracy = val_num_correct / len(self.val_id)
            val_accuracies.append(val_accuracy)

            # early stopping
            self.early_stopping(val_average_loss, self.model_infer)
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                break

            # if epoch == 25:
            #     # switch to sgd with large learning rate
            #     # https://arxiv.org/abs/1706.02677
            #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
            #     self.sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.sched_lambda['decay'])
            # elif epoch < 25:
            #     self.sched.step()
            logging.info("Epoch {:05d} | Time(s) {:.4f} | TrainLoss {:.4f} | TrainAcc {:.4f} |"
                " ValLoss {:.4f} | ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                format(epoch, np.mean(dur), train_average_loss, train_accuracy,
                        val_average_loss, val_accuracy, self.n_edges / np.mean(dur) / 1000))

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'checkpoint.pt')))

        # # logging.info()
        # acc = self.evaluate(self.features, self.labels, self.test_mask)
        # logging.info("Test Accuracy {:.4f}".format(acc))

        self.plot(train_losses, val_losses, train_accuracies, val_accuracies)

    def plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        #####################################################################
        ##################### PLOT ##########################################
        #####################################################################
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_losses)+1),np.log(train_losses), label='Training Loss')
        plt.plot(range(1,len(val_losses)+1),np.log(val_losses),label='Validation Loss')

        # find position of lowest validation loss
        minposs = val_losses.index(min(val_losses))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('log cross entropy loss')
        plt.xlim(0, len(train_losses)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.model_dir, 'loss_plot.png'), bbox_inches='tight')


        # accuracy plot
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_accuracies)+1),train_accuracies, label='Training accuracies')
        plt.plot(range(1,len(val_accuracies)+1),val_accuracies,label='Validation accuracies')

        # find position of lowest validation loss
        minposs = val_losses.index(min(val_losses))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('accuracies')
        plt.xlim(0, len(train_accuracies)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.model_dir, 'accuracies_plot.png'), bbox_inches='tight')
