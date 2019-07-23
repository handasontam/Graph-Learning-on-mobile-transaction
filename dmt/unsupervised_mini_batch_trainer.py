from .utils.metrics import torch_accuracy, accuracy, micro_f1, macro_f1, hamming_loss, micro_precision, micro_recall, macro_precision, macro_recall
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class UnsupervisedMiniBatchTrainer(object):
    def __init__(self, g, unsupervised_model, unsupervised_model_infer, encoder, encoder_infer, loss_fn, optimizer, epochs, features, labels, train_id, val_id, test_id, fast_mode, n_edges, patience, batch_size, test_batch_size, num_neighbors, n_layers, num_cpu, cuda_context, model_dir='./'):
        self.g = g
        self.unsupervised_model = unsupervised_model
        self.unsupervised_model_infer = unsupervised_model_infer
        self.encoder = encoder
        self.encoder_infer = encoder_infer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_id = train_id
        self.val_id = val_id
        self.test_id = test_id
        self.epochs = epochs
        self.features = features
        self.labels = labels
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
        self.num_cpu = num_cpu
        self.cuda_context = cuda_context

        # initialize early stopping object
        self.early_stopping = EarlyStopping(
            patience=patience, log_dir=model_dir, verbose=True)

    def train(self):
        # initialize
        dur = []
        train_losses = []  # per mini-batch
        losses = []

        if not self.fast_mode:
            for epoch in range(self.epochs):
                train_losses_temp = []
                if use_tensorboardx:
                    for i, (name, param) in enumerate(self.unsupervised_model.named_parameters()):
                        self.writer.add_histogram(name, param, epoch)
                # minibatch train
                train_total_losses = 0  # total cross entropy loss
                if epoch >= 2:
                    t0 = time.time()
                for nf in NeighborSampler(self.g,
                                        batch_size=self.batch_size,
                                        expand_factor=self.num_neighbors,
                                        neighbor_type='in',
                                        shuffle=True,
                                        num_hops=self.n_layers,
                                        add_self_loop=True,
                                        seed_nodes=None, 
                                        num_workers=self.num_cpu):
                    # Copy the features from the original graph to the nodeflow graph
                    node_embed_names = [['node_features', 'subg_norm', 'norm']]
                    for i in range(1, self.n_layers):
                        node_embed_names.append(['subg_norm', 'norm'])
                    node_embed_names.append(['subg_norm', 'norm'])
                    edge_embed_names = [['edge_features']]
                    for i in range(1, self.n_layers):
                        edge_embed_names.append(['edge_features'])
                    nf.copy_from_parent(node_embed_names=node_embed_names,
                                        edge_embed_names=edge_embed_names,
                                        ctx=self.cuda_context)

                    # Forward Pass, Calculate Loss and Accuracy
                    self.unsupervised_model.train()  # set to train mode
                    train_loss = self.unsupervised_model(nf)
                    batch_node_ids = nf.layer_parent_nid(-1)
                    batch_size = len(batch_node_ids)
                    train_total_losses += (train_loss.item() * batch_size)

                    # Train
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()

                    torch.cuda.empty_cache()

                # copy parameter to the inference model
                if epoch >= 2:
                    dur.append(time.time() - t0)

                # loss and accuracy of this epoch
                train_average_loss = train_total_losses / self.g.number_of_nodes()
                train_losses.append(train_average_loss)
                logging.info("Epoch {:05d} | Time(s) {:.4f} | TrainLoss {:.4f} | ETputs(KTEPS) {:.2f}\n".
                            format(epoch, np.mean(dur), train_average_loss, self.n_edges / np.mean(dur) / 1000))

                if epoch % 10 == 0:
                    total_losses = 0  # total cross entropy loss:
                    self.unsupervised_model_infer.load_state_dict(
                        self.unsupervised_model.state_dict())
                    for nf in NeighborSampler(self.g,
                                            batch_size=self.batch_size,
                                            expand_factor=self.g.number_of_nodes(),
                                            neighbor_type='in',
                                            shuffle=False, 
                                            num_hops=self.n_layers,
                                            add_self_loop=True,
                                            seed_nodes=None, 
                                            num_workers=self.num_cpu):
                        # Copy the features from the original graph to the nodeflow graph
                        node_embed_names = [['node_features', 'subg_norm', 'norm']]
                        for i in range(1, self.n_layers):
                            node_embed_names.append(['subg_norm', 'norm'])
                        node_embed_names.append(['subg_norm', 'norm'])
                        edge_embed_names = [['edge_features']]
                        for i in range(1, self.n_layers):
                            edge_embed_names.append(['edge_features'])
                        nf.copy_from_parent(node_embed_names=node_embed_names,
                                            edge_embed_names=edge_embed_names,
                                            ctx=self.cuda_context)

                        loss = self.unsupervised_model_infer(nf)
                        batch_node_ids = nf.layer_parent_nid(-1)
                        batch_size = len(batch_node_ids)
                        total_losses += (loss.item() * batch_size)

                        torch.cuda.empty_cache()
                    average_loss = total_losses / self.g.number_of_nodes()
                    losses.append(average_loss)
                    logging.info("************** Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | ETputs(KTEPS) {:.2f} ***************\n".
                                format(epoch, np.mean(dur), average_loss, self.n_edges / np.mean(dur) / 1000))

                    # early stopping
                    self.early_stopping(average_loss, self.unsupervised_model_infer)
                if self.early_stopping.early_stop:
                    logging.info("Early stopping")
                    break

        # # embeddings visualization
        # if use_tensorboardx:
        #     self.writer.add_embedding(embeddings, global_step=epoch, metadata=batch_labels)

        # load the last checkpoint with the best model
        self.unsupervised_model_infer.load_state_dict(torch.load(
            os.path.join(self.model_dir, 'checkpoint.pt')))

        embeds = []
        for nf in NeighborSampler(self.g,
                                    batch_size=self.batch_size,
                                    expand_factor=self.g.number_of_nodes(),
                                    neighbor_type='in',
                                    shuffle=False, 
                                    num_hops=self.n_layers,
                                    add_self_loop=True,
                                    seed_nodes=None, 
                                    num_workers=self.num_cpu):
            # Copy the features from the original graph to the nodeflow graph
            node_embed_names = [['node_features', 'subg_norm', 'norm']]
            for i in range(1, self.n_layers):
                node_embed_names.append(['subg_norm', 'norm'])
            node_embed_names.append(['subg_norm', 'norm'])
            edge_embed_names = [['edge_features']]
            for i in range(1, self.n_layers):
                edge_embed_names.append(['edge_features'])
            nf.copy_from_parent(node_embed_names=node_embed_names,
                                edge_embed_names=edge_embed_names,
                                ctx=self.cuda_context)
            embed = self.unsupervised_model_infer.encoder(nf, corrupt=False).detach().cpu().numpy()
            embeds.extend(embed)
            batch_node_ids = nf.layer_parent_nid(-1)
            torch.cuda.empty_cache()
        embeds = np.array(embeds)
        print('learnt embeddings:', embeds)

        # train classifier
        # print('Loading {}th epoch'.format(best_t))
        # self.unsupervised_model.load_state_dict(torch.load(
        #     os.path.join(self.model_dir, 'best_unsupervised.pkl')))
        # embeds = self.unsupervised_model_infer.encoder(self.features, corrupt=False)
        # embeds = embeds.detach().cpu().numpy()

        classifiers = {
                       'Logistic': LogisticRegression(solver='lbfgs', multi_class='ovr'), 
                       'LinearSVC': LinearSVC(), 
                       'DecisionTree': DecisionTreeClassifier(), 
                       'GradientBoosting': GradientBoostingClassifier(), 
                       'RandomForest': RandomForestClassifier(), 
                        }
        features = self.features.detach().cpu().numpy()

        dgi_X_train = embeds[self.train_id]
        X_train = features[self.train_id]
        y_train = self.labels.loc[self.train_id]['label']

        dgi_X_test = embeds[self.test_id]
        X_test = features[self.test_id]
        y_test = self.labels.loc[self.test_id]['label']
        for name, classifier in classifiers.items():
            print('DGI -', name, ' : ', classifier.fit(X=dgi_X_train, y=y_train).score(dgi_X_test, y_test))
        
        for name, classifier in classifiers.items():
            print(name, ' : ', classifier.fit(X=X_train, y=y_train).score(X_test, y_test))

    def plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        #####################################################################
        ##################### PLOT ##########################################
        #####################################################################
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_losses)+1),
                 np.log(train_losses), label='Training Loss')
        plt.plot(range(1, len(val_losses)+1),
                 np.log(val_losses), label='Validation Loss')

        # find position of lowest validation loss
        minposs = val_losses.index(min(val_losses))+1
        plt.axvline(minposs, linestyle='--', color='r',
                    label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('log cross entropy loss')
        plt.xlim(0, len(train_losses)+1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.model_dir, 'loss_plot.png'),
                    bbox_inches='tight')

        # accuracy plot
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_accuracies)+1),
                 train_accuracies, label='Training accuracies')
        plt.plot(range(1, len(val_accuracies)+1),
                 val_accuracies, label='Validation accuracies')

        # find position of lowest validation loss
        minposs = val_losses.index(min(val_losses))+1
        plt.axvline(minposs, linestyle='--', color='r',
                    label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('accuracies')
        plt.xlim(0, len(train_accuracies)+1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.model_dir,
                                 'accuracies_plot.png'), bbox_inches='tight')
