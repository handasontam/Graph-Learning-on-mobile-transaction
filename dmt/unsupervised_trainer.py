import argparse, time
from .utils.metrics import torch_accuracy, accuracy, micro_f1, macro_f1, hamming_loss, micro_precision, micro_recall, macro_precision, macro_recall
from .utils.torch_utils import EarlyStopping
import numpy as np
import torch
try:
    from tensorboardX import SummaryWriter
    use_tensorboardx = True
except:
    use_tensorboardx = False
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from .models.unsupervised.DGI import DGI, Classifier
import os.path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class UnsupervisedTrainer(object):
    def __init__(self, g, unsupervised_model, encoder, decoder, loss_fn, unsupervised_optimizer, decoder_optimizer, epochs, features, labels, train_id, val_id, test_id, fast_mode, n_edges, patience, n_layers, num_cpu, cuda_context, model_dir='./'):
        self.g = g
        self.unsupervised_model = unsupervised_model
        self.encoder = encoder 
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.unsupervised_optimizer = unsupervised_optimizer 
        self.decoder_optimizer = decoder_optimizer
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
        self.n_layers = n_layers
        self.model_dir = model_dir
        self.num_cpu = num_cpu
        self.cuda_context = cuda_context
        
        # initialize early stopping object
        self.early_stopping = EarlyStopping(patience=patience, log_dir=model_dir, verbose=True)

    def evaluate(self, decoder, features, labels, mask):
        decoder.eval()
        with torch.no_grad():
            logits = decoder(features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)
    
    def train(self):
        # train deep graph infomax
        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []
        if not os.path.isfile(os.path.join(self.model_dir, 'best_unsupervised.pkl')):
            for epoch in range(self.epochs):
                self.encoder.train()
                if epoch >= 3:
                    t0 = time.time()

                self.unsupervised_optimizer.zero_grad()
                loss = self.unsupervised_model(self.features)
                loss.backward()
                self.unsupervised_optimizer.step()

                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(self.unsupervised_model.state_dict(), os.path.join(self.model_dir, 'best_unsupervised.pkl'))
                else:
                    cnt_wait += 1

                if cnt_wait == self.patience:
                    print('Early stopping!')
                    break

                if epoch >= 3:
                    dur.append(time.time() - t0)

                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                    "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                    self.n_edges / np.mean(dur) / 1000))

        # create classifier model
        classifier = self.decoder


        # train classifier
        print('Loading {}th epoch'.format(best_t))
        self.unsupervised_model.load_state_dict(torch.load(os.path.join(self.model_dir, 'best_unsupervised.pkl')))
        embeds = self.encoder(self.features, corrupt=False)
        embeds = embeds.detach().cpu().numpy()

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
