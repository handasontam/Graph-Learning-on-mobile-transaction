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
    def __init__(self, g, unsupervised_model, encoder, decoder, loss_fn, unsupervised_optimizer, decoder_optimizer, epochs, features, labels, train_mask, val_mask, test_mask, fast_mode, n_edges, patience, n_layers, num_cpu, cuda_context, model_dir='./'):
        self.g = g
        self.unsupervised_model = unsupervised_model
        self.encoder = encoder 
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.unsupervised_optimizer = unsupervised_optimizer 
        self.decoder_optimizer = decoder_optimizer
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
        classifier = LogisticRegression(solver='lbfgs', multi_class='auto').fit(X=embeds[self.train_id.detach().cpu().numpy()], 
                                                                                y=self.labels[self.train_id.detach().cpu().numpy()])

        print('Logistic Accuracy:', classifier.score(embeds[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))


        classifier = LinearSVC().fit(X=embeds[self.train_id.detach().cpu().numpy()], 
                                                  y=self.labels[self.train_id.detach().cpu().numpy()])

        print('SVC Accuracy:', classifier.score(embeds[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))

        classifier = DecisionTreeClassifier().fit(X=embeds[self.train_id.detach().cpu().numpy()], 
                                                  y=self.labels[self.train_id.detach().cpu().numpy()])

        print('Decision Tree Accuracy:', classifier.score(embeds[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))

        classifier = GradientBoostingClassifier().fit(X=embeds[self.train_id.detach().cpu().numpy()], 
                                                      y=self.labels[self.train_id.detach().cpu().numpy()])

        print('GradientBoosting Accuracy:', classifier.score(embeds[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))

        classifier = RandomForestClassifier().fit(X=embeds[self.train_id.detach().cpu().numpy()], 
                                                  y=self.labels[self.train_id.detach().cpu().numpy()])

        print('Random Forest Accuracy:', classifier.score(embeds[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))
        ############################################################################################################################################
        features = self.features.detach().cpu().numpy()
        classifier = LogisticRegression(solver='lbfgs', multi_class='auto').fit(X=features[self.train_id.detach().cpu().numpy()], 
                                                                                y=self.labels[self.train_id.detach().cpu().numpy()])
        
        print('Logistic Accuracy:', classifier.score(features[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))

        classifier = LinearSVC().fit(X=features[self.train_id.detach().cpu().numpy()], 
                                                  y=self.labels[self.train_id.detach().cpu().numpy()])

        print('SVC Accuracy:', classifier.score(features[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))

        classifier = DecisionTreeClassifier().fit(X=features[self.train_id.detach().cpu().numpy()], 
                                                  y=self.labels[self.train_id.detach().cpu().numpy()])

        print('Decision Tree Accuracy:', classifier.score(features[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))

        classifier = GradientBoostingClassifier().fit(X=features[self.train_id.detach().cpu().numpy()], 
                                                      y=self.labels[self.train_id.detach().cpu().numpy()])

        print('GradientBoosting Accuracy:', classifier.score(features[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))

        classifier = RandomForestClassifier().fit(X=features[self.train_id.detach().cpu().numpy()], 
                                                  y=self.labels[self.train_id.detach().cpu().numpy()])

        print('Random Forest Accuracy:', classifier.score(features[self.test_id.detach().cpu().numpy()], 
                               self.labels[self.test_id.detach().cpu().numpy()]))
        # import sys
        # sys.exit(0)
        # dur = []
        # for epoch in range(self.epochs):
        #     classifier.train()
        #     if epoch >= 3:
        #         t0 = time.time()

        #     self.decoder_optimizer.zero_grad()
        #     preds = classifier(embeds)
        #     loss = F.nll_loss(preds[self.train_mask], self.labels[self.train_mask])
        #     loss.backward()
        #     self.decoder_optimizer.step()
            
        #     if epoch >= 3:
        #         dur.append(time.time() - t0)

        #     acc = self.evaluate(classifier, embeds, self.labels, self.val_mask)
        #     print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #         "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                         acc, self.n_edges / np.mean(dur) / 1000))

        # print()
        # acc = self.evaluate(classifier, embeds, self.labels, self.test_mask)
        # print("Test Accuracy {:.4f}".format(acc))


# def main(args):
#     # # load and preprocess dataset
#     # data = load_data(args)
#     # features = torch.FloatTensor(data.features)
#     # labels = torch.LongTensor(data.labels)
#     # train_mask = torch.ByteTensor(data.train_mask)
#     # val_mask = torch.ByteTensor(data.val_mask)
#     # test_mask = torch.ByteTensor(data.test_mask)
#     # in_feats = features.shape[1]
#     # n_classes = data.num_labels
#     # n_edges = data.graph.number_of_edges()

#     # if args.gpu < 0:
#     #     cuda = False
#     # else:
#     #     cuda = True
#     #     torch.cuda.set_device(args.gpu)
#     #     features = features.cuda()
#     #     labels = labels.cuda()
#     #     train_mask = train_mask.cuda()
#     #     val_mask = val_mask.cuda()
#     #     test_mask = test_mask.cuda()

#     # # graph preprocess
#     # g = data.graph
#     # # add self loop
#     # if args.self_loop:
#     #     g.remove_edges_from(g.selfloop_edges())
#     #     g.add_edges_from(zip(g.nodes(), g.nodes()))
#     # g = DGLGraph(g)
#     # n_edges = g.number_of_edges()

#     # # create DGI model
#     # dgi = DGI(g,
#     #           in_feats,
#     #           args.n_hidden,
#     #           args.n_layers,
#     #           nn.PReLU(args.n_hidden),
#     #           args.dropout)

#     # if cuda:
#     #     dgi.cuda()

#     # dgi_optimizer = torch.optim.Adam(dgi.parameters(),
#     #                                  lr=args.dgi_lr,
#     #                                  weight_decay=args.weight_decay)

#     # train deep graph infomax
#     cnt_wait = 0
#     best = 1e9
#     best_t = 0
#     dur = []
#     for epoch in
#     for epoch in range(args.n_dgi_epochs):
#         dgi.train()
#         if epoch >= 3:
#             t0 = time.time()

#         dgi_optimizer.zero_grad()
#         loss = dgi(features)
#         loss.backward()
#         dgi_optimizer.step()

#         if loss < best:
#             best = loss
#             best_t = epoch
#             cnt_wait = 0
#             torch.save(dgi.state_dict(), 'best_dgi.pkl')
#         else:
#             cnt_wait += 1

#         if cnt_wait == args.patience:
#             print('Early stopping!')
#             break

#         if epoch >= 3:
#             dur.append(time.time() - t0)

#         print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
#               "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
#                                             n_edges / np.mean(dur) / 1000))

#     # create classifier model
#     classifier = Classifier(args.n_hidden, n_classes)
#     if cuda:
#         classifier.cuda()

#     classifier_optimizer = torch.optim.Adam(classifier.parameters(),
#                                             lr=args.classifier_lr,
#                                             weight_decay=args.weight_decay)

#     # train classifier
#     print('Loading {}th epoch'.format(best_t))
#     dgi.load_state_dict(torch.load('best_dgi.pkl'))
#     embeds = dgi.encoder(features, corrupt=False)
#     embeds = embeds.detach()
#     dur = []
#     for epoch in range(args.n_classifier_epochs):
#         classifier.train()
#         if epoch >= 3:
#             t0 = time.time()

#         classifier_optimizer.zero_grad()
#         preds = classifier(embeds)
#         loss = F.nll_loss(preds[train_mask], labels[train_mask])
#         loss.backward()
#         classifier_optimizer.step()
        
#         if epoch >= 3:
#             dur.append(time.time() - t0)

#         acc = evaluate(classifier, embeds, labels, val_mask)
#         print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
#               "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
#                                             acc, n_edges / np.mean(dur) / 1000))

#     print()
#     acc = evaluate(classifier, embeds, labels, test_mask)
#     print("Test Accuracy {:.4f}".format(acc))