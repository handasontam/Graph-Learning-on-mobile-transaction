from examples.metrics import accuracy
import torch
from tensorboardX import SummaryWriter
import numpy as np
import time

class Trainer(object):
    def __init__(self, model, loss_fn, optimizer, epochs, features, labels, train_mask, val_mask, test_mask, fast_mode, n_edges):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.writer = SummaryWriter('/tmp/tensorboardx')
        self.fast_mode = fast_mode
        self.n_edges = n_edges

    def evaluate(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            return accuracy(logits, labels)

    def train(self):
        dur = []
        for epoch in range(self.epochs):
            for i, (name, param) in enumerate(self.model.named_parameters()):
                self.writer.add_histogram(name, param, epoch)
            self.model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = self.model(self.features)
            loss = self.loss_fn(logits[self.train_mask], self.labels[self.train_mask])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_acc = accuracy(logits[self.train_mask], self.labels[self.train_mask])

            if self.fast_mode:
                val_acc = accuracy(logits[self.val_mask], self.labels[self.val_mask])
            else:
                val_acc = self.evaluate(self.features, self.labels, self.val_mask)

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                format(epoch, np.mean(dur), loss.item(), train_acc,
                        val_acc, self.n_edges / np.mean(dur) / 1000))

        print()
        acc = self.evaluate(self.features, self.labels, self.test_mask)
        print("Test Accuracy {:.4f}".format(acc))