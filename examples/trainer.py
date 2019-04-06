from utils.metrics import accuracy
from utils.torch_utils import EarlyStopping
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

class Trainer(object):
    def __init__(self, model, loss_fn, optimizer, epochs, features, labels, train_mask, val_mask, test_mask, fast_mode, n_edges, patience=8, model_dir='./'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
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
        dur = []
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            if use_tensorboardx:
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    self.writer.add_histogram(name, param, epoch)
            self.model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = self.model(self.features)
            train_loss = self.loss_fn(logits[self.train_mask], self.labels[self.train_mask])
            train_losses.append(train_loss.item())

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_acc = accuracy(logits[self.train_mask], self.labels[self.train_mask])

            if self.fast_mode:
                val_acc = accuracy(logits[self.val_mask], self.labels[self.val_mask])
            else:
                val_acc = self.evaluate(self.features, self.labels, self.val_mask)
            val_loss = self.loss_fn(logits[self.val_mask], self.labels[self.val_mask])
            val_losses.append(val_loss.item())
            
            # early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                logging.info("Early stopping")
                break

            logging.info("Epoch {:05d} | Time(s) {:.4f} | TrainLoss {:.4f} | TrainAcc {:.4f} |"
                " ValAcc {:.4f} | ValLoss {:.4f} | ETputs(KTEPS) {:.2f}".
                format(epoch, np.mean(dur), train_loss.item(), train_acc,
                        val_acc, val_loss.item(), self.n_edges / np.mean(dur) / 1000))

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'checkpoint.pt')))

        # logging.info()
        acc = self.evaluate(self.features, self.labels, self.test_mask)
        logging.info("Test Accuracy {:.4f}".format(acc))


        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_losses)+1),train_losses, label='Training Loss')
        plt.plot(range(1,len(val_losses)+1),val_losses,label='Validation Loss')

        # find position of lowest validation loss
        minposs = val_losses.index(min(val_losses))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5) # consistent scale
        plt.xlim(0, len(train_losses)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.model_dir, 'loss_plot.png'), bbox_inches='tight')