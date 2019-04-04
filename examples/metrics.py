import torch


def accuracy(logits, labels):
    # micro f1 = accuracy
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def micro_f1(logits, labels):
    return accuracy(logits, labels)

def macro_f1(logits, labels):
    raise NotImplementedError

def hamming_loss(logits, labels):
    raise NotImplementedError

def precision(logits, labels):
    raise NotImplementedError

def recall(logits, labels):
    raise NotImplementedError
