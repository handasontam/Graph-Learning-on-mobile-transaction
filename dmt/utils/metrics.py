import torch
from sklearn import metrics


def torch_accuracy(logits, labels):
    # micro f1 = accuracy
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    accuracy = correct.item() * 1.0 / len(labels)
    # print(accuracy)
    # print(metrics.accuracy_score(labels, indices.cpu().detach().numpy()))
    return accuracy

def torch_micro_f1(logits, labels):
    _, indices = torch.max(logits, dim=1)
    pred = indices.cpu().detach().numpy()
    return metrics.f1_score(labels, pred, average='micro')

def torch_macro_f1(logits, labels):
    _, indices = torch.max(logits, dim=1)
    pred = indices.cpu().detach().numpy()
    return metrics.f1_score(labels, pred, average='micro')

def torch_hamming_loss(logits, labels):
    raise NotImplementedError

def torch_precision(logits, labels):
    _, indices = torch.max(logits, dim=1)
    pred = indices.cpu().detach().numpy()
    return metrics.precision_score(labels, pred, average='macro')

def torch_recall(logits, labels):
    _, indices = torch.max(logits, dim=1)
    pred = indices.cpu().detach().numpy()
    return metrics.recall_score(labels, pred, average='macro')




def accuracy(pred, labels):
    # micro f1 = accuracy
    # print(metrics.accuracy_score(labels, pred))
    return metrics.accuracy_score(labels, pred)

def micro_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='micro')

def macro_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='macro')

def hamming_loss(pred, labels):
    raise NotImplementedError

def micro_precision(pred, labels):
    return metrics.precision_score(labels, pred, average='micro')

def micro_recall(pred, labels):
    return metrics.recall_score(labels, pred, average='micro')

def macro_precision(pred, labels):
    return metrics.precision_score(labels, pred, average='macro')

def macro_recall(pred, labels):
    return metrics.recall_score(labels, pred, average='macro')