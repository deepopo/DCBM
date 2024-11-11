
import numpy as np
import torch
import torch.nn.functional as F

def js_div(p_output, q_output, get_softmax=True, reduction = 'batchmean'):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = torch.nn.KLDivLoss(reduction=reduction)
    if get_softmax:
        p_output = F.softmax(p_output, dim=1).clip(10e-3)
        q_output = F.softmax(q_output, dim=1).clip(10e-3)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.to(output.device)
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = output >= 0.5
    #print(list(output.data.cpu().numpy()))
    #print(list(pred.data[0].numpy()))
    #print(list(target.data[0].numpy()))
    #print(pred.size(), target.size())
    acc = (pred.int()).eq(target.int()).sum()
    acc = acc*100 / np.prod(np.array(target.size()))
    return acc