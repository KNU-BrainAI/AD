#!/usr/bin/env python3
# author: jinhee

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LabelSmoothLoss(nn.Module):
    #https://github.com/pytorch/pytorch/issues/7455
    def __init__(self, smoothing=0.0, weight=None):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weights = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1)) # - 1.)
        weights.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weights * log_prob).sum(dim=-1).mean()
        return loss


class LSCE(nn.Module):
    #https://gist.github.com/suvojit-0x55aa/0afb3eefbb26d33f54e1fb9f94d6b609
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LSCE, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftFocal(nn.Module):
    #https://github.com/ashawkey/FocalLoss.pytorch/blob/master/
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, smoothing=0.0, weight=None):
        super(SoftFocal, self).__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        weights = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1)) # - 1.)
        weights.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        logpt = -weights * logpt
        pt = torch.exp(logpt)
        loss = ((1-pt)**self.gamma * logpt).sum(dim=-1).mean()
        return loss

class FocalLoss(nn.Module):
    #https://github.com/ashawkey/FocalLoss.pytorch/blob/master/
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

class F1_Loss(nn.Module):
    #https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        y_true, y_pred = y_true[:10], y_pred[:10]
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

