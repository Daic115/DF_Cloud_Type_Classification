#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   untils.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 13:36   Daic       1.0
'''
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
import torch.optim as optim
from torch.optim import Optimizer
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def build_optimizer(params, opt):
    if opt.optimizer == 'rmsprop':
        return optim.RMSprop(params, opt.lr, opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'radam':
        return RAdam(params, opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adagrad':
        return optim.Adagrad(params, opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        return optim.SGD(params, opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgdm':
        return optim.SGD(params, opt.lr, opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgdmom':
        return optim.SGD(params, opt.lr, opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optimizer == 'adam':
        return optim.Adam(params, opt.lr,  weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def build_cls_model(name,active = 'relu',class_num=5,chanel_num=3):
    if name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
                  'efficientnet-b5','efficientnet-b6']:
        if name == 'efficientnet-b3':
            model = EfficientNet.from_pretrained('efficientnet-b3')
            model._fc = nn.Linear(in_features=1536, out_features=class_num, bias=True)
        elif name == 'efficientnet-b4':
            model = EfficientNet.from_pretrained('efficientnet-b3')
            model._fc = nn.Linear(in_features=1792, out_features=class_num, bias=True)
        elif name == 'efficientnet-b6':
            model = EfficientNet.from_pretrained('efficientnet-b6')
            model._fc = nn.Linear(in_features=2304, out_features=class_num, bias=True)
    if name not in ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6']:
        model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')
        if name in ['resnet18','resnet32']:
            if chanel_num==1:
                model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.last_linear = nn.Linear(in_features=512, out_features=class_num, bias=True)
        elif name in ['resnet50','resnet101','resnet152']:
            if chanel_num == 1:
                model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.last_linear = nn.Linear(in_features=2048, out_features=class_num, bias=True)
        elif name in ['senet154',  'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d']:
            #if chanel_num == 1:
                #model.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
            model.last_linear = nn.Linear(in_features=2048, out_features=class_num, bias=True)
        elif name in ['inceptionv4']:
            model.last_linear = nn.Linear(in_features=1536, out_features=class_num, bias=True)
        elif name in ['inceptionresnetv2']:
            model.last_linear = nn.Linear(in_features=1536, out_features=class_num, bias=True)
        else:
            raise Exception("unsupported model! {}".format(name))

    if active == 'relu':
        return model
    elif active == 'mish':
        convert_relu_to_Mish(model)
        return model

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super(LSR,self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

def build_loss_function(opt):
    if opt.loss =='lsr':
        return LSR()
    elif opt.loss =='ce':
        return nn.CrossEntropyLoss()
    elif opt.loss == 'bce':
        return nn.BCELoss()
    elif opt.loss == 'focal':
        return FocalLoss()
    elif opt.loss == 'bcel':
        return nn.BCEWithLogitsLoss()
    else:
        raise Exception("bad option opt.loss: {}".format(opt.loss))



def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

#####################################
#Pairwise Confusion for Fine-Grained Visual Classification
#https://github.com/abhimanyudubey/confusion
def PairwiseConfusion(features):
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')
    batch_left = features[:int(0.5*batch_size)]
    batch_right = features[int(0.5*batch_size):]
    loss  = torch.norm((batch_left - batch_right).abs(),2, 1).sum() / float(batch_size)

    return loss

def EntropicConfusion(features):
    batch_size = features.size(0)
    return torch.mul(features, torch.log(features)).sum() * (1.0 / batch_size)