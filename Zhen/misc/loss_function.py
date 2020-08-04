import torch
import functools
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse


def BCE_Loss(input, target):
    if input.shape != target.shape:
        input = input.view(target.shape)

    if target.dtype != input.dtype:
        target = target.type(input.dtype)

    loss = F.binary_cross_entropy(input, target)

    return loss


def Margin_Loss(inputs, target):
    """
    :param input: list[N, N, N....]
    :param target: N
    :return:
    """
    assert torch.equal(torch.unique(target), torch.tensor([0, 1], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([0], dtype=torch.int64).to(target.device)) or \
           torch.equal(torch.unique(target), torch.tensor([1], dtype=torch.int64).to(target.device))

    x2 = target.clone()
    loss = F.margin_ranking_loss(inputs.squeeze(), x2.detach(), target, reduction='mean')

    return loss


def Contrastive_loss(label, euclidean_distance, margin=1.0):
    loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2) +
                                  (label)*torch.pow(torch.clamp(margin-euclidean_distance, min=0.0), 2))
    return loss_contrastive


# Soft-margin focal loss
def criterion_margin_focal_binary_cross_entropy(logit, truth):
    weight_pos = 2
    weight_neg = 1
    gamma = 2
    margin = 0.2
    em = np.exp(margin)

    logit = logit.view(-1)
    truth = truth.view(-1)
    log_pos = -F.logsigmoid(logit)
    log_neg = -F.logsigmoid(-logit)

    log_prob = truth*log_pos + (1-truth)*log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em + (1-em)*prob)

    weight = truth*weight_pos + (1-truth)*weight_neg
    loss = margin + weight*(1 - prob) ** gamma * log_prob

    loss = loss.mean()
    return loss


key2loss = {'bce_loss': BCE_Loss}


def get_loss_fun(loss_dict):
    if loss_dict is None:
        return BCE_Loss

    else:
        loss_name = loss_dict["name"]

        if loss_name not in key2loss:
            raise NotImplementedError("{} function not implemented".format(loss_name))
        else:
            loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        return functools.partial(key2loss[loss_name], **loss_params)
