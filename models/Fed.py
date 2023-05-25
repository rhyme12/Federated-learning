import copy
import torch
from torch import nn
import math
import numpy as np

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedBa(w_global, w_locals):
    a = []
    for i, w_local in enumerate(w_locals):
        weights_k = weight_flatten(w_local)
        weights_i = weight_flatten(w_global)
        sub = (weights_k - weights_i).view(-1)
        sub = torch.dot(sub, sub).detach().cpu()
        sub = abs(np.arctan(sub))
        sub = math.exp(math.cos(sub))
        a.append(sub)
    sum_a = sum(a)
    coef = [b / sum_a for b in a]

    print("得到的聚合权重", coef, sum(coef))

    for param in w_global.parameters():
        param.data.zero_()

    for j, w_local in enumerate(w_locals):
        for param, param_j in zip(w_global.parameters(), w_local.parameters()):
            param.data += coef[j] * param_j
    return w_global


def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params