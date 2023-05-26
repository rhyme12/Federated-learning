# Federated-learning

###1. FedBa函数

​        计算每个本地模型与全局模型之间的距离，然后根据距离计算每个本地设备模型的权重系数。然后，对所有本地模型进行加权平均，得到全局模型作为返回值。

​        这个函数的目的是为了根据每个客户端与全局模型的差异程度，调整其在联邦学习中的贡献度。如果某个客户端与全局模型的差异较大，则其权重低；如果差异较小，则其权重高。这样可以使得联邦学习更加平衡，并且能够更好地保护每个参与方的隐私。

```
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
```

### 2. 

