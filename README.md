# Federated-learning

本项目主要修改，第一，在参数聚合过程中根据每个客户端与全局模型的距离，调整其对全局模型的贡献度；第二，加入kl散度来衡量本地模型的预测值和使用刚刚下发的全局模型的预测值，将差异加到损失函数里。

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

### 2. KL散度

​        KL散度是一种衡量两个概率分布之间差异的度量方法，通过计算两个分布的相对熵来评估它们之间的差异程度。

​        为了应对本地数据的非独立同分布性质可能引发的模型偏移问题，这里采用KL散度来衡量本地模型预测值与全局模型预测值之间的差异，并将其纳入损失函数中。

```
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        global_w = copy.deepcopy(net)
        global_w.eval()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                global_output = global_w(images)
                local_probs = F.softmax(log_probs, dim=1)
                global_probs = F.softmax(global_output, dim=1)
                #
                local_probs = torch.clamp(local_probs, min=1e-10)
                global_probs = torch.clamp(global_probs, min=1e-10)
                # print(global_probs)
                kl_div = F.kl_div(local_probs.log(), global_probs.detach(), reduction='batchmean')
                # print("kl_div: ", kl_div)
                loss = loss + 0.5 * kl_div

                loss.backward()
                optimizer.step()
                scheduler.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0]
        return net, sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0]
```

