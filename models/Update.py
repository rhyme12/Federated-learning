import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


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
