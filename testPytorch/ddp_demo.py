"""run.py:"""
# !/usr/bin/env python

# !/usr/bin/env python
import os
from random import Random

import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing import Process
from torchvision import datasets, transforms, models
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = models.resnet50(pretrained=False)

    def forward(self, x):
        # print("Inside: input size", x.size())
        x = self.conv(x)
        x = self.resnet(x)
        return x


""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


""" Distributed Synchronous SGD Example """


# def run(rank, size):
#     torch.manual_seed(1234)
#     train_set, bsz = partition_dataset()
#     model = ResNet50()
#     optimizer = torch.optim.SGD(model.parameters(),
#                                 lr=0.01, momentum=0.5)
#
#     num_batches = torch.ceil(len(train_set.dataset) / float(bsz))
#     for epoch in range(10):
#         epoch_loss = 0.0
#         for data, target in train_set:
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             epoch_loss += loss.item()
#             loss.backward()
#             average_gradients(model)
#             optimizer.step()
#         print('Rank ', dist.get_rank(), ', epoch ',
#               epoch, ': ', epoch_loss / num_batches)


def run(rank, size):
    """Blocking point-to-point communication."""
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


class Partition(object):
    """ Dataset partitioning helper """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning MNIST """


def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
    return train_set, bsz


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":

    size = 4
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
