import argparse
import os

import deepspeed
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets
import time


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')

    # data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=64,
                        type=int,
                        help='mini-batch size (default:64)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet50(pretrained=False)

    def forward(self, x):
        # print("Inside: input size", x.size())
        x = self.conv(x)
        x = self.resnet(x)
        return x


def main():
    model = ResNet50()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    args = add_argument()

    train_data = datasets.MNIST(root='/home/laizhiquan/dat01/lpeng/SystemTestCode/testPytorch/data', train=True,
                                download=True,
                                transform=transforms.Compose(
                                    [transforms.Resize(224), torchvision.transforms.ToTensor()]))
    print(train_data)
    test_data = datasets.MNIST(
        root='/home/laizhiquan/dat01/lpeng/SystemTestCode/testPytorch/data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Resize(224), torchvision.transforms.ToTensor()])),

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters, training_data=train_data)

    print_peak_memory('Max memory allocated after creating deepspeed model', 0)

    start_time = time.time()
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
                model_engine.local_rank)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            print_peak_memory('Max memory allocated after forward', 0)
            model_engine.backward(loss)
            print_peak_memory('Max memory allocated after backward', 0)
            model_engine.step()
            print_peak_memory('Max memory allocated after step', 0)

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    end_time = time.time()
    if model_engine.global_rank == 0:
        print(f'Throughput:{60000 / (end_time - start_time)} samples per sec.\n')
        f = open('hostfile.txt', 'r')
        lines = f.readlines()
        f.close()
        nodes = len(lines)
        gpupernodes = lines[0].strip().split()[-1]
        f = open(f'output_node{nodes}_gres{gpupernodes}.txt', 'w')
        f.writelines(lines)
        f.write(f'Throughput:{60000 / (end_time - start_time)} samples per sec.\n')

    print('Finished Training')


if __name__ == '__main__':
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # from pycallgraph import Config
    # from pycallgraph import GlobbingFilter
    #
    # config = Config()
    # config.max_depth = 10
    # # config.include_pycallgraph=True
    # config.include_stdlib = True
    # config.trace_filter = GlobbingFilter(include=[
    #     '*',
    # ],
    #     exclude=['pycallgraph.*'])
    # with PyCallGraph(output=GraphvizOutput(), config=config):
        main()
