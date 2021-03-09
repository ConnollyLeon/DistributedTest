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


model = ResNet50()
criterion = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
args = add_argument()

train_data = datasets.MNIST(root='/home/laizhiquan/dat01/lpeng/SystemTestCode/testPytorch/data', train=True,
                            download=True,
                            transform=transforms.Compose([transforms.Resize(224), torchvision.transforms.ToTensor()]))
print(train_data)
test_data = datasets.MNIST(
    root='/home/laizhiquan/dat01/lpeng/SystemTestCode/testPytorch/data',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.Resize(224), torchvision.transforms.ToTensor()])),

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_data)


start_time = time.time()
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

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
    f.write(f'Throughput:{60000/(end_time-start_time)} samples per sec.\n')


print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=2)


def test(epoch):
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    running_accuracy = 0.0
    total = 0
    count = 0
    for data, target in test_loader:
        if args.with_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item()  # sum up batch loss 把所有loss值进行累加
        test_output = torch.max(output, dim=1)[1]  # get the index of the max log-probability
        # pred = output.data.max(1, keepdim=True)[1]
        correct += torch.sum(torch.eq(test_output, target)).item()
        total += target.size(0)
        count += 1
        # correct += test_output.eq(target.data.view_as(test_output)).cpu().sum()  # 对预测正确的数据个数进行累加
        # running_accuracy += torch.sum(torch.eq(target, test_output)).item() / target.cpu().numpy().size

    test_loss /= count  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}% )\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), ))
