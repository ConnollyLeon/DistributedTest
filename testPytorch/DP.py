from __future__ import print_function  # 这个是python当中让print都以python3的形式进行print，即把print视为函数

import argparse
import time

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Training settings 就是在设置一些参数，每个都有默认值，输入python main.py -h可以获得相关帮助
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    # batch_size参数，如果想改，如改成128可这么写：python main.py -batch_size=128
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',  # test_batch_size参数，
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,  # GPU参数，默认为False
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',  # 跑多少次batch进行一次日志记录
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-tensorboard', action='store_true', default=False,
                    help='activate tensorboard to summarize training')

args = parser.parse_args()
torch.manual_seed(0)
if torch.cuda.is_available() and not args.no_cuda:
    use_gpu = True
    print("Using GPU...")
    torch.cuda.manual_seed(0)  # Set a seed to make result consistent
else:
    print("Using CPU...")
    use_gpu = False

if torch.cuda.device_count() > 1:
    multi_gpu = True
else:
    multi_gpu = False

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.Resize(224), torchvision.transforms.ToTensor()])),
    # transform=transforms.Compose([
    #     transforms.Resize(224),  # resnet默认图片输入大小224*224
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Resize(224), torchvision.transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=False)


# test_x = torch.unsqueeze(test_dataset.data, dim=1).type(torch.Tensor)
# test_y = test_dataset.targets

class LeNet(nn.Module):  # Using this module need to delete resize in dataloader
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))  # 最后的结果一定要变为 10，因为数字的选项是 0 ~ 9

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


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

if not args.no_tensorboard:
    writer = SummaryWriter('./runs/DP')
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    writer.add_graph(model, images)

criterion = nn.CrossEntropyLoss()
if use_gpu:
    model.cuda()  # 判断是否调用GPU模式

if multi_gpu:
    print("Using {} GPUs".format(torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # 初始化优化器 model.train()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
train_acc_i = []
train_acc_list = []

a = []
ac_list = []


def train(epoch):  # 定义每个epoch的训练细节
    model.train()  # 设置为trainning模式
    running_loss = 0.0
    running_accuracy = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):

        if use_gpu:  # 如果要调用GPU模式，就把数据转存到GPU
            data, target = data.cuda(), target.cuda()
        if multi_gpu:
            data, target = data.to(device), target.to(device)
            # print("Outside: input size", data.size())

        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        train_output = torch.max(output, dim=1)[1]
        loss = criterion(output, target)  # 计算损失函数
        loss.backward()  # 反向传播梯度
        running_accuracy += torch.sum(torch.eq(target, train_output)).item() / target.cpu().numpy().size
        running_loss += loss.item()
        optimizer.step()  # 结束一次前传+反传之后，更新优化器参数
        if batch_idx % args.log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), running_accuracy / args.log_interval))
            # ...log the running loss
            if not args.no_tensorboard:
                writer.add_scalar('training loss',
                                  running_loss / args.log_interval,
                                  epoch * len(train_loader.dataset) + batch_idx * args.batch_size)
                writer.add_scalar('training accuracy', running_accuracy / args.log_interval,
                                  epoch * len(train_loader.dataset) + batch_idx * args.batch_size)
            running_loss = 0.0
            running_accuracy = 0.0


def test(epoch):
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    running_accuracy = 0.0
    total = 0
    count = 0
    for data, target in test_loader:
        if use_gpu:
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

    if not args.no_tensorboard:
        writer.add_scalar('testing loss', test_loss, epoch)
        writer.add_scalar('testing accuracy', 100. * correct / len(test_loader.dataset), epoch)


def profile(dir_name='./runs/benchmark/', batch_size=args.batch_size):
    for batch_idx, (train_x, train_y) in enumerate(train_loader):  # 把取数据放在profile里会报错，所以放在外面
        with profiler.profile(use_cuda=use_gpu) as prof:
            with profiler.record_function("model_training"):
                if use_gpu:
                    train_x, train_y = train_x.cuda(), train_y.cuda()
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(train_x), train_y)
                loss.backward()
                optimizer.step()
                break

    if use_gpu:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(
            dir_name + "profiler/DP_training_profiler_cuda_{}gpus_{}.json".format(torch.cuda.device_count(),
                                                                                          batch_size))
    else:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace(dir_name + "profiler/DP_training_profiler_cpu.json")

    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        with profiler.profile(use_cuda=use_gpu) as prof:
            with profiler.record_function("model_inference"):
                if use_gpu:
                    test_x, test_y = test_x.cuda(), test_y.cuda()
                model.eval()
                output = model(test_x)
                break

    if use_gpu:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(
            dir_name + "profiler/DP_inference_profiler_cuda_{}gpus_{}.json".format(torch.cuda.device_count(),
                                                                                          batch_size))
    else:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace(dir_name + "/profiler/DP_inference_profiler_cpu.json")


if __name__ == '__main__':

    for epoch in range(1, args.epochs + 1):  # 以epoch为单位进行循环
        start = time.time()
        train(epoch)
        end = time.time()
        print("Training using time: {}s, throughput: {} items/s".format(end - start,
                                                                        len(train_loader.dataset) / (end - start)))
        start = time.time()
        test(epoch)
        end = time.time()
        print("Inference using time: {}s, throughput: {} items/s".format(end - start,
                                                                         len(test_loader.dataset) / (end - start)))

    # Profiler
    profile('./runs/DP/')
