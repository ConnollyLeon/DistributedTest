''' Pipelined Model Parallelism
'''

from __future__ import print_function  # 这个是python当中让print都以python3的形式进行print，即把print视为函数

import argparse
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, Bottleneck

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


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):  # Bottleneck and [3, 4, 6, 3] refers to ResNet50
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=1000, *args, **kwargs)
        self.conv = nn.Conv2d(1, 3, kernel_size=1)  # add for MNIST
        self.seq1 = nn.Sequential(
            self.conv,
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))


class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=32, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


model = PipelineParallelResNet50()

if not args.no_tensorboard:
    writer = SummaryWriter('./runs/PMP')
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    images = images.cuda()
    writer.add_graph(model, images)

criterion = nn.CrossEntropyLoss()

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

        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        # For MP use 从cuda:0输入数据
        output = model(data.to('cuda:0'))  # 把数据输入网络并得到输出，即进行前向传播
        train_output = torch.max(output, dim=1)[1]
        # 从output的device输入target进行反向传播
        target = target.to(output.device)
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
        output = model(data.to('cuda:0'))
        target = target.to(output.device)
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
        with profiler.profile() as prof:
            with profiler.record_function("model_training"):
                train_x = train_x.to('cuda:0')
                train_y = train_y.to('cuda:1')
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(train_x), train_y)
                loss.backward()
                optimizer.step()


    if use_gpu:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(
            dir_name + "profiler/PMP_training_profiler_cuda_{}gpus_{}.json".format(torch.cuda.device_count(),
                                                                                   batch_size))
    else:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace(dir_name + "profiler/PMP_training_profiler_cpu.json")

    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        with profiler.profile(use_cuda=use_gpu) as prof:
            with profiler.record_function("model_inference"):
                model.eval()
                output = model(test_x.to('cuda:0'))
                break

    if use_gpu:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(
            dir_name + "profiler/PMP_inference_profiler_cuda_{}gpus_{}.json".format(torch.cuda.device_count(),
                                                                                    batch_size))
    else:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace(dir_name + "/profiler/PMP_inference_profiler_cpu.json")


def train_time(model):
    num_batches = 3
    batch_size = 120
    image_w = 128
    image_h = 128
    num_classes = 1000
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 1, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
            .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        try:
            outputs = model(inputs.to('cuda:0'))
        except:
            inputs = torch.rand(batch_size, 3, image_w, image_h)
            outputs = model(inputs.to('cuda:0'))
        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


def compare():
    '''
    用来画 Single GPU, MP 和  PMP的对比图的
    :return:
    '''
    num_repeat = 10
    num_classes = 1000
    stmt = "train_time(model)"

    setup = "model = ModelParallelResNet50()"
    # globals arg is only available in Python 3. In Python 2, use the following
    # import __builtin__
    # __builtin__.__dict__.update(locals())
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    setup = "import torchvision.models as models;" + \
            "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

    setup = "model = PipelineParallelResNet50()"
    num_repeat = 10
    pp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

    def plot(means, stds, labels, fig_name):
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(means)), means, yerr=stds,
               align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
        ax.set_ylabel('ResNet50 Execution Time (Second)')
        ax.set_xticks(np.arange(len(means)))
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close(fig)

    plot([mp_mean, rn_mean],
         [mp_std, rn_std],
         ['Model Parallel', 'Single GPU'],
         'mp_vs_rn.png')

    plot([mp_mean, rn_mean, pp_mean],
         [mp_std, rn_std, pp_std],
         ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
         'mp_vs_rn_vs_pp.png')

    means = []
    stds = []
    split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60, 120]

    for split_size in split_sizes:
        setup = "model = PipelineParallelResNet50(split_size=%d)" % split_size
        pp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        means.append(np.mean(pp_run_times))
        stds.append(np.std(pp_run_times))

    fig, ax = plt.subplots()
    ax.plot(split_sizes, means)
    ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xlabel('Pipeline Split Size')
    ax.set_xticks(split_sizes)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig("split_size_tradeoff.png")
    plt.close(fig)


if __name__ == '__main__':

    # Used for comparision among Single GPU, Model Parallelism, Pipelined Model Parallelism
    # compare()

    # for epoch in range(1, args.epochs + 1):  # 以epoch为单位进行循环
    #     start = time.time()
    #     train(epoch)
    #     end = time.time()
    #     print("Training using time: {}s, throughput: {} items/s".format(end - start,
    #                                                                     len(train_loader.dataset) / (end - start)))
    #     start = time.time()
    #     test(epoch)
    #     end = time.time()
    #     print("Inference using time: {}s, throughput: {} items/s".format(end - start,
    #                                                                      len(test_loader.dataset) / (end - start)))

    # Profiler
    profile('./runs/PMP/')
