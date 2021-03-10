import argparse
import os
import random
import socket
import time

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.distributed.optim import ZeroRedundancyOptimizer

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


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def setup(rank, world_size):
    hostname = os.environ['HOSTNAME']
    # 获取本机ip
    ip = socket.gethostbyname(hostname)
    jobid = os.environ['SLURM_JOB_ID']
    hostfile = "dist_url." + jobid + ".txt"
    if rank == 0:
        port = find_free_port()
        dist_url = "tcp://{}:{}".format(ip, port)
        with open(hostfile, "w") as f:
            f.write(dist_url)
    else:
        while not os.path.exists(hostfile):
            time.sleep(1)
        with open(hostfile, 'r') as f:
            dist_url = f.read()
        ip = dist_url.strip('tcp://').split(':')[0]
        port = int(dist_url.strip('tcp://').split(':')[1])

    print(f'rank {rank}:', "dist-url:{} at PROCID {} / {}".format(dist_url, rank, world_size))

    # initialize the process group
    print(f'rank {rank}: initializing process group')
    tcp_store = dist.TCPStore(ip, port, world_size, rank == 0)
    dist.init_process_group("nccl", store=tcp_store, rank=rank, world_size=world_size)
    print(f'rank {rank}: initialization done.')
    assert torch.distributed.is_initialized()


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


# 此后训练流程与普通模型无异
def train(epoch, local_rank):  # 定义每个epoch的训练细节
    model.train()  # 设置为trainning模式
    running_loss = 0.0
    running_accuracy = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:  # 如果要调用GPU模式，就把数据转存到GPU
            data, target = data.cuda(local_rank), target.cuda(local_rank)
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        train_output = torch.max(output, dim=1)[1]
        loss = criterion(output, target)  # 计算损失函数
        print_peak_memory('Max memory allocated before backward', local_rank)
        loss.backward()  # 反向传播梯度
        print_peak_memory('Max memory allocated after backward', local_rank)
        running_accuracy += torch.sum(torch.eq(target, train_output)).item() / target.cpu().numpy().size
        running_loss += loss.item()
        optimizer.step()  # 结束一次前传+反传之后，更新优化器参数
        print_peak_memory('Max memory allocated after optimizer step', local_rank)
        if batch_idx % args.log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            print(f'rank {rank}:' + 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {:.6f}'.format(
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


def test(epoch, local_rank):
    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    total = 0
    count = 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(local_rank), target.cuda(local_rank)
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item()  # sum up batch loss 把所有loss值进行累加
        test_output = torch.max(output, dim=1)[1]  # get the index of the max log-probability
        correct += torch.sum(torch.eq(test_output, target)).item()
        total += target.size(0)
        count += 1

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
                    train_x, train_y = train_x.cuda(local_rank), train_y.cuda(local_rank)
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(train_x), train_y)
                loss.backward()
                optimizer.step()
                break
    if rank == 0:
        if use_gpu:
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
            prof.export_chrome_trace(
                dir_name + "profiler/DDP_training_profiler_cuda_{}_gpus_{}_nodes_{}.json".format(
                    torch.cuda.device_count(),
                    batch_size,
                    world_size / n_gpus))
        else:
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
            prof.export_chrome_trace(dir_name + "profiler/DDP_training_profiler_cpu.json")

    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        with profiler.profile(use_cuda=use_gpu) as prof:
            with profiler.record_function("model_inference"):
                if use_gpu:
                    test_x, test_y = test_x.cuda(local_rank), test_y.cuda(local_rank)
                model.eval()
                output = model(test_x)
                break
    if rank == 0:
        if use_gpu:
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
            prof.export_chrome_trace(
                dir_name + "profiler/DDP_inference_profiler_cuda_{}_gpus_{}_nodes_{}.json".format(
                    torch.cuda.device_count(),
                    batch_size,
                    world_size / n_gpus))
        else:
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
            prof.export_chrome_trace(dir_name + "/profiler/DDP_inference_profiler_cpu.json")


if __name__ == '__main__':
    # set process group up...
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])

    # Set seed
    random.seed(rank + 1)
    torch.manual_seed(rank + 1)
    np.random.seed(rank + 1)
    if torch.cuda.is_available() and not args.no_cuda:
        use_gpu = True
        print("Using GPU...")
        torch.cuda.manual_seed(rank + 1)  # Set a seed to make result consistent
    else:
        print("Using CPU...")
        use_gpu = False

    # Very important when using multiple CUDA devices on single node?
    print(f"rank {rank}:setting visible devices")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    print(f"rank {rank}:setting visible devices Done, Using cuda:", os.environ["CUDA_VISIBLE_DEVICES"])

    n_gpus = torch.cuda.device_count()
    # This step need to be done before process_group_init
    torch.cuda.set_device(local_rank)
    print(f'rank {rank}:device_count:', n_gpus)
    print(f'rank {rank}:torch current device:', torch.cuda.current_device())
    setup(rank, world_size)

    # Prepare DataLoader
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        pin_memory=True)

    # Prepare Model
    model = ResNet50()  # 也是按自己的模型写
    model.cuda(local_rank)
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    print_peak_memory('Max memory allocated after creating local model', local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    print_peak_memory('Max memory allocated after creating DDP', local_rank)

    optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                        optim=torch.optim.Adam,
                                        lr=args.lr)

    # Draw a picture
    if not args.no_tensorboard:
        writer = SummaryWriter('./runs/ZeRO/DDP')
        # data_iter = iter(train_loader)
        # images, labels = data_iter.next()
        # writer.add_graph(model, images)

    # Training and testing
    for epoch in range(1, args.epochs + 1):  # 以epoch为单位进行循环
        start = time.time()
        train(epoch, local_rank)
        end = time.time()
        print(f'rank {rank}:' + "Training using time: {}s, throughput: {} items/s".format(end - start,
                                                                                          len(train_loader.dataset) / (
                                                                                                  end - start)))
        start = time.time()
        test(epoch, local_rank)
        end = time.time()
        if rank == 0:
            print("Inference using time: {}s, throughput: {} items/s".format(end - start,
                                                                             len(test_loader.dataset) / (end - start)))

    # Profiler

    profile('./runs/DDP/')
