import os
import socket
import threading
import time

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torchvision import datasets, transforms
from torchvision.models.resnet import Bottleneck

#########################################################
#           Define Model Parallel ResNet50              #
#########################################################

# In order to split the ResNet50 and place it on two different workers, we
# implement it in two model shards. The ResNetBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.


num_classes = 1000


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=1000,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()

        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class ResNetShard1(ResNetBase):
    """
    The first part of ResNet.
    """

    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2)
        ).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.seq(x)
        return out.cpu()


class ResNetShard2(ResNetBase):
    """
    The second part of ResNet.
    """

    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 512, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).to(self.device)

        self.fc = nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()


class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """

    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.split_size = split_size

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetShard1,
            args=("cuda:0",) + args,
            kwargs=kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetShard2,
            args=("cuda:0",) + args,
            kwargs=kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def run_master(split_size):
    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(split_size, ["worker1", "worker2"])
    loss_fn = nn.CrossEntropyLoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.Resize(224), torchvision.transforms.ToTensor()])),
        # transform=transforms.Compose([
        #     transforms.Resize(224),  # resnet默认图片输入大小224*224
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([transforms.Resize(224), torchvision.transforms.ToTensor()])),
        batch_size=64, shuffle=False)

    running_accuracy = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # generate random inputs and labels
        # if use_gpu:  # 如果要调用GPU模式，就把数据转存到GPU
        # data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)  # 把数据转换成Variable

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            outputs = model(data)
            train_outputs = torch.max(outputs, dim=1)[1]
            loss = loss_fn(outputs, target)
            dist_autograd.backward(context_id, [loss])
            opt.step(context_id)
        running_accuracy += torch.sum(torch.eq(target, train_outputs)).item() / target.cpu().numpy().size

        if batch_idx % 10 == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {:.6f}'.format(
                0, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item(), running_accuracy / 10))
            running_accuracy = 0.0


def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def setup(rank, world_size, num_split):
    hostname = os.environ['HOSTNAME']
    # 获取本机ip
    ip = socket.gethostbyname(hostname)
    jobid = os.environ['SLURM_JOB_ID']
    hostfile = "rpc_url." + jobid + ".txt"
    if rank == 0:
        port = find_free_port()
        rpc_url = "tcp://{}:{}".format(ip, port)
        with open(hostfile, "w") as f:
            f.write(rpc_url)
    else:
        while not os.path.exists(hostfile):
            time.sleep(1)
        with open(hostfile, 'r') as f:
            rpc_url = f.read()
        ip = rpc_url.strip('tcp://').split(':')[0]
        port = int(rpc_url.strip('tcp://').split(':')[1])

    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = str(port)

    print(f'rank {rank}:', "rpc-url:{} at PROCID {} / {}".format(rpc_url, rank, world_size))

    # initialize the process group
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)
    print(f'rank {rank}: initializing process group')
    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()
    print(f'rank {rank}: shutting down.')


if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    # world_size=3
    for num_split in [8]:
        tik = time.time()
        setup(rank, world_size, num_split)
        # mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
        tok = time.time()
        print(f"number of splits = {num_split}, execution time = {tok - tik}s")
