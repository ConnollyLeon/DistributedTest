import os
import random
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50

random.seed(0)
torch.manual_seed(0)


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


def cleanup():
    dist.destroy_process_group()


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = resnet50(pretrained=False)

    def forward(self, x):
        # print("Inside: input size", x.size())
        x = self.conv(x)
        x = self.resnet(x)
        return x


def demo_basic(rank, world_size, local_rank):
    print(f"rank {rank}: Running basic DDP example on gpu {local_rank}.")

    # create model and move it to GPU with id rank
    print(f'rank {rank}: Moving models to gpu...')
    model = ResNet50().to('cuda:' + str(local_rank))
    ddp_model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 1, 224, 224))
    labels = torch.randn(20, 1000).to(local_rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    print(f'Rank {rank} Basic DDP example done~~')


def demo_checkpoint(rank, world_size, local_rank):
    print(f"rank {rank}: Running DDP checkpoint example.")

    model = ResNet50().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = "./model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 1, 224, 224))
    labels = torch.randn(20, 1000).to(local_rank)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    print(f"rank {rank}: DDP checkpoint example Done~~")


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    n_gpus = torch.cuda.device_count()
    print(f'rank {rank}:device_count:', n_gpus)
    print(f'rank {rank}:torch current device:', torch.cuda.current_device())
    setup(rank, world_size)
    demo_basic(rank, world_size, local_rank)
    demo_checkpoint(rank, world_size, local_rank)
    cleanup()
