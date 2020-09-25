# !/usr/bin/env python

import os
import socket

import torch
import torch.distributed as dist


def run(rank, size):
    """Blocking point-to-point communication."""
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    elif rank == 1:
        # Receive tensor from process 0. Can only be used once?
        dist.recv(tensor=tensor, src=0)
    elif rank == 2:
        print("i am 2")

    else:
        print("i am 3")
    print('Rank ', rank, ' has data ', tensor[0])
    # hostname = 'g' + str(os.environ['SLURM_NODELIST']).strip('g[]')[0:4]
    rank = int(os.environ['SLURM_PROCID'])
    local_machine = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    print('rank', rank)
    print('local_machine', local_machine)
    print('world_size', world_size)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    hostname = os.environ['HOSTNAME']
    # 获取本机ip
    ip = socket.gethostbyname(hostname)
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = '54321'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    processes = []
    # for rank in range(size):
    #     p = Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    hostname = os.environ['HOSTNAME']
    # 获取本机ip
    ip = socket.gethostbyname(hostname)
    # hostname = 'g' + str(os.environ['SLURM_NODELIST']).strip('g[]')[0:4]
    rank = int(os.environ['SLURM_PROCID'])
    local_machine = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    print(hostname)
    print(ip)
    print('rank', rank)
    print('local_machine', local_machine)
    print('world_size', world_size)
    init_process(rank, world_size, run)
