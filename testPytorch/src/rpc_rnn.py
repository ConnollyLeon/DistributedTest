import os

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
import socket
import time

import rnn


def _run_trainer():
    r"""
    The trainer creates a distributed RNNModel and a DistributedOptimizer. Then,
    it performs training on using random input data.
    """
    batch = 5
    ntoken = 7
    ninp = 2

    nhid = 3
    nindices = 6
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # train for 10 iterations
    for epoch in range(10):
        # create distributed autograd context
        for data, target in get_next_batch():
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])
                # run distributed optimizer
                opt.step(context_id)
                # not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def setup(rank, world_size):
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
    print(f'rank {rank}: initializing process group')
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    print(f'rank {rank}: initialization done.')
    rpc.shutdown()


if __name__=="__main__":
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    setup(rank,world_size)
