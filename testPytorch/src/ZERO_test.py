import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

import inspect
from ..utils.gpu_mem_track import MemTracker


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


def track_gpu(tracker, rank):
    if rank == 0:
        tracker.track()


def example(rank, world_size, use_zero):
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    from pycallgraph import Config
    from pycallgraph import GlobbingFilter

    config = Config()
    config.max_depth = 10
    # config.include_pycallgraph=True
    config.include_stdlib = True
    config.trace_filter = GlobbingFilter(include=[
        'main',
        '*',
        'numpy.*',
        'pandas.*',
        'np.*',
        'pd.*',
        'pandas.DataFrame.*',
        'pd.DataFrame.*',
    ],
    exclude=['pycallgraph.*'])
    with PyCallGraph(output=GraphvizOutput(), config=config):
        if rank == 0:
            device = torch.device(rank)
            frame = inspect.currentframe()
            gpu_tracker = MemTracker(frame)

        track_gpu(gpu_tracker, rank)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        # create default process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # create local model
        track_gpu(gpu_tracker, rank)
        model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        track_gpu(gpu_tracker, rank)
        torch.cuda.synchronize()
        print_peak_memory("Max memory allocated after creating local model", rank)

        # construct DDP model
        ddp_model = DDP(model, device_ids=[rank])
        track_gpu(gpu_tracker, rank)
        torch.cuda.synchronize()
        print_peak_memory("Max memory allocated after creating DDP", rank)

        # define loss function and optimizer
        loss_fn = nn.MSELoss()
        if use_zero:
            optimizer = ZeroRedundancyOptimizer(
                ddp_model.parameters(),
                optim=torch.optim.Adam,
                lr=0.01
            )
        else:
            optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)
        track_gpu(gpu_tracker, rank)
        # forward pass
        outputs = ddp_model(torch.randn(20, 2000).to(rank))
        labels = torch.randn(20, 2000).to(rank)
        # backward pass
        track_gpu(gpu_tracker, rank)
        print_peak_memory("Max memory allocated after forward()", rank)
        loss_fn(outputs, labels).backward()

        # update parameters
        track_gpu(gpu_tracker, rank)
        torch.cuda.synchronize()
        print_peak_memory("Max memory allocated after backward()", rank)

        track_gpu(gpu_tracker, rank)
        optimizer.step()
        torch.cuda.synchronize()
        print_peak_memory("Max memory allocated after optimizer step()", rank)

        print(f"params sum is: {sum(model.parameters()).sum()}")


def main():
    world_size = 1
    print("=== Using ZeroRedundancyOptimizer ===")
    mp.spawn(example,
             args=(world_size, True),
             nprocs=world_size,
             join=True)

    print("=== Not Using ZeroRedundancyOptimizer ===")
    mp.spawn(example,
             args=(world_size, False),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
