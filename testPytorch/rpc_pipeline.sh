#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --no-requeue

source ~/dat01/lpeng/env.bash


#Used for testing ddp 
#srun -N 2 --ntasks-per-node=1 --gres=gpu:1 -p gpu python -u ddp_test.py
srun -N 1 --ntasks-per-node=3 --gres=gpu:2 -p gpu python -u rpc_pipeline.py
