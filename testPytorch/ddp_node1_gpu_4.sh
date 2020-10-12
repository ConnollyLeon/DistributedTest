#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --no-requeue

source ~/dat01/lpeng/env.bash


#Used for testing ddp 
#srun -N 2 --ntasks-per-node=1 --gres=gpu:1 -p gpu python -u ddp_test.py
#srun -N 2 --ntasks-per-node=1 --gres=gpu:1 -p gpu python -u DDP.py
srun -N 1 --ntasks-per-node=4 --gres=gpu:4 -p gpu python -u  DDP.py
