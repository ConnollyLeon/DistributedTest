#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --no-requeue

source ~/dat01/lpeng/env.bash


#Used for testing ddp 
srun -N 1 --ntasks-per-node=4 --gres=gpu:4 -p gpu python -u  ./src/DDP.py
