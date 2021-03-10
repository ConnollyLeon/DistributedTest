#!/bin/bash

#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

source ~/dat01/lpeng/env.bash


#Used for testing ddp 
srun -N 2 --ntasks-per-node=1 --gres=gpu:1 -p gpu python -u ./src/ddp_test.py
