#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --no-requeue

source ~/dat01/lpeng/env.bash


srun -N 1 --ntasks-per-node=3 --gres=gpu:2 -p gpu python -u ./src/rpc_pipeline.py
