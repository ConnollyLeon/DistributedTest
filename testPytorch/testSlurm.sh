#!/bin/bash

#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH -p gpu
#SBATCH --gres=gpu:0
#SBATCH --no-requeue

source ~/dat01/lpeng/env.bash

#python ddp_demo.py
#python testSlurm.py > rank_${SLURM_LOCALID}.log
#python ~/dat01/lpeng/SystemTestCode/testPytorch/DDP.py
srun -N 2 --ntasks-per-node=2 --gres=gpu:0 -p gpu python  testSlurm.py

