#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --no-requeue

source ~/dat01/lpeng/env.bash
srun hostname -s > hostfile.txt
python edit_hostfile.py
deepspeed --hostfile hostfile.txt deepspeed_DP.py  --deepspeed --deepspeed_config ds_config.json
