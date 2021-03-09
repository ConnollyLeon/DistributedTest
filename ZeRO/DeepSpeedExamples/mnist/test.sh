#!/bin/bash

#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

echo $HOSTNAME
echo $SLURM_JOB_ID
echo $SLURM_LOCALID
echo $SLURM_JOB_NODELIST
srun hostname -s > hostfile.txt
