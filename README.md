# DistributedTest
 Some benchmark of distributed training

# Distributed Data Parallel
 使用Slurm进行实验，先写好一个sbatch用的脚本，注意脚本文件里面用的是srun命令。在对应的python文件中，需要通过os库来获取对应的slurm环境变量，来设置对应的rank。
 eg:
    rank = int(os.environ['SLURM_PROCID'])         # 全局的rank，用于init_process_group
    local_rank = int(os.environ['SLURM_LOCALID'])  # 一个节点上的rank，用于gpu的分配
    world_size = int(os.environ['SLURM_NTASKS'])   # 进程总数，用于init_process_group