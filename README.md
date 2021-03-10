# DistributedTest
 Some benchmark of distributed training

## 1.Benchmark
单节点单卡测试

Note: 可profile
## 2.Data Parallel (DP)
单节点多卡数据并行，具体使用torch中的nn.DataParallel(model)实现

Note: 可profile
## 3.Distributed Data Parallel (DDP)
多节点多卡数据并行，使用from torch.nn.parallel import DistributedDataParallel实现

DDP和DP的区别在于，在使用DDP实例化模型之前，需要先设置好通信的机制，具体可以参考DDP.py中的 setup函数。

Note: 可profile
### 3.1 Slurm如何使用DDP
 使用Slurm进行实验，先写好一个sbatch用的脚本，注意脚本文件里面用的是srun命令。
 eg:
    
    #!/bin/bash

    #SBATCH -N 4
    #SBATCH --ntasks-per-node=1
    #SBATCH -p gpu
    #SBATCH --gres=gpu:1
    #SBATCH --no-requeue
    
    source ~/dat01/lpeng/env.bash # 加载环境变量
    
    
    #Used for ddp 
    srun -N 4 --ntasks-per-node=1 --gres=gpu:1 -p gpu python -u DDP.py

 
 在对应的python文件中，需要通过os库来获取对应的slurm环境变量，来设置setup函数中所需要的对应的rank。
 
 eg:
 
 
    rank = int(os.environ['SLURM_PROCID'])         # 全局的rank，用于init_process_group
    local_rank = int(os.environ['SLURM_LOCALID'])  # 一个节点上的rank，用于gpu的分配
    world_size = int(os.environ['SLURM_NTASKS'])   # 进程总数，用于init_process_group

Note: 可profile
    
## 4. Model Parallel (MP)
使用两块卡进行模型并行的测试，需要手动划分模型，即将子模型手动.to()到对应的GPU上。

eg:

    self.seq1 = nn.Sequential(
    self.conv,
    self.conv1,
    self.bn1,
    self.relu,
    self.maxpool,
    self.layer1,
    self.layer2
    ).to('cuda:0')

    self.seq2 = nn.Sequential(
    self.layer3,
    self.layer4,
    self.avgpool,
    ).to('cuda:1')
    
    x = self.seq2(self.seq1(x).to('cuda:1'))
-----------------------------
Note: 可profile

## 5. Pipelined Model Parallel (PMP)
在继承了MP模型的基础上，修改forward过程，将每次输入的batch分成更小的micro-batch，进行前向传播训练，
此处利用到了torch异步执行的机制。

注意：PMP的backward和step是没有做优化的。

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
        
Note: 可profile 


## 6. RPC_PMP
使用RPC库实现一个PMP，可能由于测试机群的通信机制的问题，效果非常的差。也有可能是当前torch的支持不好。

## 7. ZeRO_DDP
zero的思想可以参考 ref: [DeepSpeed](https://www.deepspeed.ai/)

torch1.8的ZeRO优化器属于ZeRO-1，只对优化器状态进行划分。

目前torch1.8的ZeRO优化器实现还是比较慢的，现在还是beta版本就先忍了。吞吐率奇低，大概率是因为每一次step的时候都需要进行很多次的通信，
也可能是因为我所用的机群通信太慢的原因。
    

Note: 应该可profile，没试过

## 8. torchgpipe (not  Done yet)
只支持单节点。

torchgpipe现在已经被加入了torch1.8豪华套餐，我还没有用1.8中的torchgpipe做过测试，
但是在torchgpipe的原版代码中我进行过一点测试。基本上micro-batch数量越多，加速比越大，
不过也存在一个加速的上限，曲线为凸型。
4卡，micro-batch为32个的情况下，加速比为2.8。

Note: 暂时不支持profile

## 9. DeepSpeed DP
deepspeed是zero的老家，降低内存界的扛把子。 ref: [DeepSpeed](https://www.deepspeed.ai/)

支持多机多卡，使用时只需要deepspeed.initialize一下就可以了，然后用model_engine进行模型的forward, backward和step即可。

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_data)
    
    ...
    
    outputs = model_engine(inputs)
    loss = criterion(outputs, labels)
    model_engine.backward(loss)
    model_engine.step()

在slurm机群中使用时需要先生成一个hostfile，命令如下：

    srun hostname -s > hostfile.txt
    
然后利用edit_hostfile改成DeepSpeed所需要的格式，最后用DeepSpeed启动任务。

具体可参考运行的脚本：[run_ds.sh](https://github.com/ConnollyLeon/DistributedTest/blob/master/ZeRO/DeepSpeedExamples/mnist/run_ds.sh)

Note: 暂时不支持profile

注意：这个实验里面暂时还没用到ZeRO，目前只是单纯的多节点数据并行，后面再进行修改。
实验的时候没有考虑到batch_size需要重新调整(DeepSpeed把batch_size平均分了)，
所以在实验结果里面看着会有内存大大减低了的错觉。

    

# Experiment
测试环境: torch1.6 (ZeRO-DDP on torch1.8) + cuda10.2； deepspeed=0.3.10

模型: ResNet50

数据集: MNIST

batch_size : 64
 

##  Experiment Results
 | 实验名字               | GPU type&num         | Throughput | Memory Usage  | Accuracy |
 | --------------------- | -------------------- | -----------| ------------- | --- |
 |  benchmark            |   V100               |    332/s   | 7.7GB         | 0.98
 |  DP                   | 2 V100               |    583/s   | 7.7GB each    | 0.97
 |  DP                   | 4 V100               |    818/s   | 7.7GB         | 0.86
 | Manual MP             | 2 V100               |    331/s   | 5.9GB / 3.4GB | 0.94
 | Pipelined MP          | 2 V100               |    398/s   | 5.6GB / 3.1GB | 0.98
 | RPC PMP (split_size=8)| 2 V100               |    146/s   | 8.1GB / 7.5GB |  - 
 | DDP                   | 2 V100  x 1 node     |    516/s   | 8.9GB / 7.6GB | 0.97
 | DDP                   | 1 V100  x 2 node     |    513/s   | 7.7GB / 7.7GB | 0.95
 | DDP                   | 1 V100  x 4 node     |    880/s   | 7.7GB each    | 0.96
 | DDP                   | 4 V100  x 1 node     |    131/s   | 7.7GB each    | 0.96
 | Zero-DDP              | 2 V100 x 1 node      |    43/s    | 8.0GB each    | 0.96
 | Zero-DDP              | 4 V100 x 1 node      |    38/s    | 8.0GB each    | 0.92
 | DeepSpeed-DP          | 2 V100 x 1 node      |    580/s   | 4.6GB each    | 没测
 | DeepSpeed-DP          | 1 V100 x 2 node      |    575/s   | 4.6GB each    | 没测
 | DeepSpeed-DP          | 4 V100 x 1 node      |    809/s   | 3.2GB each    | 没测
 | DeepSpeed-DP          | 1 V100 x 4 node      |    575/s   | 3.14GB each   | 没测
 | DeepSpeed-DP          | 2 V100 x 2 node      |    564/s   | 3.14GB each   | 没测
 | DeepSpeed-DP          | 4 V100 x 2 node      |    731/s   | 2.44GB each   | 没测
 | DeepSpeed-DP          | 2 V100 x 4 node      |    477/s   | 2.44GB each   | 没测
 
 

## Note
1. DDP 在单节点4块卡上可能存在通信的问题，在两个环境上测过均有这一现象
2. 可以看出DeepSpeed很依赖节点间的通信速率，因此节点数量多的时候，吞吐率反而下降了。可以尝试换个实验环境测一下
 
