测试环境: torch1.6 
batch_size 均为64

#  Experiment Results
 | 实验名字              | GPU type&num         | Throughput | Memory Usage |
 | -------------------- | -------------------- | ----- | ------------ |
 |  benchmark           |   V100               | 332/s | 7.7GB        |
 |  DP                  | 2 V100               | 583/s | 7.7GB each   |
 |  DP                  | 4 V100               | 818/s | 7.7GB        |
 | Manual MP            | 2 V100               | 331/s | 5.9GB / 3.4GB|
 | Pipelined MP         | 2 V100               | 398/s | 5.6GB / 3.1GB|
 | DDP                   | 2 V100 x2 x 1 node  | 516/s | 8.9GB / 7.6GB|
 | DDP                   | 1 V100 x1 x 2 node  | 513/s | 7.7GB / 7.7GB|
 | DDP                   | 1 V100 x1 x 4 node  | 880/s | 7.7GB each   |
 | DDP                   | 4 V100 x4 x 1 node  | 131/s | 7.7GB each   |
 
# Note
DDP 在单节点4块卡上可能存在通信的问题，在两个环境上测过均有这一现象
 
