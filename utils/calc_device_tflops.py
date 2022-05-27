"""This is used to calculate the performance of device.
Tested device:
    V100 16GB : 6.16 TFlops
       
"""

import torch
import time

inputs = torch.rand((1024, 1024, 1024)).to(0)
weights = torch.rand((1024, 1024)).to(0)

calc_flops = 1024 ** 4

# initialize torch cuda context and so on
output = torch.matmul(inputs, weights)

torch.cuda.synchronize()
start = time.time()

for i in range(100):
    output = torch.matmul(inputs, weights)

torch.cuda.synchronize()
end = time.time()

total_time = end - start

print(f"current device calculation performace:{calc_flops * 100 / total_time / 1024 / 1024 / 1024 / 1024} TFLOPs")
