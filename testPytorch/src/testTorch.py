import torch

print("Checking torch version: ", torch.__version__)
print("Checking corresponding CUDA version: ", torch.version.cuda)
print("Checking CUDA available: ", torch.cuda.is_available())
print("Checking distributed package available: ", torch.distributed.is_available())
print("Checking MPI available: ", torch.distributed.is_mpi_available())
print("Checking gloo available: ", torch.distributed.is_gloo_available())
print("Checking nccl available: ", torch.distributed.is_nccl_available())


