import torch

from DataSet import DataSet

TORCH_SEED = 1337

torch.manual_seed(TORCH_SEED)
data_set = DataSet()

xb, yb = data_set.get_train_batch()
print(xb.shape)
print(xb)
print(yb.shape)
print(yb)
