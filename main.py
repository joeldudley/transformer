import torch

from DataSet import DataSet
from Vectoriser import Vectoriser

TORCH_SEED = 1337
DATA_PATH = 'input.txt'

torch.manual_seed(TORCH_SEED)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

vectoriser = Vectoriser(text)
data_set = DataSet(vectoriser.data)

xb, yb = data_set.get_train_batch()
print(xb.shape)
print(xb)
print(yb.shape)
print(yb)
