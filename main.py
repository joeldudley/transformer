import torch

from BigramLanguageModel import BigramLanguageModel
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
m = BigramLanguageModel(vectoriser.vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
