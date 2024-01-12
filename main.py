import torch

from Model import Model

TORCH_SEED = 1337
DATA_PATH = 'input.txt'

torch.manual_seed(TORCH_SEED)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

model = Model(text)
model.train(10000)
print(model.generate(300))
