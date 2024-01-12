import torch
from torch import Tensor

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
model = BigramLanguageModel(vectoriser.vocab_size)


def update_gradients(loss: Tensor):
    model.optimiser.zero_grad(set_to_none=True)
    loss.backward()
    model.optimiser.step()


def train(steps: int):
    for _ in range(steps):
        xb, yb = data_set.get_train_batch()
        loss = model(xb, yb)
        update_gradients(loss)
        print(loss.item())


train(10000)
print(vectoriser.decode(model.generate(torch.zeros((1, 1), dtype=torch.long), 300)[0].tolist()))
