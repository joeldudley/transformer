import torch
from torch import Tensor

from DataSet import DataSet
from Learner import Learner
from Vectoriser import Vectoriser


class Model:
    def __init__(self, text: str):
        self.vectoriser = Vectoriser(text)
        self.data_set = DataSet(self.vectoriser.data)
        self.learner = Learner(self.vectoriser.vocab_size)

    def train(self, steps: int):
        for step in range(steps):
            xb, yb = self.data_set.get_train_batch()
            loss = self.learner(xb, yb)
            self._update_gradients(loss)
            rounded_loss = "%.2f" % loss.item()
            print(f"Step {step+1}: {rounded_loss}")

    def generate(self, token_qty: int):
        tokens = self.learner.generate(torch.zeros((1, 1), dtype=torch.long), token_qty)[0].tolist()
        return self.vectoriser.decode(tokens)

    def _update_gradients(self, loss: Tensor):
        self.learner.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.learner.optimiser.step()
