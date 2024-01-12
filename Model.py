import torch
from torch import Tensor

from DataSet import DataSet
from Learner import Learner
from Vectoriser import Vectoriser

EVAL_FREQ = 200


class Model:
    def __init__(self, text: str):
        self.vectoriser = Vectoriser(text)
        self.data_set = DataSet(self.vectoriser.data)
        self.learner = Learner(self.vectoriser.vocab_size)

    def train(self, steps: int):
        total_loss = 0

        for step in range(steps):
            xb, yb = self.data_set.get_train_batch()
            loss = self.learner(xb, yb)
            self._update_gradients(loss)
            total_loss += loss.item()

            if (step + 1) % EVAL_FREQ == 0:
                rounded_loss = "%.2f" % (total_loss / EVAL_FREQ)
                total_loss = 0
                print(f"Step {step + 1}: {rounded_loss}")

    def generate(self, token_qty: int):
        context = torch.zeros((1, 1), dtype=torch.long)
        tokens = self.learner.generate(context, token_qty)[0].tolist()
        return self.vectoriser.decode(tokens)

    def _update_gradients(self, loss: Tensor):
        self.learner.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.learner.optimiser.step()
