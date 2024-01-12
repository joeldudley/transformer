import torch
from torch import Tensor

from DataSet import DataSet
from Learner import Learner
from Vectoriser import Vectoriser

EVAL_FREQ = 1000


class Model:
    def __init__(self, text: str):
        self.vectoriser = Vectoriser(text)
        self.data_set = DataSet(self.vectoriser.data)
        self.learner = Learner(self.vectoriser.vocab_size)

    def train(self, steps: int):
        for step in range(steps):
            loss = self._loss_from_rand_batch_train()
            self._update_gradients(loss)

            if (step + 1) % EVAL_FREQ == 0:
                self._estimate_loss(step)

    def generate(self, token_qty: int):
        context = torch.zeros((1, 1), dtype=torch.long)
        tokens = self.learner.generate(context, token_qty)[0].tolist()
        return self.vectoriser.decode(tokens)

    def _update_gradients(self, loss: Tensor):
        self.learner.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.learner.optimiser.step()

    def _loss_from_rand_batch_train(self) -> Tensor:
        xb, yb = self.data_set.get_train_batch()
        return self.learner(xb, yb)

    def _loss_from_rand_batch_validation(self) -> Tensor:
        xb, yb = self.data_set.get_validation_batch()
        return self.learner(xb, yb)

    def _estimate_loss(self, step: int):
        self.learner.eval()

        train_loss = sum(self._loss_from_rand_batch_train().item() / EVAL_FREQ for _ in range(EVAL_FREQ))
        validation_loss = sum(self._loss_from_rand_batch_validation().item() / EVAL_FREQ for _ in range(EVAL_FREQ))
        print(f"Step {step + 1}: {'%.2f' % train_loss}, {'%.2f' % validation_loss}")

        self.learner.train()
