import torch
from torch import nn, Tensor
from torch.nn import functional


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, blocks: Tensor, targets: Tensor) -> (Tensor, Tensor):
        per_char_logits = self.token_embedding_table(blocks)
        loss = self._cross_entropy(per_char_logits, targets)
        return per_char_logits, loss  # todo - do we need to return the per-char logits?

    def generate(self, initial_chars: Tensor, tokens_to_generate: int):
        chars = initial_chars

        for _ in range(tokens_to_generate):
            next_char_logits = self.token_embedding_table(chars)[:, -1, :]
            # Convert the logits into a probability for each char.
            next_char_probs = functional.softmax(next_char_logits, dim=1)
            next_char = torch.multinomial(next_char_probs, num_samples=1)
            chars = torch.cat((chars, next_char), dim=1)

        return chars

    @staticmethod
    def _cross_entropy(per_char_logits: Tensor, targets: Tensor) -> Tensor:
        # Reshaping the logits and targets to the format cross_entropy requires
        batch_dim, block_dim, vocab_size_dim = per_char_logits.shape
        per_char_logits = per_char_logits.view(batch_dim * block_dim, vocab_size_dim)
        targets = targets.view(batch_dim * block_dim)

        return functional.cross_entropy(per_char_logits, targets)
