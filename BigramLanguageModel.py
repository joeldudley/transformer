import torch
from torch import nn, Tensor
from torch.nn import functional, Embedding


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = self._get_random_embeddings(vocab_size)

    def forward(self, blocks: Tensor, targets: Tensor) -> Tensor:
        per_token_logits = self.token_embedding_table(blocks)
        return self._cross_entropy(per_token_logits, targets)

    def generate(self, initial_tokens: Tensor, tokens_to_generate: int) -> Tensor:
        output_tokens = initial_tokens

        for _ in range(tokens_to_generate):
            next_token = self.generate_next_token(output_tokens)
            output_tokens = torch.cat((output_tokens, next_token), dim=1)

        return output_tokens

    def generate_next_token(self, output_chars) -> Tensor:
        next_token_logits = self.token_embedding_table(output_chars)[:, -1, :]
        next_token_probs = functional.softmax(next_token_logits, dim=1)
        return torch.multinomial(next_token_probs, num_samples=1)

    @staticmethod
    def _get_random_embeddings(vocab_size) -> Embedding:
        return nn.Embedding(vocab_size, vocab_size)

    @staticmethod
    def _cross_entropy(per_char_logits: Tensor, targets: Tensor) -> Tensor:
        # Reshaping the logits and targets to the dimensions required by cross_entropy.
        batch_dim, block_dim, vocab_size_dim = per_char_logits.shape
        per_char_logits = per_char_logits.view(batch_dim * block_dim, vocab_size_dim)
        targets = targets.view(batch_dim * block_dim)

        return functional.cross_entropy(per_char_logits, targets)
