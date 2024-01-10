from torch import nn, Tensor
from torch.nn import functional


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: Tensor, targets: Tensor) -> (Tensor, Tensor):
        logits = self.token_embedding_table(idx)
        batch_dim, block_dim, channel_dim = logits.shape

        logits_reshaped = logits.view(batch_dim * block_dim, channel_dim)
        targets_reshaped = targets.view(batch_dim * block_dim)

        loss = functional.cross_entropy(logits_reshaped, targets_reshaped)
        return logits_reshaped, loss
