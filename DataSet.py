import torch

BATCH_SIZE = 4
BLOCK_SIZE = 8
VALIDATION_SPLIT = 0.9


class DataSet:
    def __init__(self, data):
        validation_split_idx = int(VALIDATION_SPLIT * len(data))
        self.train_data = data[:validation_split_idx]
        self.validation_data = data[validation_split_idx:]

    def get_train_batch(self):
        return self._get_batch(self.train_data)

    def get_validation_batch(self):
        return self._get_batch(self.validation_data)

    @staticmethod
    def _get_batch(data):
        qty_block_start_idxs = (BATCH_SIZE,)
        max_block_start_idx = len(data) - BLOCK_SIZE
        block_start_idxs = torch.randint(max_block_start_idx, qty_block_start_idxs)

        context = torch.stack([data[i:i + BLOCK_SIZE] for i in block_start_idxs])
        labels = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in block_start_idxs])

        return context, labels
