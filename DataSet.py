import torch

DATA_PATH = 'input.txt'
BATCH_SIZE = 4
BLOCK_SIZE = 8


class DataSet:
    def __init__(self):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            text = f.read()

        self.chars = sorted(list(set(text)))
        self.char_to_int = {char: int_ for int_, char in enumerate(self.chars)}
        self.int_to_char = {int_: char for int_, char in enumerate(self.chars)}

        data = torch.tensor(self.encode(text), dtype=torch.long)
        _90th = int(0.9 * len(data))

        self.train_data = data[:_90th]
        self.validation_data = data[_90th:]

    def encode(self, str_):
        return [self.char_to_int[char] for char in str_]

    def decode(self, ints):
        return ''.join([self.int_to_char[int_] for int_ in ints])

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
