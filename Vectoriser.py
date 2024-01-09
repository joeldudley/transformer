import torch


class Vectoriser:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self._char_to_int = {char: int_ for int_, char in enumerate(chars)}
        self._int_to_char = {int_: char for int_, char in enumerate(chars)}

        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def encode(self, str_):
        return [self._char_to_int[char] for char in str_]

    def decode(self, ints):
        return ''.join([self._int_to_char[int_] for int_ in ints])
