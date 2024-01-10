from typing import List

import torch


class Vectoriser:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self._char_to_int = {char: int_ for int_, char in enumerate(chars)}
        self._int_to_char = {int_: char for int_, char in enumerate(chars)}

        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def encode(self, to_encode: str) -> List[int]:
        return [self._char_to_int[char] for char in to_encode]

    def decode(self, ints: List[int]) -> str:
        return ''.join([self._int_to_char[int_] for int_ in ints])
