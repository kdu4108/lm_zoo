from typing import List, Union, Tuple, Set
import torch


class CharacterTokenizer:
    def __init__(self, vocab: Set[str]) -> None:
        self.vocab = vocab
        self.vocab_sz = len(vocab)
        self.token_to_str = {i: c for i, c in enumerate(sorted(vocab))}
        self.str_to_token = {c: i for i, c in enumerate(sorted(vocab))}

    def tokenize(self, text: str) -> torch.LongTensor:
        # Shape should be (len(text),)
        return torch.tensor([self.str_to_token[c] for c in text], dtype=torch.long)

    def batch_decode(self, sequences: Union[List[int], torch.LongTensor]):
        return "".join([self.token_to_str[i] for i in sequences])
