import torch
from torch import nn
from torch.nn import functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_sz: int):
        super(BigramLM, self).__init__()
        self.vocab_sz = vocab_sz
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_sz, embedding_dim=self.vocab_sz
        )  # Each token in the vocab maps to a probability distribution over the vocab for what could come next.

    def forward(self, x):
        # x.shape: (batch_sz, max_context_width)
        # Return a probability distribution over the vocab
        return self.embeddings(x)  # shape: (batch_sz, max_context_width, vocab_sz)

    def loss(self, pred, target):
        return torch.mean(F.cross_entropy(input=pred, target=target))
