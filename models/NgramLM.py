import torch
from torch import nn
from torch.nn import functional as F


class NgramLM(nn.Module):
    def __init__(self, vocab_sz: int, emb_sz: int = 16, N: int = 4):
        super(NgramLM, self).__init__()
        self.N = N
        self.num_preceding = N - 1
        self.vocab_sz = vocab_sz
        self.emb_sz = emb_sz
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_sz, embedding_dim=self.emb_sz
        )  # Each token in the vocab maps to a probability distribution over the vocab for what could come next.
        self.linear = nn.Linear(self.num_preceding * self.emb_sz, self.vocab_sz)

    def pad(self, x):
        # Extend the context width by (self.num_preceding - 1) by adding a padding embedding of zeros to the beginning of each sequence.
        # We only need to pad by num_preceding - 1 because the first token in a training sequence counts as a preceding token
        pad_emb = torch.zeros(size=(x.shape[0], self.num_preceding - 1, self.emb_sz), device=x.device)
        return torch.cat([pad_emb, x], dim=1)

    def forward(self, x):
        # x.shape: (batch_sz, max_context_width)
        # Return a probability distribution over the vocab
        embs = self.embeddings(x)  # shape: (bs, mcw, es)
        max_context_width = embs.shape[1]

        embs = self.pad(embs)  # shape: (bs, mcw + (N - 2), es)
        ngram_embs = torch.cat(
            [embs[:, i : max_context_width + i, :] for i in range(self.num_preceding)],
            dim=-1,
        )  # shape: (batch_sz, max_context_width, (N - 1) * emb_sz)
        # the width should always be mcw, and we want to stack num_preceding number of embeddings together, so it's: [0: mcw], [1: mcw + 1], [2:mcw+2], ..., [num_preceding-1: mcw + (num_preceding-1)].

        return self.linear(ngram_embs)  # shape: (batch_sz, max_context_width, vocab_sz)

    def loss(self, pred, target):
        return F.cross_entropy(input=pred, target=target, reduction="mean")
