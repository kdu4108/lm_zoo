import torch
from torch import nn
from torch.nn import functional as F


class TrigramLM(nn.Module):
    def __init__(self, vocab_sz: int, emb_sz: int = 16):
        super(TrigramLM, self).__init__()
        self.vocab_sz = vocab_sz
        self.emb_sz = emb_sz
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_sz, embedding_dim=self.emb_sz
        )  # Each token in the vocab maps to a probability distribution over the vocab for what could come next.
        self.linear = nn.Linear(2 * self.emb_sz, self.vocab_sz)

    def pad(self, x):
        # Extend the context width by 1 by adding a padding embedding of zeros to the beginning of each sequence
        pad_emb = torch.zeros(size=(x.shape[0], 1, self.emb_sz))
        return torch.cat([pad_emb, x], dim=1)

    def forward(self, x):
        # x.shape: (batch_sz, max_context_width)
        # Return a probability distribution over the vocab
        embs = self.embeddings(x)  # shape: (batch_sz, max_context_width, emb_sz)
        embs = self.pad(embs)
        trigram_embs = torch.cat(
            [embs[:, :-1, :], embs[:, 1:, :]], dim=-1
        )  # shape: (batch_sz, max_context_width, 2 * emb_sz)
        return self.linear(trigram_embs)  # shape: (batch_sz, max_context_width - 1, vocab_sz)

    def loss(self, pred, target):
        return F.cross_entropy(input=pred, target=target, reduction="mean")
