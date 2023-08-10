import torch
from torch import nn
from torch.nn import functional as F


class WeightedAveragePrevEmbeddingsLM(nn.Module):
    """
    This model starts to implement self-attention, but with no projection to keys/value/queries.
    It does manage to learn different weights for each word in the context, based on the embedding.
    """

    def __init__(self, vocab_sz: int, emb_sz: int = 16):
        super(WeightedAveragePrevEmbeddingsLM, self).__init__()
        self.vocab_sz = vocab_sz
        self.emb_sz = emb_sz
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_sz, embedding_dim=self.emb_sz
        )  # Each token in the vocab maps to a probability distribution over the vocab for what could come next.
        self.linear = nn.Linear(self.emb_sz, self.vocab_sz)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Generate a probability distribution over vocab for each token in the input x.

        Given a batch of input tokens of shape (bs, mcw),
        produce a (bs, mcw, vocab_sz) float tensor whose (i, j)-th slice represents the
        probability distribution over the vocab for the (j+1)-th token of the ith sentence in the batch.

        Args:
            x - batch of input tokens of shape (bs, mcw)

        Returns:
            (bs, mcw, vocab_sz) float tensor whose (i, j)-th slice represents the
            probability distribution over the vocab for the (j+1)-th token of the
            ith sentence in the batch.
        """
        # x.shape: (batch_sz, max_context_width)
        # Return a probability distribution over the vocab
        embs = self.embeddings(x)  # shape: (bs, mcw, emb_sz)
        batch_sz, max_context_width, emb_sz = embs.shape

        # We want a weighted average of the embeddings preceding the current token
        # Start by just getting the average of the previous tokens in the context
        mask = torch.tril(torch.ones(size=(max_context_width, max_context_width))).to(
            x.device
        )  # (mcw, mcw) tensor of 1s only below the diagonal (inclusive)

        # Compute weights based on how similar the embeddings are to the last word.
        # Assume the query of a word and the key of a word are both just represented by the embs of that word
        # In first tensor, we consider the words the queries. In the second tensor, we consider the words the keys.
        # This essentially takes the dot product of each query with each key and we result in a
        #  (batch_sz, num_queries=mcw, num_keys=mcw) tensor.
        weights = torch.einsum("bqe,bke->bqk", embs, embs)

        # Zero out scores/weights to the right of the query
        weights = torch.einsum("bqk,qk->bqk", weights, mask)
        weights = torch.where(weights == 0.0, -torch.inf, weights)
        weights = F.softmax(weights, dim=1)  # (mcw, mcw), aka the (words, prob dist p across words)

        # Weighted average of the values for each word (represented as the word's embedding) according to weights
        # Take as input:
        #   (a) a tensor functioning as a "lookup table" of the embedding
        #       for each word in the sentence
        #       for each sentence in the batch
        #       with shape (batch_sz, number of words in the sentence = mcw, emb_sz)
        #       A[b, p] = the embedding for word p in sentence b
        #   and
        #   (b) a tensor functioning as the weights of the embeddings in (a),
        #       for every word you care about,
        #       with shape (words you care about = mcw, number of words in the sentence = mcw)
        #       B[w, p] = the weight that word p matters to word w
        # And output:
        #    a tensor with the weighted average embedding for every word you care about in every sentence in the batch
        #    with shape (batch_sz, num words you care about, emb_sz)
        # Why in this case does the num words you care about (w) = num words being weighted (p)?
        # It's because we want an average embedding for every token in the sentence, of which there are num words!
        # If we only cared about getting embeddings for the first 5 words for each sentence in the batch,
        # then this gives us output of size (batch_sz, 5, emb_sz).
        # The embedding for the wth word is the weighted average of the embedding of the words from [0...w]
        out = torch.einsum("bpe,bwp->bwe", embs, weights)  # shape: (bs, mcw, emb_sz)

        logits = self.linear(out)

        return logits

    def loss(self, pred, target):
        return F.cross_entropy(input=pred, target=target, reduction="mean")
