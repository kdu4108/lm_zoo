import torch
from torch import nn
from torch.nn import functional as F


class SelfAttentionLM(nn.Module):
    """
    This model starts to implement self-attention, including projection to keys/value/queries.
    It manages to learn different weights for each word in the context, based on the specialized representations of
    query, key, and value based on each word's embedding.
    However, this does not normalize the weights after dot-product of query and keys, which may increase the variance of the
    distribution, making the softmax pointier and therefore resulting in too much attention on one/a few words.
    """

    def __init__(self, vocab_sz: int, emb_sz: int = 16):
        super(SelfAttentionLM, self).__init__()
        self.vocab_sz = vocab_sz
        self.emb_sz = emb_sz
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_sz, embedding_dim=self.emb_sz
        )  # Each token in the vocab maps to a probability distribution over the vocab for what could come next.

        self.query_proj = nn.Linear(self.emb_sz, self.emb_sz)
        self.key_proj = nn.Linear(self.emb_sz, self.emb_sz)
        self.value_proj = nn.Linear(self.emb_sz, self.emb_sz)

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

        # Compute weights based on how well the current token's query representation matches the other tokens' key representations.
        # Here, we project the embedding of each token into a specialized "query" representation (also of size emb_sz because why not)
        # and also into a specialized "key" representation.
        # For decoding, we only compare the query of the most recent token i to the
        # key of each of the words in the context up to that token [0,...,i] (inclusive).
        # But since we ultimate want to predict the next token for every token in a sentence,
        # we need a query for all tokens and therefore we do some matmuls to compute every query for every token
        # and every key for every token efficiently.
        #
        # In first tensor, we consider the words the queries.
        # In the second tensor, we consider the words the keys.
        # This essentially takes the dot product of each word's query with each word's key, resulting in a
        #  (batch_sz, num_queries=mcw, num_keys=mcw) tensor.

        # Project from the embedding of each token to the query and key representations of each token
        queries = self.query_proj(embs)  # shape: (bs, mcw, emb_sz)
        keys = self.key_proj(embs)  # shape: (bs, mcw, emb_sz)

        # Compute similarity scores
        weights = torch.einsum("bqe,bke->bqk", queries, keys)

        # Zero out scores/weights to the right of the query
        weights = torch.einsum("bqk,qk->bqk", weights, mask)
        weights = torch.where(weights == 0.0, -torch.inf, weights)
        weights = F.softmax(weights, dim=1)  # (mcw, mcw), aka the (words, prob dist p across words)

        # Weighted average of the value representation for each word (projected from the word's embedding) according to weights
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

        # Project from the embedding of each token to the value representation of each token
        values = self.value_proj(embs)  # shape: (bs, mcw, emb_sz)
        out = torch.einsum("bpe,bwp->bwe", values, weights)  # shape: (bs, mcw, emb_sz)

        logits = self.linear(out)

        return logits

    def loss(self, pred, target):
        return F.cross_entropy(input=pred, target=target, reduction="mean")
