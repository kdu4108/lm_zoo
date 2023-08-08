############
# train.py #
############

"""
Decisions:
1. Dataset
    - Karpathy uses tiny shakespeare
2. Tokenize
    - Karpathy uses character level tokenization
3.

Notes:
- NLL Loss at the start of training should be around -log(1/vocab size) because it's predicting the right value uniformly at random over the vocab
- Get an early "hello world" that can do a forward pass and generate early and sanity check. e.g. a bigram model
- Good default LR for big networks: 3e-4. For bigger networks can get away with bigger, like 1e-3
- Decoder block: uses triangular masking so that current words can't see future ones! Encoder block: allows seeing full context, no triangular masking
- Why normalize by sqrt(head size)? Goal: reduce the variance of the embedding to 1. Why? If the variance is high, then softmax will make too pointy of a probability/weights distribution (focus too much on one word)
"""
import os
from typing import Tuple, Union

import torch

from tokenizers import CharacterTokenizer
from models import BigramLM, TrigramLM, NgramLM


def load_data(path: Union[str, os.PathLike]) -> str:
    """
    Loads the data from the given path into a string.
    """
    with open(path, "r") as f:
        return f.read()


def tokenize(text: str, tokenizer) -> torch.LongTensor:
    """
    Args:
        text - the raw text

    Returns:
        a 1D tensor consisting of all the tokens in text
    """
    return tokenizer.tokenize(text)


def make_batch(
    tokens: torch.LongTensor, batch_size: int, max_context_width: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Args:
        tokens: a (len(tokens),) shape tensor containing all the tokens in the dataset.


    Returns:
        (X, y) where:
        X is a tensor containing tokens of shape (batch_size, max_context_width), and
        y is a tensor containing tokens of shape (batch_size, max_context_width)
        where the token that comes after X[i, j] is y[i, j] (that is, X[i, j+1] = y[i, j])

    Note: training involves context from 1 to max_context_width - 1 number of preceding tokens, so there's actually batch_size * max_context_width training examples per batch.
    """
    start_idxs: torch.LongTensor = torch.randint(
        0, len(tokens) - max_context_width - 1, size=(batch_size,)
    )  # shape: (batch_size,)

    batch_X = torch.zeros(size=(batch_size, max_context_width), dtype=torch.long)
    batch_y = torch.zeros(size=(batch_size, max_context_width), dtype=torch.long)
    for i, start_idx in enumerate(start_idxs):
        batch_X[i, :] = tokens[start_idx : start_idx + max_context_width]
        batch_y[i, :] = tokens[
            (start_idx + 1) : (start_idx + 1) + max_context_width
        ]  # the label for each token is the next token in tokens

    return batch_X, batch_y


def train(
    model: torch.nn.Module,
    tokens: str,
    batch_size: int,
    max_context_width: int,
    num_iters: int,
    lr: float = 1e-3,
):
    """
    Trains a model on the given tokens.

    Args:
        model: the model to train
        tokens: the tokens to train on
        batch_size: the batch size to use
        max_context_width: the maximum context width to use
        num_iters: the number of iterations to train for
        lr: the learning rate to use

    Returns:
        the trained model
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    for i in range(num_iters):
        X, y = make_batch(
            tokens, batch_size=batch_size, max_context_width=max_context_width
        )  # X.shape: (batch_sz, max_context_width), y.shape: (batch_sz, max_context_width)
        optimizer.zero_grad()
        out: torch.FloatTensor = model(X)  # shape: (batch_sz, max_context_width, vocab_sz)
        out = torch.reshape(out, shape=(-1, out.shape[-1]))  # shape: (batch_sz * max_context_width, vocab_sz)
        y = torch.reshape(y, shape=(-1,))  # shape: (batch_sz * max_context_width)
        loss = model.loss(out, y)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(loss)

    return model


def generate(model, num_tokens: int, tokenizer):
    """
    Generate a sequence of tokens from the model.

    Args:
        model: the model to generate from
        num_tokens: the number of tokens to generate
        tokenizer: the tokenizer to use to decode the tokens

    Returns:
        a string containing the generated tokens
    """
    model.eval()
    tokens = [[0]]
    for _ in range(num_tokens):
        out = model(torch.tensor(tokens))  # (bs, mcw, vs)
        token = torch.multinomial(torch.softmax(out[0, -1, :], dim=0), 1).squeeze().item()
        tokens[0].append(token)

    model.train()

    return tokenizer.batch_decode(tokens[0])


def main():
    NUM_ITERS = 10000
    BATCH_SIZE = 8
    MAX_CONTEXT_WIDTH = 16
    NUM_GEN_TOKENS = 500

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device: {device}")

    text: str = load_data("data/tiny-shakespeare.txt")
    tokenizer = CharacterTokenizer(vocab=set(text))
    tokens: torch.LongTensor = tokenize(text, tokenizer=tokenizer)
    # model = BigramLM(vocab_sz=tokenizer.vocab_sz).to(device=device)
    # model = TrigramLM(vocab_sz=tokenizer.vocab_sz).to(device=device)
    model = NgramLM(vocab_sz=tokenizer.vocab_sz, N=10).to(device=device)
    print(
        "Text before training:",
        generate(model, num_tokens=NUM_GEN_TOKENS, tokenizer=tokenizer),
    )
    train(
        model,
        tokens,
        batch_size=BATCH_SIZE,
        max_context_width=MAX_CONTEXT_WIDTH,
        num_iters=NUM_ITERS,
    )
    print(
        f"Text after training for {NUM_ITERS} iters:",
        generate(model, num_tokens=NUM_GEN_TOKENS, tokenizer=tokenizer),
    )


if __name__ == "__main__":
    torch.random.manual_seed(0)
    main()
