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
import sys
from typing import Tuple, Union

import torch
import wandb

from tokenizers import CharacterTokenizer
from models import *


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
    tokens: torch.LongTensor, batch_size: int, max_context_width: int, device: torch.device
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

    return batch_X.to(device), batch_y.to(device)


def train_val_test_split(tokens: str, train_frac: float, val_frac: float, test_frac: float):
    """
    Splits the tokens into train, val, and test sets. NOTE: this does not shuffle the tokens,
    so beware if the text distribution changes over the course of the text.

    Args:
        tokens: the tokens to split
        train_frac: the fraction of tokens to use for training
        val_frac: the fraction of tokens to use for validation
        test_frac: the fraction of tokens to use for testing

    Returns:
        The train, val, and test sets, each as strings of tokens.
    """
    if train_frac + val_frac + test_frac != 1:
        raise ValueError("train_frac + val_frac + test_frac must equal 1")
    return (
        tokens[: int(train_frac * len(tokens))],
        tokens[int(train_frac * len(tokens)) : int((train_frac + val_frac) * len(tokens))],
        tokens[int((train_frac + val_frac) * len(tokens)) :],
    )


def validation(model, val_tokens: str, val_batch_sz: int, max_context_width: int, device: torch.device):
    """
    Args:
        model: the model to validate
        val_tokens: the tokens to validate on
        max_context_width: the maximum context width to use

    Returns:
        the validation loss
    """
    model.eval()
    num_iters = (len(val_tokens) - max_context_width) // val_batch_sz
    batches = []
    for i in range(num_iters):
        start_idxs = val_tokens[i * val_batch_sz : (i + 1) * val_batch_sz]
        batch_X = torch.zeros(size=(val_batch_sz, max_context_width), dtype=torch.long)
        batch_y = torch.zeros(size=(val_batch_sz, max_context_width), dtype=torch.long)
        for i, start_idx in enumerate(start_idxs):
            batch_X[i, :] = val_tokens[start_idx : start_idx + max_context_width]
            batch_y[i, :] = val_tokens[
                (start_idx + 1) : (start_idx + 1) + max_context_width
            ]  # the label for each token is the next token in tokens

        batches.append((batch_X.to(device), batch_y.to(device)))

    total_loss = 0
    total_num_examples = 0
    for batch_X, batch_y in batches:
        out = model(batch_X)
        out = torch.reshape(out, shape=(-1, out.shape[-1]))  # shape: (batch_sz * max_context_width, vocab_sz)
        batch_y = torch.reshape(batch_y, shape=(-1,))  # shape: (batch_sz * max_context_width)
        total_loss += model.loss(out, batch_y) * len(batch_y)
        total_num_examples += len(batch_y)
    model.train()

    return total_loss / total_num_examples


def train(
    model: torch.nn.Module,
    train_tokens: str,
    val_tokens: str,
    batch_size: int,
    val_batch_sz: int,
    max_context_width: int,
    num_iters: int,
    device: torch.device,
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
            train_tokens, batch_size=batch_size, max_context_width=max_context_width, device=device
        )  # X.shape: (batch_sz, max_context_width), y.shape: (batch_sz, max_context_width)
        optimizer.zero_grad()
        out: torch.FloatTensor = model(X)  # shape: (batch_sz, max_context_width, vocab_sz)
        out = torch.reshape(out, shape=(-1, out.shape[-1]))  # shape: (batch_sz * max_context_width, vocab_sz)
        y = torch.reshape(y, shape=(-1,))  # shape: (batch_sz * max_context_width)
        loss = model.loss(out, y)

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            val_loss = validation(
                model,
                val_tokens=val_tokens,
                val_batch_sz=val_batch_sz,
                max_context_width=max_context_width,
                device=device,
            )
            wandb.log({"val_loss": val_loss})
            print(f"Train loss: {loss}. Val loss: {val_loss}.")

        wandb.log({"train_loss": loss})

    return model


def generate(model, num_tokens: int, tokenizer, device: torch.device):
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
        out = model(torch.tensor(tokens, device=device))  # (bs, mcw, vs)
        token = torch.multinomial(torch.softmax(out[0, -1, :], dim=0), 1).squeeze().item()
        tokens[0].append(token)

    model.train()

    return tokenizer.batch_decode(tokens[0])


def main():
    SEED = 0
    NUM_ITERS = 10000
    BATCH_SIZE = 8
    VAL_BATCH_SIZE = 64
    MAX_CONTEXT_WIDTH = 16
    NUM_GEN_TOKENS = 500
    EMBEDDING_SZ = 16
    MODEL_NAME, MODEL_KWARGS = "WeightedAveragePrevEmbeddingsLM", dict(
        emb_sz=EMBEDDING_SZ
    )  # This must match the class name exactly
    # MODEL_NAME, MODEL_KWARGS = "BigramLM", dict()  # This must match the class name exactly
    # MODEL_NAME, MODEL_KWARGS = "TrigramLM", dict(emb_sz=EMBEDDING_SZ)  # This must match the class name exactly
    # MODEL_NAME, MODEL_KWARGS = "NgramLM", dict(emb_sz=EMBEDDING_SZ)  # This must match the class name exactly
    params_to_log = locals()

    WANDB_PROJECT_NAME = "lm_zoo"
    TAGS = []
    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        entity="kdu",
        # group=GROUP_NAME,
        config=params_to_log,
        tags=TAGS,
    )

    torch.random.manual_seed(SEED)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    text: str = load_data("data/tiny-shakespeare.txt")
    tokenizer = CharacterTokenizer(vocab=set(text))
    tokens: torch.LongTensor = tokenize(text, tokenizer=tokenizer)
    train_tokens, val_tokens, test_tokens = train_val_test_split(tokens, 0.99, 0.001, 0.009)
    print(len(train_tokens), len(val_tokens), len(test_tokens))

    # model = BigramLM(vocab_sz=tokenizer.vocab_sz).to(device=device)
    # model = TrigramLM(vocab_sz=tokenizer.vocab_sz).to(device=device)
    # model = NgramLM(vocab_sz=tokenizer.vocab_sz, N=10).to(device=device)
    model = getattr(sys.modules[__name__], MODEL_NAME)(vocab_sz=tokenizer.vocab_sz, **MODEL_KWARGS).to(device=device)

    text_before_training = generate(model, num_tokens=NUM_GEN_TOKENS, tokenizer=tokenizer, device=device)
    print("Text before training:", text_before_training)
    wandb.log({"text_before_training": text_before_training})

    train(
        model,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        batch_size=BATCH_SIZE,
        val_batch_sz=VAL_BATCH_SIZE,
        max_context_width=MAX_CONTEXT_WIDTH,
        num_iters=NUM_ITERS,
        device=device,
    )

    text_after_training = generate(model, num_tokens=NUM_GEN_TOKENS, tokenizer=tokenizer, device=device)
    print(f"Text after training for {NUM_ITERS} iters:", text_after_training)
    wandb.log({"text_after_training": text_after_training})


if __name__ == "__main__":
    main()
