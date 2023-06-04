import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import logging

logging.set_verbosity_error()


def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, model, layers, concat_tokens=False):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    # Only select the tokens that constitute the requested word
    if concat_tokens:
        word_tokens_output = torch.flatten(output[1:3])
    else:
        word_tokens_output = output[1:3, :].mean(dim=0)

    return word_tokens_output


def get_word_vector(sent, tokenizer, model, layers, concat_tokens=False):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest

    return get_hidden_states(encoded, model, layers, concat_tokens=concat_tokens)


def main(layers=None):
    # Use last four layers by default
    layers = [-4, -3, -2, -1] if layers is None else layers
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)

    sent = "I like cookies ."
    idx = get_word_idx(sent, "cookies")

    word_embedding = get_word_vector(sent, idx, tokenizer, model, layers, concat_tokens=False)

    return word_embedding


if __name__ == '__main__':
    embed = main()
    print(embed.shape)
    print(embed)
