import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import logging

logging.set_verbosity_error()


def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, model, layers, token_ids, concat_tokens=False):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze(0)
    # return output[0]

    # Only select the tokens that constitute the requested word
    word_embeddings = []
    for _tok_ids in token_ids:
        # average over subword tokens
        # https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
        subword_avg_embedding = output[_tok_ids].mean(dim=0)
        word_embeddings.append(subword_avg_embedding)

    if concat_tokens:
        word_tokens_output = torch.concat(word_embeddings, dim=0)
    else:
        word_tokens_output = torch.stack(word_embeddings, dim=0).mean(dim=0)

    return word_tokens_output


def get_word_vector(sent, tokenizer, model, layers, concat_tokens=False):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    # mask_sent = "I'm going to " + sent + " today."
    mask_sent = sent
    encoded = tokenizer.encode_plus(mask_sent, return_tensors="pt")
    # if encoded["input_ids"].size(1) != 4:
    #     print(
    #         f"Expected output to have 4 tokens, got {encoded['input_ids'].size(1)}."
    #         f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}"
    #     )

    # get all token idxs that belong to the word of interest
    word_0_idx = get_word_idx(mask_sent, sent.split(" ")[0])
    word_1_idx = get_word_idx(mask_sent, sent.split(" ")[1])
    # word_0_idx = 0
    # word_1_idx = 1

    encoded_word_ids = np.array(encoded.word_ids())
    token_ids_0 = np.where(encoded_word_ids == word_0_idx)[0]
    token_ids_1 = np.where(encoded_word_ids == word_1_idx)[0]
    token_ids = (token_ids_0, token_ids_1)

    return get_hidden_states(encoded, model, layers, token_ids, concat_tokens=concat_tokens)


def main(layers=None):
    # Use last four layers by default
    layers = [-4, -3, -2, -1] if layers is None else layers
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)

    sent = "I like cookies ."
    idx = get_word_idx(sent, "cookies")

    word_embedding = get_word_vector(sent, idx, tokenizer, model, layers, concat_tokens=False)

    return word_embedding


if __name__ == "__main__":
    embed = main()
    print(embed.shape)
    print(embed)
