from typing import *
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import logging

logging.set_verbosity_error()


def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, model, layers, batch_token_ids, concat_tokens=False):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and mean over layers
    output = torch.stack([states[i] for i in layers]).mean(0) # n_contexts x n_tokens x embed_dim
    # return output[0]

    # # Only select the tokens that constitute the requested word
    # token_embeddings = output[torch.arange(output.size(0)), token_ids]
    word_embeddings = []
    for i, ctx_token_ids in enumerate(batch_token_ids):
        ctx_embeddings = []
        for _tok_ids in ctx_token_ids:
            # average over subword tokens
            # https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
            subword_avg_embedding = output[i, _tok_ids].mean(dim=0)
            ctx_embeddings.append(subword_avg_embedding)
        word_embeddings.append(torch.stack(ctx_embeddings))

    # mean across contexts, i.e. batch
    word_embeddings = torch.stack(word_embeddings, dim=0).mean(dim=0)

    assert word_embeddings.size() == (2, 768)

    if concat_tokens:
        word_tokens_output = word_embeddings.reshape(-1)
    else:
        word_tokens_output = torch.mean(word_embeddings, dim=0)

    return word_tokens_output


def get_word_vector(
    contexts: List[str], 
    context_words:List[List[str]],
    tokenizer,
    model,
    layers,
    concat_tokens=False,
):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`.
    
    If contexts are provided, average token representations across all contexts.
    """
    # mask_sent = "I'm going to " + sent + " today."
                    
    encoded = tokenizer(contexts, padding=True, return_tensors="pt")
    # if encoded["input_ids"].size(1) != 4:
    #     print(
    #         f"Expected output to have 4 tokens, got {encoded['input_ids'].size(1)}."
    #         f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}"
    #     )

    def __get_word_from_word_id(batch_idx, cxt, word_id):
        word_span = encoded.word_to_chars(batch_idx, word_id)
        word = cxt[word_span[0]:word_span[1]]
        return word


    # get all token idxs that belong to the word of interest
    encoded_word2toks = []
    for i, (cxt, cxt_words) in enumerate(zip(contexts, context_words)):
        word2tokens = {}
        for word_id in encoded.word_ids(i):
            if word_id is not None:
                start, end = encoded.word_to_tokens(i, word_id)
                if start == end - 1:
                    tokens = [start]
                else:
                    tokens = list(range(start, end))
                if word_id not in word2tokens: # or word2tokens[-1] != tokens:
                    word2tokens[word_id]=tokens

        # find tokens corresponding to context_words in word2tokens
        word1_toks, word2_toks = None, None
        for word1_id in word2tokens:
            word1 = __get_word_from_word_id(i, cxt, word1_id)
            for offset in range(1, 8):
                word2_id = word1_id+offset
                # attempt to match second word
                if word1 == cxt_words[0] and __get_word_from_word_id(i, cxt, word2_id) == cxt_words[1]:
                    word1_toks = word2tokens[word1_id]
                    word2_toks = word2tokens[word2_id]
                    break
            if word1_toks and word2_toks:
                break

        if word1_toks and word2_toks:
            encoded_word2toks.append((word1_toks, word2_toks))
        else:
            raise ValueError("Could not find tokens for", cxt_words, "in", cxt)
        
        # encoded_word2toks.append(word2tokens)        

    # word_0_idx = get_word_idx(mask_sent, string.split(" ")[0])
    # word_1_idx = get_word_idx(mask_sent, string.split(" ")[1])
    # word_0_idx = 0
    # word_1_idx = 1

    # encoded_word_ids = np.array(encoded.word_ids())
    # token_ids_0 = np.where(encoded_word_ids == word_0_idx)[0]
    # token_ids_1 = np.where(encoded_word_ids == word_1_idx)[0]
    # token_ids = (token_ids_0, token_ids_1)

    return get_hidden_states(encoded, model, layers, encoded_word2toks, concat_tokens=concat_tokens)


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
