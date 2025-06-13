## DOCUMENT DETAILS ----------------------------------------------------------

# PROBABILITY MODELS

# Project: CDT in NLP Individual Project
# Working Title: Investigating Collocational Processing with Minerva2
# Author: Sydelle de Souza
# Institution: University of Edinburgh
# Supervisors: Dr Frank Mollica and Dr Alex Doumas
# Date: 2022/12/21
# Python version: 3.9.12

# -----------------------------------------------------------------------------#

## COMMENTS -------------------------------------------------------------------

# this file contains the code for the two probability models.
# model 1: probability of recognition given a probe (verb+noun embedding)
# & a memory matrix,
# model 2: probability of recognition given verb embedding +
# noised noun embedding & a memory matrix

# -----------------------------------------------------------------------------#


## Set-Up ---------------------------------------------------------------------
import logging
from typing import Literal, Optional, Union
import torch  # for tensors
import random  # for random number generation
import pandas as pd  # for dataframe manipulation
import os  # for file management
import pickle  # for saving and loading objects
import matplotlib.pyplot as plt  # for plotting
import numpy as np
from pathlib import Path
import json
import csv as csv  # for reading in the dataset, etc.
from joblib import Parallel, delayed  # for parallel processing
from filelock import FileLock
import argparse

# from extract_embeddings import (get_word_vector, get_fasttext_vector)

from run_one_experiment import get_embeddings

# -----------------------------------------------------------------------------#

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# -----------------------------------------------------------------------------#

# set up the model ------------------------------------------------------------#


class ProbModel(object):
    """This can be used to model the probability of recognition given a probe and a memory matrix"""

    def __init__(self, Mat):
        self.Mat = Mat
        self.M = Mat.shape[0]
        self.F = Mat.shape[1]

    def utility(self, probe, items_to_aggregate):
        """This function calculates the utility of the probe given each item in memory.
        each item appears multiple times in the memory matrix.
        we want to sum the utilities of all instances of the item
        then divide the softmax by the total sum"""

        # Calculate the similarity of the probe to each item in memory
        similarity = torch.cosine_similarity(probe, self.Mat, dim=1)

        # Apply softmax to the similarities
        softmax = torch.softmax(similarity, dim=0)

        # Aggregate the utilities of the items in memory
        sum_softmax = torch.sum(softmax[items_to_aggregate])

        # Ensure sum_softmax is a scalar
        assert (
            sum_softmax.ndim == 0
        ), f"sum_softmax should be a scalar, but has shape {sum_softmax.shape}"

        return similarity, sum_softmax


# -----------------------------------------------------------------------------#

# running experiments----------------------------------------------------------#


def run_iteration(
    p,
    s,
    device,
    colloc_embeddings,
    norm_freq_en,
    do_equal_frequency,
    M,
    embed_dim,
    forget_prob,
    noisy_probes: Optional[Literal["nouns", "verbs", "both"]] = None,
):
    # print(f"\nSeed {s}\n")
    torch_generator = torch.Generator().manual_seed(s)

    # stack the embeddings into a tensor
    colloc_bert_embeddings = torch.stack([c["vec"] for c in colloc_embeddings.values()]).to(
        "cpu"
    )  # move the embeddings to the cpu

    # normalize the embeddings to standard normal
    # TODO: why does normalizing per dimension produce drastically different results?
    # specifically, if norm by dim here and applying non-normed noise
    colloc_bert_embeddings = (
        colloc_bert_embeddings - colloc_bert_embeddings.mean()
    ) / colloc_bert_embeddings.std()

    # sample from the collocations to make a M x 768 matrix
    n_items = len(colloc_bert_embeddings)
    sample_k = M - n_items

    if do_equal_frequency:
        frequencies = torch.ones(n_items).float()
    else:
        frequencies = torch.tensor(norm_freq_en).float()

    sampled_item_indices = torch.cat(
        (
            torch.arange(n_items),
            torch.multinomial(frequencies, sample_k, replacement=True, generator=torch_generator),
        )
    )
    matrix = colloc_bert_embeddings[
        sampled_item_indices
    ]  # sampled_item_indices is a tensor of indices

    assert matrix.size() == (M, embed_dim), "Huh? Memory matrix has wrong shape."

    # noise the memory matrix
    noise_mean = torch.tensor([0.0]).expand(M, embed_dim)
    # tie noise to the st dev of the matrix
    noise_std = matrix.std().expand(M, embed_dim) / 2

    print(f"Noising with std {noise_std.mean()}")
    noise_gaussian = torch.normal(noise_mean, noise_std, generator=torch_generator)
    noise_mask = torch.rand((M, embed_dim), generator=torch_generator)  #
    noisy_mem = torch.where(
        noise_mask < forget_prob, matrix + noise_gaussian, matrix
    )  # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
    # noisy_mem = torch.where(
    #     noise_mask < L, 0.0, matrix
    # )  # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
    noisy_matrix = noisy_mem.to(device)

    probes = {}

    for item, v in colloc_embeddings.items():
        vec = v["vec"]
        assert vec.ndim == 1

        if (
            noisy_probes is not None
        ):  # if we are not using noisy nouns, keep the memory matrix as is
            half_embed_dim = (
                vec.size(0) // 2
            )  # splitting the memory matrix in half gives us verbs in the first half and nouns in the second half
            # as the memory matrix is a concatenation of the verb and noun embeddings
            # the first half of the memory matrix is verbs, second is nouns
            split_verb, split_noun = vec[:half_embed_dim], vec[half_embed_dim:]

            # mean and std of the embeddings of colloc_bert_embeddings, which we already normalized
            mean, std = 0.0, 1.0

            if noisy_probes == "nouns":
                # generate random noise for each embedding in the second half of the memory matrix
                noisy_split_noun = torch.normal(
                    mean, std, generator=torch_generator, size=(half_embed_dim,)
                )
                vec = torch.cat(
                    (split_verb, noisy_split_noun), 0
                )  # concatenate the two halves to get the noisy memory matrix

            elif noisy_probes == "verbs":
                noisy_split_verb = torch.normal(
                    mean, std, generator=torch_generator, size=(half_embed_dim,)
                )
                vec = torch.cat((noisy_split_verb, split_noun), 0)

            elif noisy_probes == "both":
                noisy_split_verb = torch.normal(
                    mean, std, generator=torch_generator, size=(half_embed_dim,)
                )
                noisy_split_noun = torch.normal(
                    mean, std, generator=torch_generator, size=(half_embed_dim,)
                )
                vec = torch.cat((noisy_split_verb, noisy_split_noun), 0)

            else:
                raise ValueError(
                    "noised_embeddings must be either None, 'nouns', 'verbs' or 'both'"
                )
        probes[item] = vec

    print(f"Memory matrix shape: {noisy_matrix.shape}")

    # Initialize the model
    minz = ProbModel(noisy_matrix)

    output = []  # initialize an empty list to store the output

    if os.environ.get("PROBMODEL_DEBUG"):  # if we are in debug mode, only use the first 10 items
        DEBUG_N = 10
        logging.warn(f"DEBUG MODE: only using first {DEBUG_N} items")
        probes = list(probes.items())[:DEBUG_N]
    else:
        probes = probes.items()

    for item_i, (item, vec) in enumerate(probes):
        assert item == list(colloc_embeddings.keys())[item_i]

        # get indices of item in sampled_item_indices
        item_indices = torch.where(sampled_item_indices == item_i)[0]

        similarity, utility = minz.utility(
            vec.to(device), item_indices
        )  # calculate the utility of the probe given the memory matrix

        output.append(
            [
                item,
                utility.item(),
                # similarity
            ]
        )
        print(
            f"Participant {p+1} \t| Seed {s}\t | Running on {device} \t| {output[-1][:3] if output else ''}"
        )

    print(
        f" Done with Participant {p+1} | Seed {s}  \n----------------------------------",
        flush=True,
    )
    results_df = pd.DataFrame(
        data=output,
        columns=[
            "item",
            "utility",
            # "similarity"
        ],
    )

    results_df["id"] = s
    results_df["participant"] = p + 1
    results_df["noisy_probes"] = noisy_probes

    return results_df


def run_experiment(
    dataset_to_use: str,
    kwics_file_to_use: str,
    num_participants: int,
    embedding_model="sbert",
    forget_prob=0.6,
    do_noise_embeddings=False,
    do_equal_frequency=False,
    do_log_freq=False,
    num_workers=1,
    do_concat_tokens=False,
    avg_last_n_layers=1,
    label=None,
    noisy_probe=None,
):
    ## read in the dataset
    df = pd.read_csv(dataset_to_use)
    dataset = df[["item"]]

    norm_freq_en = df["fitem"]

    if do_log_freq:
        norm_freq_en = norm_freq_en.apply(np.log10)

    if kwics_file_to_use == "none":
        kwics = None
    else:
        with open(kwics_file_to_use) as f:
            kwics = json.load(f)

    print("loaded the dataset and normalized the collocational frequencies")

    M = 10000  # number of items in memory

    embeddings_cache_filename = f'data/processed/{embedding_model}_{Path(dataset_to_use).name[:-4]}-last_{avg_last_n_layers}-{"kwics" if kwics else "nokwics"}{"-concat" if do_concat_tokens else ""}{"-" + label if label else ""}.dat'
    os.makedirs(os.path.dirname(embeddings_cache_filename), exist_ok=True)
    if not os.path.isfile(embeddings_cache_filename):
        colloc_embeddings = get_embeddings(
            dataset, kwics, embedding_model, do_concat_tokens, avg_last_n_layers
        )
        # write the embeddings dictionary to a file to be re-used next time we run the code
        with open(embeddings_cache_filename, "wb") as colloc2BERTfile:
            pickle.dump(colloc_embeddings, colloc2BERTfile)
        print("Dictionary written to file\n")

    else:
        # get the previously calculated embeddings from the file in which they were stored
        with open(embeddings_cache_filename, "rb") as colloc2BERTfile:
            colloc_embeddings = pickle.load(colloc2BERTfile)
        print(f"Read from file {embeddings_cache_filename}")

    embed_dim = 384 if embedding_model == "sbert" else 300
    if do_concat_tokens:
        embed_dim *= 2  # if we are concatenating the tokens, the embedding dimension is doubled

    if do_noise_embeddings:
        # generate random vectors for the items in the dataset
        # noise is generated from the mean and std of each embedding dimension
        colloc_bert_embeddings = torch.stack([c["vec"] for c in colloc_embeddings.values()])

        noise_means = colloc_bert_embeddings.mean(dim=0)
        noise_stds = colloc_bert_embeddings.std(dim=0)

        for item in colloc_embeddings:
            colloc_embeddings[item]["vec"].data = torch.randn(embed_dim) * noise_stds + noise_means

    ## Let's run our experiment. First we generate random seeds to simulate
    ## 99 l1 participants from Souza and Chalmers (2021)
    participant_seeds = []
    for _ in range(num_participants):
        participant_seeds.append(random.randint(0, 9999999))

    ## Now we run the experiment

    NUM_WORKERS = min(num_workers, num_participants)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        worker_devices = [torch.device(i) for i in range(n_gpus)]
        worker_devices = worker_devices * int(np.ceil(NUM_WORKERS / n_gpus))
    elif torch.has_mps:
        worker_devices = ["mps"] * NUM_WORKERS
    else:
        worker_devices = ["cpu"] * NUM_WORKERS
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using devices: {worker_devices}")

    # if os.path.exists(out_file):
    #     os.remove(out_file)
    # if os.path.exists(out_file + ".lock"):
    #     os.remove(out_file + ".lock")

    results = Parallel(n_jobs=NUM_WORKERS, backend="threading")(
        delayed(run_iteration)(
            p,
            s,
            worker_devices[p % NUM_WORKERS],
            colloc_embeddings,
            norm_freq_en,
            do_equal_frequency,
            M,
            embed_dim,
            forget_prob,
            noisy_probe,
        )
        for p, s in enumerate(participant_seeds)
    )

    results_df: pd.DataFrame = pd.concat(results, ignore_index=True)

    results_df["embedding_model"] = embedding_model
    results_df["is_noise_embeddings"] = do_noise_embeddings
    results_df["is_equal_frequency"] = do_equal_frequency
    results_df["avg_last_n_layers"] = avg_last_n_layers
    results_df["forget_prob"] = forget_prob
    results_df["noisy_probe"] = noisy_probe

    return results_df


# declare the dataset and the kwics file
df = "data/stimuli_idioms_clean.csv"
kwics = "data/stimuli_idioms_kwics.json"
toy = "data/toy_stimuli.csv"
# run the experiment


noisy_probes = ["nouns", "verbs", "both", None]

for noisy_probe in noisy_probes:
    results = run_experiment(
        df,
        kwics,
        5,
        embedding_model="sbert",
        forget_prob=0.6,
        do_noise_embeddings=False,
        do_equal_frequency=False,
        do_log_freq=False,
        num_workers=8,
        do_concat_tokens=True,
        avg_last_n_layers=1,
        label=None,
        noisy_probe=noisy_probe,
    )

    results.to_csv(
        f"data/processed/probmodel_results_noised_{noisy_probe}.csv",
        index=False,
    )