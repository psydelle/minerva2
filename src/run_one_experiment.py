## DOCUMENT DETAILS ----------------------------------------------------------

# Project: CDT in NLP Individual Project
# Working Title: Investigating Collocational Processing with Minerva2
# Author: Sydelle de Souza
# Institution: University of Edinburgh
# Supervisors: Dr Frank Mollica and Dr Alex Doumas
# Date: 2022/12/21
# Python version: 3.9.12

# -----------------------------------------------------------------------------#

## COMMENTS -------------------------------------------------------------------

# this file contains the code for the MINERVA2 model, a simulation model
# of human memory, which we are using to investigate collocational processing.
# The model accounts for data from both episodic and semantic memory from a single
# system. Theoretically speaking, the model comprises a long-term memory system
# as well as a short-term memory system that can communicate with the
# each other. The long-term memory system is a matrix of M x N, where M is the
# The short-term memory system can send a "probe" to the long-term memory system
# and the long-term memory system can reply with an "echo".

# -----------------------------------------------------------------------------#

## ACKNOWLEDGEMENTS  ----------------------------------------------------------

# Ivan Vegner
# Sean Memery
# Giulio Zhou
# Mattia Opper

# -----------------------------------------------------------------------------#

## Set-Up ---------------------------------------------------------------------
import logging
from typing import Optional, Union
import torch  # for tensors
import random  # for random number generation
import pandas as pd  # for dataframe manipulation
import os  # for file management
import pickle  # for saving and loading objects
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt  # for plotting
import numpy as np
from pathlib import Path
import json

import csv as csv  # for reading in the dataset, etc.
from joblib import Parallel, delayed  # for parallel processing
from filelock import FileLock
import argparse

from extract_embeddings import get_word_vector  # for BERT embeddings

# from newest_regression import CollocNet
# -----------------------------------------------------------------------------#

# set the random seeds for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# set current working directory to this folder

# os.chdir(Path(__file__).parent.absolute())

# EN_BERT = "distilbert-base-cased"
# PT_BERT = "adalbertojunior/distilbert-portuguese-cased"
EN_BERT = "sentence-transformers/all-MiniLM-L6-v2"
NORM_ALPHA = 0.9



class Minerva2(object):
    """
    This is a class for the Minerva2 model
    """

    def __init__(self, F=None, M=None, Mat=None):
        if Mat is not None:
            self.Mat = Mat
            self.M = Mat.shape[0]
            self.F = Mat.shape[1]
        else:
            assert F is not None, "You need to specify the number of features"

    def activate(self, probe, tau=1.0):
        similarity = torch.cosine_similarity(
            probe, self.Mat, dim=1
        )  # had the wrong axis
        activation = torch.abs(similarity**tau) * torch.sign(
            similarity
        )  # make sure we preserve the signs
        return activation

    def echo(self, probe, tau=1.0):
        activation = self.activate(probe, tau)
        return torch.tensordot(activation, self.Mat, dims=([0], [0]))

    def recognize(
        self, probe, tau=1.0, k=0.955, maxiter=450
    ):  # maxiter is set to 450 because Souza and Chalmers (2021) set their timeout to 4500ms
        echo = self.echo(probe, tau)
        big = torch.cosine_similarity(echo, probe, dim=0)
        # similarity = torch.cosine_similarity(echo, self.Mat, dim=1)
        # big = torch.max(similarity)
        if big < k and tau < maxiter:
            big, tau = self.recognize(probe, tau + 1, k, maxiter)
        return big, tau


def get_embeddings(
    dataset: pd.DataFrame,
    kwics: dict,
    model_name: str,
    do_concat_tokens: bool,
    avg_last_n_layers: int,
):
    colloc2BERT = dict()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"

    if model_name == "sbert":
        # set up the model and tokenizer for SBERT embeddings
        tokenizer = AutoTokenizer.from_pretrained(EN_BERT)
        model = AutoModel.from_pretrained(EN_BERT, output_hidden_states=True).to(device)

        def grab_bert(contexts, context_words, model, tokenizer):
            # layers = [0] +  list(range(-avg_last_n_layers, 0))
            layers = list(range(-avg_last_n_layers, 0))
            return get_word_vector(
                contexts,
                context_words,
                tokenizer,
                model,
                layers,
                concat_tokens=do_concat_tokens,
            )

        for item in dataset["item"]:
            if kwics and kwics[item]["kwics"]:
                colloc_kwics = kwics[item]["kwics"]
                colloc_kwics_words = kwics[item]["kwic_words"]
                n_kwics = len(colloc_kwics)
            else:
                raise ValueError(f"No kwics provided for {item}")
                colloc_kwics = [item]
                colloc_kwics_words = [item.split(" ")]
                n_kwics = 0

            print(f'Retrieving vector for "{item}" from {model.config._name_or_path}')
            vec = grab_bert(colloc_kwics, colloc_kwics_words, model, tokenizer)

            # dictionary contains en keys regardless of what the embeddings are
            colloc2BERT[item] = {"vec": vec.to("cpu"), "n_kwics": n_kwics}

    return colloc2BERT


##-----------------------------------------------------------------------------##


def run_experiment(
    dataset_to_use: str,
    kwics_file_to_use: str,
    num_participants: int,
    embedding_model="sbert",
    do_noise_embeddings=False,
    do_equal_frequency=False,
    minerva_k=0.955,
    minerva_max_iter=450,
    num_workers=1,
    do_concat_tokens=False,
    avg_last_n_layers=4,
    label=None,
    # do_log_freq=True
):
    ## read in the dataset
    df = pd.read_csv(dataset_to_use)
    dataset = df[["item"]]

    # # convert comma-sep number strings to numbers
    # _coll_freq_en = df[["fitem", "fnode", "fcoll"]].applymap(
    #     lambda x: float(str(x).replace(",", ""))
    # )

    # if do_log_freq:
    #     _coll_freq_en = (_coll_freq_en+2).applymap(np.log10)
    #     _coll_freq_pt = (_coll_freq_pt+2).applymap(np.log10)

    # norm_freq_en = norm_jm(
    #     _coll_freq_en["fitem"], _coll_freq_en["fnode"], _coll_freq_en["fcoll"]
    # )
    norm_freq_en = df["fitem"]

    if kwics_file_to_use == "none":
        kwics = None
    else:
        with open(kwics_file_to_use) as f:
            kwics = json.load(f)

    print("loaded the dataset and normalized the collocational frequencies")

    M = 10000

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

    embed_dim = 384 if not do_concat_tokens else 384 * 2

    if do_noise_embeddings:
        # generate random vectors for the items in the dataset
        # noise is generated from the mean and std of each embedding dimension
        colloc_bert_embeddings = torch.stack(
            [c["vec"] for c in colloc_embeddings.values()]
        )

        noise_means = colloc_bert_embeddings.mean(dim=0)
        noise_stds = colloc_bert_embeddings.std(dim=0)

        for item in colloc_embeddings:
            colloc_embeddings[item]["vec"].data = (
                torch.randn(embed_dim) * noise_stds + noise_means
            )

    # stack the embeddings into a tensor
    colloc_bert_embeddings = torch.stack([c["vec"] for c in colloc_embeddings.values()])

    # normalize the embeddings to standard normal
    # TODO: why does normalizing per dimension produce drastically different results?
    # specifically, if norm by dim here and applying non-normed noise
    colloc_bert_embeddings = (
        colloc_bert_embeddings - colloc_bert_embeddings.mean()
    ) / colloc_bert_embeddings.std()

    # norm by row, (as suggested in SBERT?)
    # colloc_bert_embeddings = colloc_bert_embeddings / colloc_bert_embeddings.norm(dim=1).unsqueeze(1)

    ## Now we got to add some noise to the memory matrix (parameter L)
    L = 0.6  # 0.6 is what the meta paper says
    # noise between 0 and 1

    ## Let's run our experiment. First we generate random seeds to simulate
    ## 99 l1 participants from Souza and Chalmers (2021)
    participant_seeds = []
    for _ in range(num_participants):
        participant_seeds.append(random.randint(0, 9999999))

    ## Now we run the experiment

    def iter(p, s, device):
        # print(f"\nSeed {s}\n")
        random_generator = random.Random(s)
        torch_generator = torch.Generator().manual_seed(s)

        # sample from the collocations to make a M x 768 matrix
        sample_k = M - len(colloc_bert_embeddings)

        if do_equal_frequency:
            sampled_collocs = torch.stack(
                random_generator.choices(colloc_bert_embeddings, k=sample_k)
            )
        else:
            sampled_collocs = torch.stack(
                random_generator.choices(
                    colloc_bert_embeddings, k=sample_k, weights=norm_freq_en
                )
            )

        matrix = torch.concat([colloc_bert_embeddings, sampled_collocs], dim=0)

        assert matrix.size() == (M, embed_dim), "Huh?"

        # TODO: document noise procedure
        # again, why is noising per dimension so different?
        noise_mean = torch.tensor([0.0]).expand(M, embed_dim)
        noise_std = (
            matrix.std().expand(M, embed_dim) / 2
        )  # tie noise to the std of the matrix

        print(f"Noising with std {noise_std.mean()}")
        noise_gaussian = torch.normal(noise_mean, noise_std, generator=torch_generator)
        noise_mask = torch.rand((M, embed_dim), generator=torch_generator)
        noisy_mem = torch.where(
            noise_mask < L, matrix + noise_gaussian, matrix
        )  # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
        # noisy_mem = torch.where(
        #     noise_mask < L, 0.0, matrix
        # )  # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
        noisy_mem = noisy_mem.to(device)

        minz = Minerva2(
            Mat=noisy_mem
        )  # initialize the Minerva2 model with the noisy memory matrix

        # print(f"\nBegin simulation: {n} L1 Subjects\n---------------------------------")

        output = []  # initialize an empty list to store the output

        if os.environ.get("MINERVA_DEBUG"):
            DEBUG_N = 10
            logging.warn(f"DEBUG MODE: only using first {DEBUG_N} collocations")
            items = list(colloc_embeddings.items())[:DEBUG_N]
        else:
            items = colloc_embeddings.items()

        for item, data in items:
            vec = data["vec"]
            act, rt = minz.recognize(
                vec.to(device), k=minerva_k, maxiter=minerva_max_iter
            )
            output.append([item, act.detach().cpu().item(), rt, data["n_kwics"]])
            print(
                f"Participant {p+1} \t| Seed {s}\t | Running on {device} \t| {output[-1] if output else ''}"
            )

        print(
            f" Done with Participant {p+1} | Seed {s}  \n----------------------------------",
            flush=True,
        )
        results_df = pd.DataFrame(
            data=output,
            columns=["item", "act", "rt", "n_kwics"],
        )
        # results_df["mode"] = "l1"
        results_df["id"] = s
        results_df["participant"] = p + 1

        return results_df

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
        delayed(iter)(p, s, worker_devices[p % NUM_WORKERS])
        for p, s in enumerate(participant_seeds)
    )

    results_df = pd.concat(results, ignore_index=True)
    results_df["embedding_model"] = embedding_model
    results_df["is_noise_embeddings"] = do_noise_embeddings
    results_df["is_equal_frequency"] = do_equal_frequency
    results_df["minerva_k"] = minerva_k
    results_df["minerva_max_iter"] = minerva_max_iter
    results_df["avg_last_n_layers"] = avg_last_n_layers

    return results_df

    # if os.path.exists(out_file + ".lock"):
    #     os.remove(out_file + ".lock")

    print("****************************\n\nAll done!\n\n****************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_to_use", help="Dataset to use", default="data/stimuli_idioms.csv"
    )
    parser.add_argument(
        "-k",
        "--kwics_file_to_use",
        help="Kwics complement to dataset to use, or 'none' to use no kwics",
        default="data/stimuli_idioms_kwics.json",
    )
    parser.add_argument(
        "-n",
        "--num_participants",
        help="How many participants to model?",
        default=99,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--embedding_model",
        help="Which model to use for embeddings (sbert, fasttext)",
        default="sbert",
        choices=["sbert", "fasttext"],
    )
    parser.add_argument(
        "--do_noise_embeddings",
        help="Use random noise embeddings",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--do_equal_frequency",
        help="Sample collocations with equal frequency",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--minerva_k",
        help="Minerva k (threshold) parameter",
        default=0.93,
        type=float,
    )
    parser.add_argument(
        "--minerva_max_iter",
        help="Minerva max_iter parameter",
        default=300,
        type=int,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers to use",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--no_concat_tokens",
        dest="concat_tokens",
        help="Concatenate BERT tokens instead of averaging",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--avg_n",
        "--avg_last_n_layers",
        dest="avg_last_n_layers",
        help="Average last n layers of BERT",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--label",
        help="Arbitrary label to append to all files created",
        default=None,
    )
    parser.add_argument(
        "--append_to_file",
        help="Filename of existing csv file to append results to",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    assert Path(args.dataset_to_use).name == "stimuli_idioms_clean.csv"

    results_df = run_experiment(
        dataset_to_use=args.dataset_to_use,
        kwics_file_to_use=args.kwics_file_to_use,
        num_participants=args.num_participants,
        embedding_model=args.embedding_model,
        do_noise_embeddings=args.do_noise_embeddings,
        do_equal_frequency=args.do_equal_frequency,
        minerva_k=args.minerva_k,
        minerva_max_iter=args.minerva_max_iter,
        num_workers=args.num_workers,
        do_concat_tokens=args.concat_tokens,
        avg_last_n_layers=args.avg_last_n_layers,
        label=args.label,
    )

    if args.append_to_file:
        results_df.to_csv(args.append_to_file, mode="a", header=False, index=False)
        print(f"Appended results to {args.append_to_file}")
    else:
        out_file = f"results/results-{Path(args.dataset_to_use).name[:-4]}-{args.embedding_model}-{args.num_participants}p-{'noise' if args.do_noise_embeddings else ''}-{'equal_f' if args.do_equal_frequency else ''}-last_{args.avg_last_n_layers}-{'nokwics' if args.kwics_file_to_use=='none' else 'kwics'}{'-concat' if args.concat_tokens else ''}-m2k_{args.minerva_k}-m2mi_{args.minerva_max_iter}{'-' + args.label if args.label else ''}.csv"
        results_df.to_csv(out_file, index=False)
        print(f"Wrote results to {out_file}")
