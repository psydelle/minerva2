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

# -----------------------------------------------------------------------------#

## Set-Up ---------------------------------------------------------------------
from typing import Optional
import torch  # for tensors
import random  # for random number generation
import pandas as pd  # for dataframe manipulation
import os  # for file management
import pickle  # for saving and loading objects
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt  # for plotting
import numpy as np
from pathlib import Path

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

EN_BERT = "distilbert-base-cased"
PT_BERT = "adalbertojunior/distilbert-portuguese-cased"
NORM_ALPHA = 0.9


def norm_jm(fitem, fnode, fcoll):
    fitem_div = fitem / fitem.sum()
    fnode_div = fnode / fnode.sum()
    fcoll_div = fcoll / fcoll.sum()
    norm = NORM_ALPHA * fitem_div + (1 - NORM_ALPHA) * fnode_div * fcoll_div
    return norm


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
        similarity = torch.cosine_similarity(probe, self.Mat, dim=1)  # had the wrong axis
        # TODO: absolute value the exponentiation
        activation = (similarity**tau) * torch.sign(similarity)  # make sure we preserve the signs
        return activation

    def echo(self, probe, tau=1.0):
        activation = self.activate(probe, tau)
        return torch.tensordot(activation, self.Mat, dims=([0], [0]))

    def recognize(
        self, probe, tau=1.0, k=0.955, maxiter=450
    ):  # maxiter is set to 450 because Souza and Chalmers (2021) set their timeout to 4500ms
        echo = self.echo(probe, tau)
        similarity = torch.cosine_similarity(echo, self.Mat, dim=1)
        big = torch.max(similarity)
        if big < k and tau < maxiter:
            big, tau = self.recognize(probe, tau + 1, k, maxiter)
        return big, tau


##-----------------------------------------------------------------------------##


def run_experiment(
    dataset_to_use: str,
    space_lang: str,
    frequency_lang: str,
    num_participants: int,
    en_pt_trans_pickle: Optional[str] = None,
    freq_fraction_pt: Optional[float] = None,
    minerva_k=0.955,
    minerva_max_iter=450,
    num_workers=1,
    concat_tokens=False,
    label=None,
):
    ## read in the dataset
    if Path(dataset_to_use).name == "stimuli.csv":
        df = pd.read_csv(dataset_to_use)  # same dataset as MSc project
        # fcoll = list(df['fcoll'].str.replace(r'\D', '')) # collocation frequencies
        dataset = df[["item", "item_pt"]]
        # if space_lang in ["en", "en_aligned"]:
        #     dataset = list(df['item']) # list of items
        # elif space_lang == "pt":
        #     dataset = list(df['item_pt']) # list of items
        # else:
        #     raise NotImplementedError(f"{space_lang} space lang not implemented yet")

        # convert comma-sep number strings to numbers
        _coll_freq_en = df[["fitem", "fnode", "fcoll"]].applymap(
            lambda x: float(str(x).replace(",", ""))
        )
        # no replacement because it's actually numbers
        _coll_freq_pt = df[["fitempt", "fnodept", "fcollpt"]].astype(float)

        norm_freq_en = norm_jm(
            _coll_freq_en["fitem"], _coll_freq_en["fnode"], _coll_freq_en["fcoll"]
        )
        norm_freq_pt = norm_jm(
            _coll_freq_pt["fitempt"], _coll_freq_pt["fnodept"], _coll_freq_pt["fcollpt"]
        )

        # if frequency_lang == "en":
        #     fcoll = norm_freq_en
        # elif frequency_lang == "pt":
        #     fcoll = norm_freq_pt
        # elif frequency_lang == "mix":
        #     raise NotImplementedError("Mix not implemented yet hehe")

    elif dataset_to_use == "FinalDataset.csv":
        raise NotImplementedError(
            "FinalDataset.csv doesn't yet support any of the parameters you're probably trying to use"
        )
        df = pd.read_csv("FinalDataset.csv")
        dataset = list(df["item"])  # list of items
        fcoll = list(df["collFrequency"])  # collocation frequencies

    print("loaded the dataset and normalized the collocational frequencies")

    M = 10000

    bert_embeddings_cache_filename = f'data/processed/colloc2BERT-{Path(dataset_to_use).name[:-4]}-lang_{space_lang}{"-concat" if concat_tokens else ""}{"-" + label if label else ""}.dat'
    if not os.path.isfile(bert_embeddings_cache_filename):
        # set up the model and tokenizer for BERT embeddings
        def get_bert(mod_name="distilbert-base-uncased"):
            tokenizer = AutoTokenizer.from_pretrained(mod_name)
            model = AutoModel.from_pretrained(mod_name, output_hidden_states=True)
            return tokenizer, model

        def grab_bert(colloc, model, tokenizer, layers=[-4, -3, -2, -1]):
            # TODO: use only one layer
            return get_word_vector(colloc, tokenizer, model, layers, concat_tokens=concat_tokens)

        # grab BERT embeddings for the items in the dataset
        colloc2BERT = dict()
        if space_lang == "en":
            tokenizer, model = get_bert(EN_BERT)
        elif space_lang == "pt":
            tokenizer, model = get_bert(PT_BERT)
        elif space_lang == "en_aligned":
            # en_aligned is using en embeddings aligned to pt space, i.e., replace en embeddings with en in pt space
            with open(en_pt_trans_pickle, "rb") as f:
                en_pt_dict = pickle.load(f)
        else:
            raise Exception("Weird space lang")

        # if "aligned" in space_lang:
        #     alignment_net = CollocNet(embed_dim=768 if not concat_tokens else 768*2)

        # for item in dataset:
        #     if space_lang == "en_aligned":
        #         vec = alignment_net.en_to_pt(vec)
        #     elif space_lang == "pt_aligned":
        #         vec = alignment_net.pt_to_en(vec)

        for item_en, item_pt in zip(dataset["item"], dataset["item_pt"]):
            item = item_en if space_lang in ["en", "en_aligned"] else item_pt

            print(
                f'Retrieving vector for "{item}" from {"dictionary" if "aligned" in space_lang else model.config._name_or_path}'
            )
            if space_lang == "en_aligned":
                # just get pt projection from dictionary
                vec = torch.tensor(en_pt_dict[item]["pt"])
            else:
                vec = grab_bert(item, model, tokenizer)

            # dictionary contains en keys regardless of what the embeddings are
            colloc2BERT[item_en] = vec

        # write the embeddings dictionary to a file to be re-used next time we run the code
        #
        colloc2BERTfile = open(bert_embeddings_cache_filename, "wb")
        pickle.dump(colloc2BERT, colloc2BERTfile)
        colloc2BERTfile.close()
        print("Dictionary written  to file\n")

    else:
        # get the previously calculated embeddings from the file in which they were stored
        #
        colloc2BERTfile = open(bert_embeddings_cache_filename, "rb")
        colloc2BERT = pickle.load(colloc2BERTfile)
        colloc2BERTfile.close()
        print(f"Read from file {bert_embeddings_cache_filename}")

    # if en_pt_trans_pickle:
    #     with open(en_pt_trans_pickle, "rb") as f:
    #         en_pt_dict = pickle.load(f)
    #     for k in colloc2BERT:
    #         # print(np.array(colloc2BERT[k])[:10], np.array(en_pt_dict[k]["en"])[:10])
    #         # assert all(np.array(colloc2BERT[k]) == np.array(en_pt_dict[k]["en"]))

    #         # TODO: en embeddings here don't match en embeddings there bc uncased vs. cased models
    #         # TODO: rerun all experiments with cased bert if time
    #         colloc2BERT[k] = torch.tensor(en_pt_dict[k]["pt"])

    # stack the embeddings into a tensor
    colloc_bert_embeddings = torch.stack(list(colloc2BERT.values()))

    ## Now we got to add some noise to the memory matrix (parameter L)
    L = 0.6  # 0.6 is what the meta paper says
    # noise between 0 and 1

    ## Let's run our experiment. First we generate random seeds to simulate
    ## 99 l1 participants from Souza and Chalmers (2021)
    n = num_participants  # sample size
    p = 0
    seed = []
    for s in range(n):
        seed.append(random.randint(0, 9999999))

    ## Now we run the experiment

    def iter(p, s, device):
        # print(f"\nSeed {s}\n")
        random_generator = random.Random(s)
        torch_generator = torch.Generator().manual_seed(s)

        # sample from the collocations to make a M x 768 matrix
        sample_k = M - len(colloc_bert_embeddings)

        if frequency_lang == "en":
            sampled_collocs = torch.stack(
                random_generator.choices(colloc_bert_embeddings, k=sample_k, weights=norm_freq_en)
            )
        elif frequency_lang == "pt":
            sampled_collocs = torch.stack(
                random_generator.choices(colloc_bert_embeddings, k=sample_k, weights=norm_freq_pt)
            )
        elif frequency_lang == "mix":
            sample_k_pt = round(sample_k * freq_fraction_pt)
            sample_k_en = sample_k - sample_k_pt
            _sampled_collocs_pt = torch.stack(
                random_generator.choices(
                    colloc_bert_embeddings, k=sample_k_pt, weights=norm_freq_pt
                )
            )
            _sampled_collocs_en = torch.stack(
                random_generator.choices(
                    colloc_bert_embeddings, k=sample_k_en, weights=norm_freq_en
                )
            )
            sampled_collocs = torch.concat([_sampled_collocs_pt, _sampled_collocs_en], dim=0)

        matrix = torch.concat([colloc_bert_embeddings, sampled_collocs], dim=0)

        embed_dim = 768
        if concat_tokens:
            embed_dim = 768 * 2

        assert matrix.size() == (M, embed_dim), "Huh?"

        noise_gaussian = torch.normal(0, 1, (M, embed_dim), generator=torch_generator)
        # noise is a tensor of random numbers between 0 and 1
        noise_mask = torch.rand((M, embed_dim), generator=torch_generator)
        noisy_mem = torch.where(
            noise_mask < L, matrix + noise_gaussian, matrix
        )  # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
        noisy_mem = noisy_mem.to(device)

        minz = Minerva2(Mat=noisy_mem)  # initialize the Minerva2 model with the noisy memory matrix

        # print(f"\nBegin simulation: {n} L1 Subjects\n---------------------------------")

        output = []  # initialize an empty list to store the output

        for item, vector in list(colloc2BERT.items()):
            # vector = colloc2BERT['forget dream']
            act, rt = minz.recognize(vector.to(device), k=minerva_k, maxiter=minerva_max_iter)
            output.append([item, act.detach().cpu().item(), rt])
            print(
                f"Participant {p+1} \t| Seed {s}\t | Running on {device} \t| {output[-1] if output else ''}"
            )

            # # set up a dataframe to write the current results to a uniquely-named CSV file

            # results_l1 = pd.DataFrame(
            #     data={
            #         "mode": "l1",
            #         "id": [s],
            #         "participant": [p + 1],
            #         "item": [item],
            #         "act": [act.item()],
            #         "rt": [rt],
            #     }
            # )

            # with FileLock(out_filename + ".lock"):
            #     if not os.path.exists(out_filename):
            #         # delete the file if it exists and write the dataframe with column names
            #         # to the top of the new file
            #         results_l1.to_csv(out_filename, mode="w", header=True, index=False)
            #     else:
            #         # append the dataframe to the existing file without column names
            #         results_l1.to_csv(out_filename, mode="a", header=False, index=False)

        print(
            f" Done with Participant {p+1} | Seed {s}  \n----------------------------------",
            flush=True,
        )
        results_df = pd.DataFrame(
            data=output,
            columns=["item", "act", "rt"],
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
    # elif torch.has_mps:
    #     worker_devices = ["mps"] * NUM_WORKERS
    else:
        worker_devices = ["cpu"] * NUM_WORKERS
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using devices: {worker_devices}")

    # if os.path.exists(out_file):
    #     os.remove(out_file)
    # if os.path.exists(out_file + ".lock"):
    #     os.remove(out_file + ".lock")

    results = Parallel(n_jobs=NUM_WORKERS, backend="threading")(
        delayed(iter)(p, s, worker_devices[p % NUM_WORKERS]) for p, s in enumerate(seed)
    )

    results_df = pd.concat(results, ignore_index=True)
    results_df["space_lang"] = space_lang
    results_df["frequency_lang"] = frequency_lang
    results_df["minerva_k"] = minerva_k
    results_df["minerva_max_iter"] = minerva_max_iter
    results_df["freq_fraction_pt"] = freq_fraction_pt if frequency_lang == "mix" else -1


    return results_df

    # if os.path.exists(out_file + ".lock"):
    #     os.remove(out_file + ".lock")

    print("****************************\n\nAll done!\n\n****************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_to_use", help="Dataset to use", default="data/stimuli.csv")
    parser.add_argument(
        "-l",
        "--space_lang",
        help="Use embeddings from this language space (en, pt, en_aligned)",
        default="en",
        choices=["en", "pt", "en_aligned"],
    )
    parser.add_argument(
        "-f",
        "--frequency_lang",
        help="Use frequency counts from this language (en, pt or mix)",
        default="en",
        choices=["en", "pt", "mix"],
    )
    parser.add_argument(
        "-n",
        "--num_participants",
        help="How many participants to model?",
        default=99,
        type=int,
    )
    parser.add_argument(
        "--en_pt_trans_pickle",
        help="Path to pickle file containing pt translations of en embeddings (required for aligned space_lang)",
        default=None,
    )
    parser.add_argument(
        "--freq_fraction_pt",
        help="Use this fraction of PT in frequency mixing (only applicable for aligned space_lang, default 0.6)",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--minerva_k",
        help="Minerva k (threshold) parameter",
        default=0.955,
        type=float,
    )
    parser.add_argument(
        "--minerva_max_iter",
        help="Minerva max_iter parameter",
        default=450,
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

    results_df = run_experiment(
        dataset_to_use=args.dataset_to_use,
        space_lang=args.space_lang,
        frequency_lang=args.frequency_lang,
        num_participants=args.num_participants,
        en_pt_trans_pickle=args.en_pt_trans_pickle,
        freq_fraction_pt=args.freq_fraction_pt,
        minerva_k=args.minerva_k,
        minerva_max_iter=args.minerva_max_iter,
        num_workers=args.num_workers,
        concat_tokens=args.concat_tokens,
        label=args.label,
    )

    if args.append_to_file:
        results_df.to_csv(args.append_to_file, mode="a", header=False, index=False)
    else:
        out_file = f"results/results-{Path(args.dataset_to_use).name[:-4]}-{args.num_participants}p-lang_{args.space_lang}-freq_{args.frequency_lang}{f'-mix{args.freq_fraction_pt}' if args.frequency_lang == 'mix' else ''}{'-concat' if args.concat_tokens else ''}-m2k_{args.minerva_k}-m2mi_{args.minerva_max_iter}{'-' + args.label if args.label else ''}.csv"
        results_df.to_csv(out_file, index=False)
