import logging
from typing import Optional, Tuple, Union
import torch  # for tensors
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
import random  # for random number generation
import pandas as pd  # for dataframe manipulation
import os  # for file management
import pickle  # for saving and loading objects
import numpy as np
from pathlib import Path
import json

import csv as csv  # for reading in the dataset, etc.
from joblib import Parallel, delayed  # for parallel processing
import argparse
import wandb
import wandb.data_types  # Weights and Biases for experiment tracking

from run_one_experiment import get_embeddings
from hflayers import Hopfield, HopfieldLayer


class HfModel(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        beta: float,
        wandb_run,
        stored_patterns: Optional[torch.Tensor] = None,
    ):
        super(HfModel, self).__init__()
        self.wandb_run = wandb_run

        if stored_patterns is not None:
            self.hopfield = Hopfield(
                input_size=embed_dim,
                hidden_size=hidden_size,
                stored_pattern_size=embed_dim,  # idk why necessary
                pattern_projection_size=embed_dim,  # idk why necessary
                scaling=beta,
            )
            self.stored_patterns = stored_patterns
        else:
            # learn stored patterns, i.e., "lookup table"
            pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute result of Hopfield network on specified data.

        :param input: data to be processed by the Hopfield network
        :return: result as computed by the Hopfield network
        """
        if self.stored_patterns is not None:
            p = self.stored_patterns
            # expand batch dimension of p to match stored patterns
            p = p.unsqueeze(0).expand(input.size(0), -1, -1)
            H = self.hopfield((p, input, p))
        else:
            # if no stored patterns are provided, we assume this is a lookup table
            # and we just return the input as output
            pass

        return H

    def calculate_retrieval_failures(
        self, input: torch.Tensor, target: torch.Tensor, threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute whether similarity between input and target is above a certain threshold.

        :param input: data to be processed by the Hopfield network
        :param target: target to be used to compute the retrieval error
        :param threshold: threshold to be used to compute the retrieval error
        :return: retrieval failure as well as the predicted output
        """
        # Process data by Hopfield-based network.
        Y_hat = self.forward(input)
        # Compute similarity between input and target.
        similarity = torch.nn.functional.cosine_similarity(Y_hat, target, dim=-1)
        assert (
            similarity.shape == input.shape[:-1]
        ), f"Expected similarity shape {input.shape[:-1]}, got {similarity.shape}"
        # Compute error based on threshold.
        error = torch.where(
            similarity > threshold,
            torch.tensor(0.0, device=input.device),
            torch.tensor(1.0, device=input.device),
        )

        return error, similarity, Y_hat

    def calculate_objective(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute objective of the current model.

        :param input: data to be processed by the Hopfield-based pooling network
        :param target: target to be used to compute the objective of the current model
        :return: objective as well as dummy A (see accompanying paper for more information)
        """
        # Process data by Hopfield-based network.
        H = self.forward(input)

        # Compute objective of current model.
        loss = torch.nn.functional.mse_loss(H, target.float())

        return loss


def train_epoch(
    network: HfModel, optimiser: AdamW, data_loader: DataLoader, threshold: float
) -> Tuple[float, float]:
    """
    Execute one training epoch.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    device = next(network.parameters()).device
    losses, failures = [], []
    for data, target in data_loader:
        data, target = data.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        loss = network.calculate_objective(data, target)

        # Update network parameters.
        optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        item_failures, _, _ = network.calculate_retrieval_failures(
            data, target, threshold=threshold
        )
        failures.append(item_failures.detach().sum().item())
        losses.append(loss.detach().item())

    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(failures)


def eval_iter(
    network: HfModel, data_loader_eval: DataLoader, threshold: float
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate the current model.

    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    device = next(network.parameters()).device
    with torch.no_grad():
        losses, failures, item_failures, item_sim, item_predictions = [], [], [], [], []

        for eval_x, eval_y in data_loader_eval:
            # Move data to the correct device.
            eval_x, eval_y = eval_x.to(device=device), eval_y.to(device=device)

            # Process data by Hopfield-based network.
            loss = network.calculate_objective(eval_x, eval_y)

            # Compute performance measures of current model.
            _item_failures, _item_sim, _item_predictions = network.calculate_retrieval_failures(
                eval_x, eval_y, threshold=threshold
            )
            failures.append(_item_failures.detach().sum().item())
            losses.append(loss.detach().item())
            item_failures.append(_item_failures.detach().cpu())
            item_sim.append(_item_sim.detach().cpu())
            item_predictions.append(_item_predictions.detach().cpu())

    # Concatenate item failures, similarities, and predictions.
    item_failures = torch.cat(item_failures, dim=0)
    item_sim = torch.cat(item_sim, dim=0)
    item_predictions = torch.cat(item_predictions, dim=0)

    # Report progress of validation procedure.
    return sum(losses) / len(losses), sum(failures), item_failures, item_sim, item_predictions


def operate(
    network: HfModel,
    optimiser: AdamW,
    data_loader_train: DataLoader,
    data_loader_eval: DataLoader,
    eval_df: pd.DataFrame,
    num_epochs: int,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param eval_df: dataframe containing evaluation metadata (e.g., item types)
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, failures = {r"train": [], r"eval": []}, {r"train": [], r"eval": []}
    for epoch in range(num_epochs):
        # Train network.
        t_loss, t_fails = train_epoch(network, optimiser, data_loader_train, threshold=threshold)
        losses[r"train"].append(t_loss)
        failures[r"train"].append(t_fails)
        network.wandb_run.log(
            {
                "epoch": epoch,
                "train/loss": t_loss,
                "train/failures": t_fails,
            }
        )
        # Evaluate current model.
        eval_df = eval_df.copy()
        e_loss, e_fails, item_failures, item_sim, item_predictions = eval_iter(network, data_loader_eval, threshold=threshold)
        assert item_failures.shape[0] == eval_df.shape[0], (
            f"Expected {eval_df.shape[0]} item failures, got {item_failures.shape[0]}"
        )
        eval_df["is_failure"] = item_failures.detach().cpu().numpy()
        eval_df["sim_to_correct"] = item_sim.detach().cpu().numpy()
        # eval_df["item_predictions"] = [p.tolist() for p in item_predictions.detach().cpu()]
        losses[r"eval"].append(e_loss)
        failures[r"eval"].append(e_fails)
        network.wandb_run.log(
            {
                "epoch": epoch,
                "eval/loss": e_loss,
                "eval/failures": e_fails,
                "eval/fail_per_type": eval_df.groupby("type")["is_failure"].sum().to_dict(),
                "eval/item_meta": wandb.Table(
                    columns=["item", "type", "verb", "noun", "is_failure", "sim_to_correct"],
                    data=eval_df[["item", "type", "verb", "noun", "is_failure", "sim_to_correct"]].values.tolist()
                ),
            }
        )
        print(
            f"Epoch {epoch} | Train Loss: {t_loss:.4f} #| Train Failures: {t_fails:.1f} | Eval Loss: {e_loss:.4f} | Eval Failures: {e_fails:.1f}"
        )
    network.wandb_run.finish()

    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(failures)


def run_iteration_lookup(
    p,
    s,
    device,
    colloc_embeddings,
    norm_freq_en,
    do_equal_frequency,
    M,
    embed_dim,
    forget_prob,
    minerva_k,
):
    """Run training for one participant, using the Hopfield Lookup model.

    For this, we do not need to noise embeddings.
    Model is trained to output unnoised probe given noisy probes as input.
    """
    pass


def run_iteration_general(
    p,
    s,
    device,
    df,  # now the full dataframe with embeddings and metadata
    norm_freq_en,
    do_equal_frequency,
    M,
    embed_dim,
    forget_prob,
    minerva_k,
    num_epochs=100,
    wandb_group_name="hopfield-general-experiment",
):
    """Run training for one participant, using the general Modern Hopfield model.

    For this, we give the model a noisy memory matrix
    Model is trained to output unnoised probe given unnoised probe as input.
    That is, it is an autoencoder-like model.
    """
    # print(f"\nSeed {s}\n")
    random_generator = random.Random(s)
    torch_generator = torch.Generator().manual_seed(s)

    # Use the dataframe to get all item info and embeddings
    if os.environ.get("MINERVA_DEBUG"):
        DEBUG_N = 10
        logging.warning(f"DEBUG MODE: only using first {DEBUG_N} collocations")
        df = df.iloc[:DEBUG_N]
        norm_freq_en = norm_freq_en[:DEBUG_N]

    # stack the embeddings into a tensor
    colloc_bert_embeddings = torch.stack(df["vec"].tolist()).to("cpu")
    # normalize the embeddings to standard normal
    # NEW: remove normalization
    # colloc_bert_embeddings = (
    #     colloc_bert_embeddings - colloc_bert_embeddings.mean()
    # ) / colloc_bert_embeddings.std()
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
    matrix = colloc_bert_embeddings[sampled_item_indices]

    assert matrix.size() == (M, embed_dim), "Huh?"

    # TODO: document noise procedure
    # again, why is noising per dimension so different?
    noise_mean = torch.tensor([0.0]).expand(M, embed_dim)
    # tie noise to the std of the matrix
    noise_std = matrix.std().expand(M, embed_dim) / 2

    print(f"Noising with std {noise_std.mean()}")
    noise_gaussian = torch.normal(noise_mean, noise_std, generator=torch_generator)
    noise_mask = torch.rand((M, embed_dim), generator=torch_generator)
    noisy_mem = torch.where(
        noise_mask < forget_prob, matrix + noise_gaussian, matrix
    )  # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
    # noisy_mem = torch.where(
    #     noise_mask < L, 0.0, matrix
    # )  # if the noise is less than L, then add gaussian noise, otherwise it is the original matrix
    noisy_mem = noisy_mem.to(device)

    beta = 1.0  # Set a default float value for beta for wandb compatibility
    hidden_size = 100
    # Pass a group name for wandb grouping (e.g., experiment label or run type)
    wandb_run = wandb.init(
        project="hopfield-experiments",
        group=wandb_group_name,
        config={
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "beta": beta,
            "participant": p + 1,
            "seed": s,
            "num_epochs": num_epochs,
            "device": str(device),
            "M": M,
            "forget_prob": forget_prob,
            "minerva_k": minerva_k,
            "do_equal_frequency": do_equal_frequency,
        },
        reinit="create_new",
    )

    hopfield = HfModel(
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        beta=beta,
        wandb_run=wandb_run,
        stored_patterns=noisy_mem,  # use the noisy memory matrix as the stored patterns
    ).to(device)

    # output = []  # initialize an empty list to store the output

    # stack the embeddings into a tensor
    train_x = colloc_bert_embeddings.unsqueeze_(1)  # add a sequence length dimension
    train_y = train_x.clone()

    optimiser = AdamW(params=hopfield.parameters(), lr=5e-4, weight_decay=1e-4)

    # create a data loader for the training data
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch_generator,
        # pin_memory=True,
        # persistent_workers=True,
    )
    # create a data loader for the validation data
    eval_data = torch.utils.data.TensorDataset(train_x, train_y)
    eval_loader = DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False, # MUST BE FALSE for evaluation
        num_workers=0,
        generator=torch_generator,
        # pin_memory=True,
        # persistent_workers=True,
    )
    eval_df = df.drop(columns=["vec"])

    print(
        f"Participant {p+1} | Seed {s} | Running on {device} | Training with {len(train_x)} samples, {len(train_y)} eval samples"
    )
    # train the model
    losses, failures = operate(
        network=hopfield,
        optimiser=optimiser,
        data_loader_train=train_loader,
        data_loader_eval=eval_loader,
        eval_df=eval_df,
        num_epochs=num_epochs,
        threshold=minerva_k,
    )
    print(losses, failures)

    # for item, data in items:
    #     vec = data["vec"]
    #     act, rt, activations_0, activations_tau = minz.recognize(
    #         vec.to(device), k=minerva_k, maxiter=minerva_max_iter
    #     )

    #     def agg_activations_by_item(activations):
    #         agg_activations = torch.zeros(n_items, device=device)
    #         for i in range(n_items):
    #             where = sampled_item_indices == i
    #             agg_activations[i] = activations[where].mean()
    #         return agg_activations

    #     output.append(
    #         [
    #             item,
    #             act.detach().cpu().item(),
    #             rt,
    #             data["n_kwics"],
    #             agg_activations_by_item(activations_0).detach().cpu().tolist(),
    #             agg_activations_by_item(activations_tau).detach().cpu().tolist(),
    #         ]
    #     )
    #     print(
    #         f"Participant {p+1} \t| Seed {s}\t | Running on {device} \t| {output[-1][:3] if output else ''}"
    #     )

    print(
        f" Done with Participant {p+1} | Seed {s}  \n----------------------------------",
        flush=True,
    )
    # results_df = pd.DataFrame(
    #     data=output,
    #     columns=[
    #         "item",
    #         "act",
    #         "rt",
    #         "n_kwics",
    #         "activations_0",
    #         "activations_tau",
    #     ],
    # )
    # # results_df["mode"] = "l1"
    # results_df["id"] = s
    # results_df["participant"] = p + 1

    # return results_df


def run_experiment(
    dataset_to_use: str,
    kwics_file_to_use: str,
    num_participants: int,
    embedding_model="sbert",
    forget_prob=0.6,
    do_noise_embeddings=False,
    do_equal_frequency=False,
    do_log_freq=False,
    minerva_k=0.955,
    num_workers=1,
    do_concat_tokens=False,
    avg_last_n_layers=1,
    label=None,
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
        # Add embedding vectors to the dataframe for each item

    embed_dim = 384 if embedding_model == "sbert" else 300
    if do_concat_tokens:
        embed_dim *= 2

    if do_noise_embeddings:
        # generate random vectors for the items in the dataset
        # noise is generated from the mean and std of each embedding dimension
        colloc_bert_embeddings = torch.stack([c["vec"] for c in colloc_embeddings.values()])

        noise_means = colloc_bert_embeddings.mean(dim=0)
        noise_stds = colloc_bert_embeddings.std(dim=0)

        for item in colloc_embeddings:
            colloc_embeddings[item]["vec"].data = torch.randn(embed_dim) * noise_stds + noise_means

    colloc_embeddings = [{"item": item, **d} for item, d in colloc_embeddings.items()]
    # join the embeddings to the dataframe
    df = df.join(
        pd.DataFrame(colloc_embeddings).set_index("item"),
        on="item",
        how="left",
    )

    participant_seeds = []
    for _ in range(num_participants):
        participant_seeds.append(random.randint(0, 9999999))

    ## Now we run the experiment

    NUM_WORKERS = min(num_workers, num_participants)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        worker_devices = [torch.device(i) for i in range(n_gpus)]
        worker_devices = worker_devices * int(np.ceil(NUM_WORKERS / n_gpus))
    elif torch.mps.is_available():
        worker_devices = ["mps"] * NUM_WORKERS
    else:
        worker_devices = ["cpu"] * NUM_WORKERS
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using devices: {worker_devices}")

    # if os.path.exists(out_file):
    #     os.remove(out_file)
    # if os.path.exists(out_file + ".lock"):
    #     os.remove(out_file + ".lock")

    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_group_name = f"hopfield-{current_time}"

    results = Parallel(n_jobs=NUM_WORKERS, backend="threading")(
        delayed(run_iteration_general)(
            p,
            s,
            worker_devices[p % NUM_WORKERS],
            df,  # pass the dataframe with embeddings and all columns
            norm_freq_en,
            do_equal_frequency,
            M,
            embed_dim,
            forget_prob,
            minerva_k,
            wandb_group_name=wandb_group_name,
        )
        for p, s in enumerate(participant_seeds)
    )

    results_df: pd.DataFrame = pd.concat(results, ignore_index=True)

    # # average the activations over all participants
    # activations_0 = results_df.groupby("item")["activations_0"].apply(
    #     lambda x: torch.tensor(x.tolist()).mean(dim=0)
    # )
    # activations_tau = results_df.groupby("item")["activations_tau"].apply(
    #     lambda x: torch.tensor(x.tolist()).mean(dim=0)
    # )

    results_df["embedding_model"] = embedding_model
    results_df["is_noise_embeddings"] = do_noise_embeddings
    results_df["is_equal_frequency"] = do_equal_frequency
    results_df["minerva_k"] = minerva_k
    results_df["avg_last_n_layers"] = avg_last_n_layers
    results_df["forget_prob"] = forget_prob

    return results_df

    # if os.path.exists(out_file + ".lock"):
    #     os.remove(out_file + ".lock")

    print("****************************\n\nAll done!\n\n****************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_to_use",
        help="Dataset to use",
        default="data/stimuli_idioms.csv",
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
        "-f",
        "--forget_prob",
        help="Probability of forgetting (noising an embedding dimension)",
        default=0.6,
        type=float,
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
        "--do_log_freq",
        help="Log-transform the frequency data before sampling",
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
        default=1,
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

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    args = parser.parse_args()

    assert Path(args.dataset_to_use).name == "stimuli_idioms_clean.csv"

    results_df = run_experiment(
        dataset_to_use=args.dataset_to_use,
        kwics_file_to_use=args.kwics_file_to_use,
        num_participants=args.num_participants,
        embedding_model=args.embedding_model,
        forget_prob=args.forget_prob,
        do_noise_embeddings=args.do_noise_embeddings,
        do_equal_frequency=args.do_equal_frequency,
        do_log_freq=args.do_log_freq,
        minerva_k=args.minerva_k,
        num_workers=args.num_workers,
        do_concat_tokens=args.concat_tokens,
        avg_last_n_layers=args.avg_last_n_layers,
        label=args.label,
    )

    if args.write_activations_json:
        activations_df = results_df[["item", "participant", "activations_0", "activations_tau"]]
        results_df = results_df.drop(columns=["activations_0", "activations_tau"])

    # # average the activations over all participants
    # activations_0 = results_df.groupby("item")["activations_0"].apply(
    #     lambda x: torch.tensor(x.tolist()).mean(dim=0)
    # )
    # activations_tau = results_df.groupby("item")["activations_tau"].apply(
    #     lambda x: torch.tensor(x.tolist()).mean(dim=0)
    # )

    if args.append_to_file:
        results_df.to_csv(args.append_to_file, mode="a", header=False, index=False)
        print(f"Appended results to {args.append_to_file}")
    else:
        out_file_stem = f"results/results-{Path(args.dataset_to_use).name[:-4]}-{args.embedding_model}-{args.num_participants}p-{'noise-' if args.do_noise_embeddings else ''}{'equal_f-' if args.do_equal_frequency else ''}last_{args.avg_last_n_layers}-{'nokwics' if args.kwics_file_to_use=='none' else 'kwics'}{'-concat' if args.concat_tokens else ''}-m2k_{args.minerva_k}-m2mi_{args.minerva_max_iter}{'-' + args.label if args.label else ''}"
        csv_file = out_file_stem + ".csv"
        results_df.to_csv(csv_file, index=False)
        if args.write_activations_json:
            json_file = out_file_stem + "_activations.json"
            # activations_df.to_json(json_file, orient="index")
            print(f"Wrote results to {csv_file} and {json_file}")
        else:
            print(f"Wrote results to {csv_file}")
