# Import the W&B Python Library
import random
import numpy as np
import torch
import wandb
from datetime import datetime
from run_hopfield_experiments import run_experiment_sweep_wrapper
import click


# 1: Define objective/training function
def run():
    results_df = run_experiment_sweep_wrapper()
    # return results_df["score"].mean()


@click.command()
@click.option(
    "--id",
    "--sweep_id",
    "sweep_id",
    type=str,
    default=None,
    help="W&B sweep ID to use for the experiment. If not provided, a new sweep will be created.",
)
@click.option(
    "--lookup/--memories",
    "lookup",
    default=True,
    help="If --lookup, models will be trained with lookup memory. "
    "If --memories, models will be trained with explicit noisy memories.",
)
def run_sweep(sweep_id: str, lookup: bool):
    # seed = 0

    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # 2: Define the search space

    if lookup:
        num_epochs = 100
        c = {
            "batch_size": {"values": [64]},
            "num_epochs": {"values": [num_epochs]},
            # "memory_size": {"values": [3] + list(range(10, 100, 20)) + list(range(100, 1000, 100))},
            "memory_size": {"min": 3, "max": 1000},
            "learn_lookup": {"value": True},
            "lookup_n_train_samples": {"values": [1000, 5000, 10000]},
        }
    else:
        num_epochs = 500
        c = {
            "batch_size": {"values": [16]},
            "num_epochs": {"values": [num_epochs]},
            "memory_size": {"min": 246, "max": 10000},
            "learn_lookup": {"value": False},
            "lookup_n_train_samples": {"value": None}
        }

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "score"},
        "parameters": {
            "dataset_to_use": {"value": "data/stimuli_idioms_clean.csv"},
            "kwics_file_to_use": {"value": "data/stimuli_idioms_kwics.json"},
            "embedding_model": {"value": "sbert"},
            "do_concat_tokens": {"value": True},
            "avg_last_n_layers": {"value": 1},
            "num_participants": {"value": 1},
            "num_workers": {"value": 1},
            "loss": {"value": "cosine"},
            # "forget_prob": {"values": [0.0, 0.2, 0.4, 0.6, 0.8]},
            "forget_prob": {"min": 0.0, "max": 0.8},
            # "minerva_k": {"values": [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]},
            "minerva_k": {"min": 0.95, "max": 0.995},
            # "hidden_size": {"values": [50, 100, 200, 300, 400, 500, 600, 700, 768]},
            "hidden_size": {"min": 50, "max": 768},
            **c
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10,
            "max_iter": num_epochs,
            "s": 3, # Number of brackets
            "eta": 3,  # Reduction factor
        },
    }

    if sweep_id is None:
        # 3: Start the sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="hopfield-experiments")

    wandb.agent(sweep_id, function=run)


if __name__ == "__main__":
    wandb.login()
    run_sweep()
