# Import the W&B Python Library
import random
import numpy as np
import torch
import wandb
from datetime import datetime
from run_hopfield_experiments import run_experiment_sweep_wrapper

seed = 0

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# 1: Define objective/training function
def run():
    results_df = run_experiment_sweep_wrapper()
    # return results_df["score"].mean()


# 2: Define the search space

num_epochs = 100

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "dataset_to_use": {"value": "data/stimuli_idioms_clean.csv"},
        "kwics_file_to_use": {"value": "data/stimuli_idioms_kwics.json"},
        "embedding_model": {"value": "sbert"},
        "do_concat_tokens": {"value": True},
        "avg_last_n_layers": {"value": 1},
        "num_participants": {"value": 1},
        "num_workers": {"value": 1},
        "forget_prob": {"values": [0.0, 0.2, 0.4, 0.6, 0.8]},
        "minerva_k": {"values": [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]},
        "num_epochs": {"values": [num_epochs]},
        "hidden_size": {"values": [50, 100, 200, 300, 400, 500, 600, 700, 768]},
        "batch_size": {"values": [64]},
        "memory_size": {"max": 1000, "min": 3},
        "learn_lookup": {"values": [True]},
        "lookup_n_train_samples": {"values": [1000, 5000, 10000]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 1,
        # "max_iter": num_epochs,
        # "s": 5 # Number of brackets
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="hopfield-experiments")

wandb.agent(sweep_id, function=run, count=2)
