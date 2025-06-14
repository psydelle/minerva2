# Import the W&B Python Library
import random
import numpy as np
import torch
import wandb
from datetime import datetime
from run_hopfield_experiments import run_experiment

seed = 0

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# 1: Define objective/training function
def run(**config):
    results_df = run_experiment(
        dataset_to_use="data/stimuli_idioms_clean.csv",
        kwics_file_to_use="data/stimuli_idioms_kwics.json",
        # num_participants=,
        embedding_model="sbert",
        # forget_prob=args.forget_prob,
        # do_noise_embeddings=args.do_noise_embeddings,
        # do_equal_frequency=args.do_equal_frequency,
        # do_log_freq=args.do_log_freq,
        # minerva_k=args.minerva_k,
        do_concat_tokens=True,
        avg_last_n_layers=1,
        no_individual_wandb_runs=True,  # because sweeping
        # num_epochs=args.num_epochs,
        # hidden_size=args.hidden_size,
        # batch_size=args.batch_size,
        # label=args.label,
        # beta=args.beta,
        # memory_size=args.memory_size,
        # learn_lookup=args.learn_lookup,
        # lookup_n_train_samples=args.lookup_n_train_samples,
        **config
    )
    return results_df["score"].mean()

def main():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb_group_name = f"hopfield-{current_time}"
    wandb.init(project="hopfield-experiments", group=wandb_group_name)
    score = run(wandb_group_name=wandb_group_name, **wandb.config)
    wandb.log({"score": score})


# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        # "x": {"max": 0.1, "min": 0.01},
        # "y": {"values": [1, 3, 7]},
        "num_participants": {"values": [1]},
        "num_workers": {"values": [1]},
        "forget_prob": {"max": 0.8, "min": 0.0},
        "minerva_k": {"values": [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]},
        "num_epochs": {"values": [100]},
        "hidden_size": {"values": [50, 100, 200, 300, 400, 500, 600, 700, 768]},
        "batch_size": {"values": [64]},
        "memory_size": {"max": 1000, "min": 3},
        "learn_lookup": {"values": [True]},
        "lookup_n_train_samples": {"values": [1000, 5000, 10000]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="hopfield-experiments")

wandb.agent(sweep_id, function=main, count=2)
