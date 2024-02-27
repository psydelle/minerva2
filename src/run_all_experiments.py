import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
from run_one_experiment import run_experiment


def _make_filename(**kwargs):
    return (
        f"results-{Path(args.dataset_to_use).name[:-4]}-"
        f"{kwargs['num_participants']}p-last_{kwargs['avg_last_n_layers']}-"
        f"{'nokwics' if kwargs['kwics_file_to_use']=='none' else 'kwics'}"
        f"{'-logfreq' if kwargs['do_log_freq'] else ''}"
        f"{'-concat' if kwargs['concat_tokens'] else ''}"
        f"-{kwargs['embedding_model']}-fp_{kwargs['forget_prob']}-m2k_{kwargs['minerva_k']}-m2mi_{kwargs['minerva_max_iter']}"
        f"{'-noise' if kwargs.get('do_noise_embeddings') else ''}"
        f"{'-equalfreq' if kwargs.get('do_equal_frequency') else ''}"
        f"{'-' + kwargs['label'] if kwargs['label'] else ''}.csv"
    )


# Run all experiments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_to_use",
        help="Dataset to use",
        default="data/stimuli_idioms_clean.csv",
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
        "--do_log_freq",
        help="Log-transform the frequency data before sampling",
        action="store_true",
        default=False,
    ),
    parser.add_argument(
        "--minerva_k",
        help="Minerva k (threshold) parameter",
        default=0.95,
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
        "--resume_from_dir",
        help="Directory to resume from",
        default=None,
    )
    args = parser.parse_args()

    results_dfs = []
    results_dir = Path("results")
    if args.resume_from_dir:
        checkpoint_dir = Path(args.resume_from_dir)
        assert checkpoint_dir.exists(), f"{checkpoint_dir} does not exist"
        print(f"Resuming from {checkpoint_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = (
            results_dir
            / "_checkpoints"
            / f"{args.label+'_' if args.label else ''}{timestamp}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=False)

    for (
        embedding_model,
        forget_prob,
        minerva_k,
        do_noise_embeddings,
        do_equal_frequency,
    ) in product(
        ["sbert", "fasttext"],
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [0.95, 0.96, 0.97, 0.98, 0.99, 0.995],
        [False, True],
        [False, True],
    ):
        results_file = checkpoint_dir / _make_filename(
            num_participants=args.num_participants,
            concat_tokens=args.concat_tokens,
            avg_last_n_layers=args.avg_last_n_layers,
            kwics_file_to_use=args.kwics_file_to_use,
            do_log_freq=args.do_log_freq,
            embedding_model=embedding_model,
            forget_prob=forget_prob,
            minerva_k=minerva_k,
            minerva_max_iter=args.minerva_max_iter,
            do_noise_embeddings=do_noise_embeddings,
            do_equal_frequency=do_equal_frequency,
            label=args.label,
        )
        if results_file.exists():
            print(f"Skipping {results_file}, already exists")
            results_df = pd.read_csv(results_file)
            results_dfs.append(results_df)
            continue

        print(
            f"Running experiment with model {embedding_model}"
            + (" with noise" if do_noise_embeddings else "")
            + (" with equal frequency" if do_equal_frequency else "")
            + f" and forget probability {forget_prob}"
            + f" and minerva k {minerva_k}"
        )
        results_df = run_experiment(
            dataset_to_use=args.dataset_to_use,
            kwics_file_to_use=args.kwics_file_to_use,
            num_participants=args.num_participants,
            embedding_model=embedding_model,
            do_noise_embeddings=do_noise_embeddings,
            do_equal_frequency=do_equal_frequency,
            do_log_freq=args.do_log_freq,
            minerva_k=minerva_k,
            minerva_max_iter=args.minerva_max_iter,
            num_workers=args.num_workers,
            do_concat_tokens=args.concat_tokens,
            avg_last_n_layers=args.avg_last_n_layers,
            label=args.label,
            forget_prob=forget_prob,
        )
        results_df.to_csv(
            results_file,
            index=False,
        )
        results_dfs.append(results_df)

    final_results_df = pd.concat(results_dfs)
    out_file = results_dir / _make_filename(
        num_participants=args.num_participants,
        avg_last_n_layers=args.avg_last_n_layers,
        concat_tokens=args.concat_tokens,
        kwics_file_to_use=args.kwics_file_to_use,
        do_log_freq=args.do_log_freq,
        embedding_model="all",
        forget_prob="0.0-0.8",
        minerva_k="0.95-0.995",
        minerva_max_iter=args.minerva_max_iter,
        label=args.label,
    )
    final_results_df.to_csv(out_file, index=False)
    print("Wrote results to", out_file)
