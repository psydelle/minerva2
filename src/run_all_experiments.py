import argparse
from itertools import product
from pathlib import Path

import pandas as pd
from run_one_experiment import run_experiment

# Run all experiments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_to_use", help="Dataset to use", default="data/stimuli_idioms_clean.csv")
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
    args = parser.parse_args()

    results_dfs = []
    for embedding_model, forget_prob, minerva_k, do_noise_embeddings, do_equal_frequency in product(
        # ["sbert", "fasttext"],
        ["sbert"],
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [0.95, 0.96, 0.97, 0.98, 0.99],
        [False, True],
        [False, True],
    ):
        print(
            f"Running experiment with model {embedding_model}" +
            (' with noise' if do_noise_embeddings else '') +
            (' with equal frequency' if do_equal_frequency else '')
        )
        results_df = run_experiment(
            dataset_to_use=args.dataset_to_use,
            kwics_file_to_use=args.kwics_file_to_use,
            num_participants=args.num_participants,
            embedding_model=embedding_model,
            do_noise_embeddings=do_noise_embeddings,
            do_equal_frequency=do_equal_frequency,
            minerva_k=minerva_k,
            minerva_max_iter=args.minerva_max_iter,
            num_workers=args.num_workers,
            do_concat_tokens=args.concat_tokens,
            avg_last_n_layers=args.avg_last_n_layers,
            label=args.label,
            forget_prob=forget_prob,
        )

        results_dfs.append(results_df)

    results_df = pd.concat(results_dfs)
    out_file = f"results/combo_results-{Path(args.dataset_to_use).name[:-4]}-{args.num_participants}p-last_{args.avg_last_n_layers}-{'nokwics' if args.kwics_file_to_use=='none' else 'kwics'}{'-concat' if args.concat_tokens else ''}-m2k_{args.minerva_k}-m2mi_{args.minerva_max_iter}{'-' + args.label if args.label else ''}.csv"
    results_df.to_csv(out_file, index=False)
    print("Wrote results to", out_file)
