import argparse
from pathlib import Path

import pandas as pd
from run_one_experiment import run_experiment

# Run all experiments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_to_use", help="Dataset to use", default="data/stimuli.csv")
    parser.add_argument(
        "-k",
        "--kwics_file_to_use",
        help="Kwics complement to dataset to use, or 'none' to use no kwics",
        default="data/stimuli_kwics.json",
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
        help="Path to pickle file containing pt translations of "
        "en embeddings (required for aligned space_lang)",
        default=None,
    )
    parser.add_argument(
        "--freq_fraction_pt",
        help="Use this fraction of PT in frequency mixing "
        "(only applicable for mixed? aligned? space_lang, default 0.6)",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--minerva_k",
        help="Minerva k (threshold) parameter",
        default=0.99,
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
    args = parser.parse_args()

    results_dfs = []
    for space_lang in ["en", "pt", "en_noise", "pt_noise"]:
        for frequency_lang in ["en", "pt", "mix", "equal"]:
            print(
                f"Running experiment with space_lang={space_lang} and frequency_lang={frequency_lang}"
            )
            results_df = run_experiment(
                dataset_to_use=args.dataset_to_use,
                kwics_file_to_use=args.kwics_file_to_use,
                space_lang=space_lang,
                frequency_lang=frequency_lang,
                num_participants=args.num_participants,
                en_pt_trans_pickle=args.en_pt_trans_pickle,
                freq_fraction_pt=args.freq_fraction_pt,
                minerva_k=args.minerva_k,
                minerva_max_iter=args.minerva_max_iter,
                num_workers=args.num_workers,
                concat_tokens=args.concat_tokens,
                avg_last_n_layers=args.avg_last_n_layers,
                label=args.label,
            )

            results_dfs.append(results_df)

    results_df = pd.concat(results_dfs)
    out_file = f"results/combo_results-{Path(args.dataset_to_use).name[:-4]}-{args.num_participants}p-{f'-mix{args.freq_fraction_pt}'}-last_{args.avg_last_n_layers}-{'nokwics' if args.kwics_file_to_use=='none' else 'kwics'}{'-concat' if args.concat_tokens else ''}-m2k_{args.minerva_k}-m2mi_{args.minerva_max_iter}{'-' + args.label if args.label else ''}.csv"
    results_df.to_csv(out_file, index=False)
    print("Wrote results to", out_file)
