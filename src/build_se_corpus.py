import json
import os
from pathlib import Path
import requests as r
import pandas as pd
import math
import click

from se_utils import get_ws, get_corp_info, get_vn_kwics

# df = pd.DataFrame(
#     columns=[
#         "trialType",
#         "item",
#         "item_pt",
#         "length",
#         # "syllables",
#         "node",
#         "fnode",
#         # "knode",
#         "coll",
#         "fcoll",
#         # "kcoll",
#         "fitem",
#         "logDice",
#         "MI",
#         # "k",
#         "kwic",
#         "fitempt",
#         "nodept",
#         "fnodept",
#         "collpt",
#         "fcollpt",
#         "kwic_pt",
#     ]
# )


CORPUS_NAME = "preloaded/ententen21_tt31"
N_QUERY_NOUNS = 100


def mi(freq_xy, freq_x, freq_y, N):
    return math.log2(2 * freq_xy * N / (freq_x * freq_y))


def process_verb(verb, corp_info):
    sketch_data = get_ws(CORPUS_NAME, verb, pos="-v", max_items=N_QUERY_NOUNS)

    for rel in sketch_data["Gramrels"]:
        if rel["name"] == 'objects of "%w"':
            all_coll_data = rel["Words"]

    page = 0
    PAGE_N = 25
    data = []
    selected_indices = {}
    while page * PAGE_N < N_QUERY_NOUNS:
        while True:
            for i, coll_data in enumerate(
                all_coll_data[page * PAGE_N : (page + 1) * PAGE_N]
            ):
                index = i + page * PAGE_N
                print(
                    f'{index}{selected_indices.get(index, "")}: {(verb + " " + coll_data["word"]).ljust(25, " ")} - Score: {coll_data["score"]}, {coll_data["count"]}'
                )

            print(
                "Type c <N>, f <N>, i <N>, d to move to next verb, n for next page, p for prev page, exit to exit: "
            )
            inp = input()
            if inp == "exit":
                # return
                if data:
                    print("Unwritten data! Write d to save.")
                    continue
                return None
            if inp == "d":
                # return
                return data
            if inp == "n":
                page = min(N_QUERY_NOUNS / PAGE_N - 1, page + 1)
                break
            if inp == "p":
                page = max(page - 1, 0)
                break

            try:
                command, index = inp.split(" ")
                index = int(index)
            except ValueError as e:
                print(e)
                continue
            if index in selected_indices:
                print(f"Index already selected for {selected_indices[index]}!")
                continue

            coll_data = all_coll_data[index]

            item_data = {}

            if command == "c":
                # collocation
                item_data["itemType"] = "coll"
            elif command == "f":
                item_data["itemType"] = "fc"
            elif command == "i":
                item_data["itemType"] = "idiom"
            else:
                print("Invalid command!")

            item_data["item"] = verb + " " + coll_data["word"]
            item_data["length"] = len(verb + coll_data["word"])
            item_data["node"] = verb
            item_data["fnode"] = sketch_data["freq"]
            item_data["coll"] = coll_data["word"]
            item_data["fitem"] = coll_data["count"]
            item_data["logDice"] = coll_data["score"]

            # get frequency stats
            noun_stats = get_ws(
                CORPUS_NAME, coll_data["word"], "-n", max_items=N_QUERY_NOUNS
            )
            item_data["fcoll"] = noun_stats["freq"]
            item_data["MI"] = mi(
                item_data["fitem"],
                item_data["fnode"],
                item_data["fcoll"],
                N=int(corp_info["sizes"]["wordcount"]),
            )
            # kwics, kwic_words = get_vn_kwics(CORPUS_NAME, verb, coll_data["word"])
            kwics, kwic_words = get_vn_kwics(
                CORPUS_NAME,
                ws_seek_id=coll_data["seek"],
                n_kwics=100,
                n_ctx_sentences=1,
            )

            k = dict()
            k["kwics"] = kwics
            k["kwic_words"] = kwic_words

            selected_indices[index] = command
            data.append((item_data["item"], item_data, k))


@click.command()
@click.option(
    "-o",
    "--out_file",
    required=True,
    help="Path to which to write CSV. WILL OVERWRITE EXISTING.",
)
@click.option(
    "--do_append/--no_append",
    default=False,
    help="Append to out_file instead of overwriting it.",
)
def process_corpus(out_file, do_append):
    if not out_file.endswith(".csv"):
        raise ValueError("Out file must be a CSV!")

    corp_info = get_corp_info(CORPUS_NAME)

    ## read in the dataset
    stimuli_df = pd.read_csv("data/stimuli.csv")
    verb_list = stimuli_df["node"].unique()

    kwics_json_path = "".join(out_file.split(".")[:-1]) + "_kwics.json"

    if do_append:
        prev_data_df = pd.read_csv(out_file)
        with open(kwics_json_path) as f:
            prev_data_kwics = json.load(f)
        existing_verbs = prev_data_df["node"].unique()
        verb_list = set(verb_list) - set(existing_verbs)

    verb_list = sorted(verb_list)

    data = []
    kwics = {}
    for verb in verb_list:
        p = process_verb(verb, corp_info)
        if p is not None:
            for item, item_data, item_kwics in p:
                data.append(item_data)
                kwics[item] = item_kwics
        else:
            break

    data_df = pd.DataFrame(data)

    if do_append:
        data_df = pd.concat([prev_data_df, data_df], axis="index", ignore_index=True)
        kwics = {**prev_data_kwics, **kwics}

    data_df.to_csv(out_file, index=False)

    with open(kwics_json_path, "w") as f:
        json.dump(kwics, f)


if __name__ == "__main__":
    process_corpus()
