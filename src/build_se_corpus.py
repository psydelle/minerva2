import os
import requests as r
import pandas as pd
import math
import click
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

API_KEY = os.environ.get("SE_API_KEY")
USERNAME = os.environ.get("SE_USERNAME")
base_url = "https://api.sketchengine.eu/bonito/run.cgi"

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

verb_list = ["kick", "spend"]

CORPUS_NAME = "preloaded/ententen21_tt31"
N_QUERY_NOUNS = 100

def get_ws(word, pos="-v"):
    data = {
        "corpname": CORPUS_NAME,
        "format": "json",
        "lemma": word,
        "lpos": pos,
        "maxitems": N_QUERY_NOUNS,
        "structured": 1,
    }
    sketch_data = r.get(
        base_url + "/wsketch?corpname=%s" % data["corpname"],
        params=data,
        auth=(USERNAME, API_KEY),
    ).json()
    return sketch_data


def mi(freq_xy, freq_x, freq_y, N):
    return math.log2(2 * freq_xy * N / (freq_x * freq_y))

def get_corp_info(corpus_name):
    data = {
        "corpname": corpus_name
    }
    corp_data = r.get(
        "https://api.sketchengine.eu/search/corp_info",
        params=data,
        auth=(USERNAME, API_KEY),
    ).json()
    return corp_data

def process_verb(verb, corp_info):
    sketch_data = get_ws(verb, pos="-v")

    for rel in sketch_data["Gramrels"]:
        if rel["name"] == 'objects of "%w"':
            all_coll_data = rel["Words"]

    page = 0
    PAGE_N = 25
    data = []
    selected_indices = {}
    while page * PAGE_N < N_QUERY_NOUNS:
        while True:
            for i, coll_data in enumerate(all_coll_data[page * PAGE_N : (page + 1) * PAGE_N]):
                index = i + page * PAGE_N
                print(
                    f'{index}{selected_indices.get(index, "")}: {(verb + " " + coll_data["word"]).ljust(25, " ")} - Score: {coll_data["score"]}, {coll_data["count"]}'
                )

            print("Type c <N>, f <N>, i <N>, d for done, n for next page, p for prev page: ")
            inp = input()
            if inp == "d":
                # return
                return data
            if inp == "n":
                page = min(N_QUERY_NOUNS/PAGE_N - 1, page+1)
                break
            if inp == "p":
                page = max(page-1, 0)
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

            item_data["item"] = verb + " " + coll_data["word"]
            item_data["length"] = len(verb + coll_data["word"])
            item_data["node"] = verb
            item_data["fnode"] = sketch_data["freq"]
            item_data["coll"] = coll_data["word"]
            item_data["fitem"] = coll_data["count"]
            item_data["logDice"] = coll_data["score"]

            # get frequency stats
            noun_stats = get_ws(coll_data["word"], "-n")
            item_data["fcoll"] = noun_stats["freq"]
            item_data["MI"] = mi(item_data["fitem"], item_data["fnode"], item_data["fcoll"], N=int(corp_info["sizes"]["wordcount"]))

            if command == "c":
                # collocation
                item_data["itemType"] = "coll"
            elif command == "f":
                item_data["itemType"] = "fc"
            elif command == "i":
                item_data["itemType"] = "idiom"
            else:
                print("Invalid command!")
            
            selected_indices[index] = command
            data.append(item_data)


@click.command()
@click.option("-o", "--out-file", required=True)
def process_corpus(out_file):
    corp_info = get_corp_info(CORPUS_NAME)

    data = []
    for verb in verb_list:
        data.extend(process_verb(verb, corp_info))

    df = pd.DataFrame(data)
    df.to_csv(out_file, index=True)

if __name__ == "__main__":
    process_corpus()