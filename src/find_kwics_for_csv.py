import os
import requests as r
import pandas as pd
import math
import click
from dotenv import load_dotenv
from tqdm import tqdm
import json
import re

load_dotenv()  # take environment variables from .env.

API_KEY = os.environ.get("SE_API_KEY")
USERNAME = os.environ.get("SE_USERNAME")
base_url = "https://api.sketchengine.eu/bonito/run.cgi"

## read in the dataset
df = pd.read_csv("data/stimuli.csv")
verb_list = df["node"].unique()

CORPUS_EN = "preloaded/ententen21_tt31"
CORPUS_PT = "preloaded/pttenten20_fl5"


def get_kwics(verb, noun, corpus_name):
    data = {
        "corpname": corpus_name,
        "q": f'q[lempos_lc="{verb}-v"][]?[lempos_lc="{noun}-n"] within <s />',
        "concordance_query[queryselector]": "iqueryrow",
        "concordance_query[iquery]": f'q[lempos_lc="{verb}-v"][]?[lempos_lc="{noun}-n"] within <s />',
        "default_attr": "lemma",
        "attr": "word",
        # "refs": "=bncdoc.alltyp",
        "attr_allpos": "all",
        "cup_hl": "q",
        "structs": "s,g",
        "fromp": "1",
        "pagesize": "100",
        "kwicleftctx": "300#",
        "kwicrightctx": "300#",
    }
    kwics_data = r.get(
        base_url + "/concordance",
        params=data,
        auth=(USERNAME, API_KEY),
    ).json()
    lines = kwics_data["Lines"]
    clean_lines = []
    kwic_words = []
    for line in lines:
        left = [x.get("str", x.get("strc")) for x in line["Left"]]
        kwic = [x.get("str", x.get("strc")) for x in line["Kwic"]]
        right = [x.get("str", x.get("strc")) for x in line["Right"]]

        _ss = "</s><s>"

        if _ss in left:
            left.reverse()
            left_start = left.index(_ss)
            left_start = len(left) - left_start - 1
            left.reverse()
        else:
            left_start = 0
        right_end = right.index(_ss) if _ss in right else -1
        assert _ss not in kwic
        kwic = [re.sub("[^\\w]", "", s) for s in kwic] # strip all non-alnum

        left_clean = left[left_start + 1 :]
        right_clean = right[:right_end]
        clean_line = " ".join(left_clean + kwic + right_clean)
        clean_lines.append(clean_line)
        kwic_words.append((kwic[0], kwic[-1]))

    return clean_lines, kwic_words


def process_item(line):
    data = {}
    data["kwics"], data["kwic_words"] = get_kwics(line["node"], line["coll_lemma"], CORPUS_EN)
    data["kwics_pt"], data["kwic_words_pt"] = get_kwics(
        line["nodept"], line["collpt_lemma"], CORPUS_PT
    )
    return data


@click.command()
@click.option("-f", "--csv-file", required=True)
@click.option("-o", "--out-file", required=True)
def process_corpus(csv_file, out_file):
    in_df = pd.read_csv(csv_file)

    data = {}
    for line in tqdm(in_df.iloc, total=len(in_df)):
        if line["trialType"] == "baseline":
            p = {
                "kwics": [],
                "kwic_words": [],
                "kwics_pt": [],
                "kwic_words_pt": [],
            }
        else:
            p = process_item(line)
            if not p["kwics_pt"]:
                print("No Portuguese found for", line["item_pt"])
        data[line["item"]] = p

    with open(out_file, "w") as f:
        json.dump(data, f, ensure_ascii=True)


if __name__ == "__main__":
    process_corpus()
