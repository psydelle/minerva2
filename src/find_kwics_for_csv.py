import os
import time
import requests as r
import pandas as pd
import math
import click
from dotenv import load_dotenv
from tqdm import tqdm
import json
import re
from tqdm.contrib.concurrent import thread_map, process_map
from se_utils import get_vn_kwics

load_dotenv()  # take environment variables from .env.

CORPUS_EN = "preloaded/ententen21_tt31"


# def get_kwics(verb, noun, corpus_name):
#     data = {
#         "corpname": corpus_name,
#         "q": f'q[lempos_lc="{verb}-v"][]?[lempos_lc="{noun}-n"] within <s />',
#         "concordance_query[queryselector]": "iqueryrow",
#         "concordance_query[iquery]": f'q[lempos_lc="{verb}-v"][]?[lempos_lc="{noun}-n"] within <s />',
#         "default_attr": "lemma",
#         "attr": "word",
#         # "refs": "=bncdoc.alltyp",
#         "attr_allpos": "all",
#         "cup_hl": "q",
#         "structs": "s,g",
#         "fromp": "1",
#         "pagesize": "150",
#         "kwicleftctx": "-2:s",  # 1 sentence of left context
#         "kwicrightctx": "2:s",  # 1 sentence of right context
#     }
#     try:
#         kwics_data = r.get(
#             base_url + "/concordance",
#             params=data,
#             auth=(USERNAME, API_KEY),
#         )
#         kwics_data.raise_for_status()  # Check for any request errors
#     except r.exceptions.RequestException as e:
#         print("Error occurred during the request:", str(e))
#         # Handle the error here

#     kwics_data = kwics_data.json()

#     lines = kwics_data["Lines"]
#     clean_lines = []
#     kwic_words = []
#     for line in lines:
#         left = [
#             x["str"] for x in line["Left"] if "str" in x
#         ]  # filter "strc", like sentence breaks
#         kwic = [x["str"] for x in line["Kwic"] if "str" in x]
#         right = [x["str"] for x in line["Right"] if "str" in x]

#         _ss = "</s><s>"

#         # if _ss in left:
#         #     left.reverse()
#         #     left_start = left.index(_ss)
#         #     left_start = len(left) - left_start - 1
#         #     left.reverse()
#         # else:
#         #     left_start = 0
#         # right_end = right.index(_ss) if _ss in right else -1
#         assert _ss not in kwic
#         kwic = [re.sub("[^\\w]", "", s) for s in kwic]  # strip all non-alnum

#         # left_clean = left[left_start + 1 :]
#         # right_clean = right[:right_end]
#         left_clean = left
#         right_clean = right
#         full_clean = left_clean + kwic + right_clean
#         if len(full_clean) > 100:
#             # skip if more than 100 words
#             continue
#         clean_line = " ".join(full_clean)
#         clean_lines.append(clean_line)
#         kwic_words.append((kwic[0], kwic[-1]))

#     # keep first 100 kwics under 100 words
#     if len(clean_lines) < 100:
#         raise RuntimeError("Only", len(clean_lines), "KWICs obtained for", verb, noun)
#     else:
#         clean_lines = clean_lines[:100]

#     return clean_lines, kwic_words


def process_item(line):
    data = {}
    kwics_sentences, kwic_words = get_vn_kwics(
        CORPUS_EN,
        line["verb"],
        line["noun"],
        n_kwics=100,
        n_ctx_sentences=1,
        max_word_count=100,
    )
    data["verb"] = line["verb"]
    data["noun"] = line["noun"]
    data["kwics"], data["kwic_words"] = kwics_sentences, kwic_words
    return data


def process_line(line):
    p = process_item(line)
    if not p["kwics"]:
        print("No KWICS found for", line["item"])
    time.sleep(1)  # SketchEngine timeout mitigation:(
    return (line["item"], p)


@click.command()
@click.option("-f", "--csv-file", required=True)
@click.option("-o", "--out-file", required=True)
def process_corpus(csv_file, out_file):
    in_df = pd.read_csv(csv_file)

    # sadness
    data = [process_line(line) for line in tqdm(in_df.iloc, total=len(in_df))]

    data = dict(data)

    with open(out_file, "w") as f:
        json.dump(data, f, ensure_ascii=True)


if __name__ == "__main__":
    process_corpus()