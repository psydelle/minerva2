import os
import re
import time
from dotenv import load_dotenv
import requests as r

load_dotenv()  # take environment variables from .env.

API_KEY = os.environ.get("SE_API_KEY")
USERNAME = os.environ.get("SE_USERNAME")
base_url = "https://api.sketchengine.eu/bonito/run.cgi"


def _get_request(url, params):
    try:
        response = r.get(url, params=params, auth=(USERNAME, API_KEY))
        response.raise_for_status()  # Check for any request errors
    except r.exceptions.RequestException as e:
        print("Error occurred during the request:", str(e))
    return response.json()

def get_ws(corpus_name, word, pos="-v", max_items=100):
    data = {
        "corpname": corpus_name,
        "format": "json",
        "lemma": word,
        "lpos": pos,
        "maxitems": max_items,
        "structured": 1,
        "minfreq": 1
    }
    sketch_data = _get_request(base_url + "/wsketch?corpname=%s" % data["corpname"], data)
    return sketch_data


def get_verb_noun_sketch_seek_id(corpus_name, verb, noun):
    # get the wordsketch seek id
    sketch_data = get_ws(corpus_name, verb, pos="-v")  # get verb sketch
    for rel in sketch_data["Gramrels"]:
        if rel["name"] == 'objects of "%w"':  # get the objects of the verb
            coll_nouns = rel["Words"]
            # get the seek id of the noun we want
            for noun_data in coll_nouns:
                if noun_data["word"] == noun:
                    return noun_data["seek"], sketch_data
    raise ValueError(f"Could not find seek id for {verb} and {noun}")


def get_corp_info(corpus_name):
    data = {"corpname": corpus_name}
    corp_data = _get_request("https://api.sketchengine.eu/search/corp_info", data)
    return corp_data


def get_vn_kwics(
    corpus_name,
    verb=None,
    noun=None,
    wordsketch_seek_id=None,
    n_kwics=10,
    max_word_count=100,
    n_ctx_sentences=1,
):
    try:
        if wordsketch_seek_id is None:
            wordsketch_seek_id, _sketch_data = get_verb_noun_sketch_seek_id(corpus_name, verb, noun)
        query = f"w{wordsketch_seek_id} within <s />"
        fallback = False
    except ValueError:
        print(f"Could not find seek id for {verb} and {noun}, falling back to default")
        query = f'q[lempos_lc="{verb}-v"][]?[lempos_lc="{noun}-n"] within <s />'
        fallback = True

    def get_data():
        for page in range(1, 3):
            data = {
                "corpname": corpus_name,
                "q": query,
                "concordance_query[queryselector]": "iqueryrow",
                "concordance_query[iquery]": query,
                "default_attr": "lemma",
                "attr": "word",
                # "refs": "=bncdoc.alltyp",
                "attr_allpos": "all",
                "cup_hl": "q",
                "structs": "s,g",
                "fromp": page,
                # get more than we need, to filter out long ones. TODO: requery on demand
                "pagesize": n_kwics * 2,
                "kwicleftctx": f"-{n_ctx_sentences+1}:s",  # num sentences of left context
                "kwicrightctx": f"{n_ctx_sentences+1}:s",  # num sentences of right context
            }
            kwics_data = _get_request(base_url + "/concordance", data)
            lines = kwics_data["Lines"]

            yield from lines


    clean_lines = []
    kwic_words = []
    for line in get_data():
        # filter "strc", like sentence breaks
        left = [x["str"] for x in line["Left"] if "str" in x]
        right = [x["str"] for x in line["Right"] if "str" in x]

        if not fallback:
            assert len(line["Kwic"]) == 1, f"Expected only one kwic, got {line['Kwic']}"
            kwic = line["Kwic"][0]["str"]
            # grab colocate from right context
            # it will be the first element which has the "coll": 1 key
            kwic2 = next((x["str"] for x in line["Right"] if "coll" in x), None)
            # if kwic2 is None:
            #     kwic2 = next((x["str"] for x in line["Left"] if "coll" in x), None)
            if kwic2 is None:
                continue # skip if collocate not in right context (inverted order)

            if not kwic.isalnum() or not kwic2.isalnum():
                # skip if not alnum
                continue

            full_clean = left + [kwic] + right
        else:
            kwic_str = [x["str"] for x in line["Kwic"] if "str" in x]
            if not all([x.isalnum() for x in kwic_str]):
                # skip if not alnum
                continue
            kwic, kwic2 = kwic_str[0], kwic_str[-1]
            full_clean = left + kwic_str + right

        if len(full_clean) > max_word_count:
            # skip if too many words
            continue

        if kwic in left:
            # more than one instance of the verb appears in the left context
            # filter to avoid doing character vs. token offset math later
            continue

        clean_line = " ".join(full_clean)
        clean_lines.append(clean_line)
        kwic_words.append((kwic, kwic2))
        if len(clean_lines) >= n_kwics:
            break

    # keep first N kwics under max_word_count
    if len(clean_lines) != n_kwics:
        raise RuntimeError("Only", len(clean_lines), "KWICs obtained for", verb, noun)

    return clean_lines, kwic_words
