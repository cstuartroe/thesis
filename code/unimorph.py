from tqdm import tqdm

from .general import *

POS = ["N", "V", "ADJ", "DET"]
SPECIAL_VERBS = ["V.CVB", "V.PCTP", "V.PTCP", "V.MSDR"]
DET = ["DEF", "INDF"]
PRS = ["1", "2", "3"]


def language_unimorph_sets(lang_name):
    d = {}
    pos_examples = {}
    for pos in POS:
        d[pos] = set()
        pos_examples[pos] = 0

    for triplet in get_language_training_data(lang_name):
        tags = triplet["unimorph_tags"].split(";")

        if tags[0] in POS:
            d[tags[0]] = d[tags[0]] | set(tags[1:])
            pos_examples[tags[0]] += 1

        elif tags[0] in SPECIAL_VERBS:
            d["V"] = d["V"] | set(tags)
            pos_examples["V"] += 1

        elif tags[0] in DET:
            d["DET"] = d["DET"] | set(tags)
            pos_examples["DET"] += 1

        else:  # this condition only fails to be met for Kurmanji and Sorani, idk why they're formatted differently
            tags = set(tags)
            if "V" in tags:
                tags.remove("V")
                d["V"] = d["V"] | tags
                pos_examples["V"] += 1
            else:
                pass  # I just give up at this point

    return d, pos_examples


def pull():
    for pos in POS:
        if f"{pos} categories" not in language_info.columns:
            language_info[f"{pos} categories"] = ""
        if f"{pos} examples" not in language_info.columns:
            language_info[f"{pos} examples"] = ""

    for index, row in tqdm(list(language_info.iterrows())):
        lus, pos_examples = language_unimorph_sets(row["Name"])
        for pos in POS:
            language_info.loc[index, f"{pos} categories"] = ";".join(sorted(list(lus[pos])))
            language_info.loc[index, f"{pos} examples"] = pos_examples[pos]

    language_info.to_csv(language_info_filename, index=False)
