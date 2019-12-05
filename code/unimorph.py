import os
import json

from .general import *

POS = ["N", "V", "ADJ", "DET"]
SPECIAL_VERBS = ["V.CVB", "V.PCTP", "V.PTCP", "V.MSDR"]
DET = ["DEF", "INDF"]
PRS = ["1", "2", "3"]


def get_language_training_file(lang_name):
    lang_name_formatted = lang_name.replace(" ", "-").lower()

    pair_dirs = next(os.walk("sigmorphon2019/task1"))[1]

    source_dirs = [dirname for dirname in pair_dirs if dirname.startswith(lang_name_formatted)]
    target_dirs = [dirname for dirname in pair_dirs if dirname.endswith(lang_name_formatted)]

    if len(source_dirs) > 0:
        return f"sigmorphon2019/task1/{source_dirs[0]}/{lang_name_formatted}-train-high"

    elif len(target_dirs) > 0:
        return f"sigmorphon2019/task1/{target_dirs[0]}/{lang_name_formatted}-train-low"

    else:
        raise ValueError(f"No file for language: {lang_name}")


def language_unimorph_sets(lang_name):
    d = {}
    for pos in POS + SPECIAL_VERBS:
        d[pos] = set()

    training_file = get_language_training_file(lang_name)
    with open(training_file) as fh:
        lines = fh.readlines()

    for line in lines:
        lemma, inflected_form, unimorph_tags = line.replace("\n", "").split("\t")

        if ";" in unimorph_tags:
            tags = unimorph_tags.split(";")
        else:
            tags = [unimorph_tags]

        if tags[0] in POS:
            d[tags[0]] = d[tags[0]] | set(tags[1:])

        elif tags[0] in SPECIAL_VERBS:
            d["V"] = d["V"] | set(tags)

        elif tags[0] in DET:
            d["DET"] = d["DET"] | set(tags)

        else:  # this condition only fails to be met for Kurmanji and Sorani, idk why they're formatted differently
            tags = set(tags)
            if "V" in tags:
                tags.remove("V")
                d["V"] = d["V"] | tags
            else:
                pass  # I just give up at this point

    return d


def pull():
    for pos in POS:
        if f"{pos} categories" not in language_info.columns:
            language_info[f"{pos} categories"] = ""

    for row in language_info.iterrows():
        lus = language_unimorph_sets(row[1]["Name"])
        for pos in POS:
            row[1][f"{pos} categories"] = ";".join(sorted(list(lus[pos])))

    language_info.to_csv(language_info_filename, index=False)
