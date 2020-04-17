import pandas as pd

import os

language_info_filename = "csv/language_info.csv"
SIGMORPHON_2018_results_filename = "csv/SIGMORPHON_2018_results.csv"
SIGMORPHON_2019_results_filename = "csv/SIGMORPHON_2019_results.csv"
my_results_filename = "csv/my_results.csv"

language_info = pd.read_csv(language_info_filename)
# SIGMORPHON_2018_results = pd.read_csv(SIGMORPHON_2018_results_filename, index_col="Language")
# SIGMORPHON_2019_results = pd.read_csv(SIGMORPHON_2019_results_filename)
my_results = pd.read_csv(my_results_filename) if os.path.exists(my_results_filename) else pd.DataFrame()

ARCHES = [d.split("-")[1] for d in os.listdir("model")]


def get_language_by_name(name):
    rows = language_info[language_info["Name"] == name]
    if len(rows) == 1:
        return rows.iloc[0]
    else:
        raise KeyError(f"There are {len(rows)} language with name {name}")


def get_language_by_iso(iso_code):
    return language_info[language_info["ISO 639-2"] == iso_code].iloc[0]


def get_language_training_file(lang_name):
    lang_name_formatted = lang_name.replace(" ", "-").lower()

    levels = ["high", "medium", "low"]

    for level in levels:
        filename = f"conll2018/task1/all/{lang_name_formatted}-train-{level}"
        if os.path.exists(filename):
            return filename

    raise ValueError("No such language found: " + lang_name)


def get_language_training_data(lang_name):
    training_file = get_language_training_file(lang_name)
    with open(training_file) as fh:
        lines = fh.readlines()

    data = []
    for i, line in enumerate(lines):
        try:
            lemma, inflected_form, unimorph_tags = line.replace("\n", "").split("\t")
        except ValueError as e:
            print(lang_name, "line", i)
            raise e
        data.append({"lemma":lemma, "inflected_form": inflected_form, "unimorph_tags": unimorph_tags})

    return data


def flatten(l):
    return [item for sublist in l for item in sublist]
