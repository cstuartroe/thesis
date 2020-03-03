import pandas as pd

import os

language_info_filename = "csv/language_info.csv"
SIGMORPHON_2018_results_filename = "csv/SIGMORPHON_2018_results.csv"
SIGMORPHON_2019_results_filename = "csv/SIGMORPHON_2019_results.csv"

language_info = pd.read_csv(language_info_filename)
SIGMORPHON_2018_results = pd.read_csv(SIGMORPHON_2018_results_filename, index_col="Language")
SIGMORPHON_2019_results = pd.read_csv(SIGMORPHON_2019_results_filename)


def get_language_by_name(name):
    return language_info[language_info["Name"] == name].iloc[0]


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
