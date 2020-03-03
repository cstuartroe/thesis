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

    pair_dirs = next(os.walk("sigmorphon2019/task1"))[1]

    source_dirs = [dirname for dirname in pair_dirs if dirname.startswith(lang_name_formatted)]
    target_dirs = [dirname for dirname in pair_dirs if dirname.endswith(lang_name_formatted)]

    if len(source_dirs) > 0:
        return f"sigmorphon2019/task1/{source_dirs[0]}/{lang_name_formatted}-train-high"

    elif len(target_dirs) > 0:
        return f"sigmorphon2019/task1/{target_dirs[0]}/{lang_name_formatted}-train-low"

    else:
        raise ValueError(f"No file for language: {lang_name}")


def get_language_training_data(lang_name):
    training_file = get_language_training_file(lang_name)
    with open(training_file) as fh:
        lines = fh.readlines()

    data = []
    for line in lines:
        lemma, inflected_form, unimorph_tags = line.replace("\n", "").split("\t")
        data.append({"lemma":lemma, "inflected_form": inflected_form, "unimorph_tags": unimorph_tags})

    return data
