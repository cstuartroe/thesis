import pandas as pd

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