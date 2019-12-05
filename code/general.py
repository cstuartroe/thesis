import pandas as pd

language_info_filename = "csv/language_info.csv"
SIGMORPHON_2019_results_filename = "csv/SIGMORPHON_2019_results.csv"

language_info = pd.read_csv(language_info_filename)
SIGMORPHON_2019_results = pd.read_csv(SIGMORPHON_2019_results_filename)
