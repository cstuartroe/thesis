from .general import *

COLUMN_ORDER = [
    "Source Language",
    "Target Language",
    "Distance",
    "Baseline Accuracy",
    "Best Transfer Accuracy",
    "Accuracy Improvement",
    "Best Accuracy Team",
    "Baseline Levenshtein",
    "Best Transfer Levenshtein",
    "Levenshtein Improvement",
    "Best Levenshtein Team"
]


def get_language(iso_or_name):
    langs = language_info[language_info["ISO 639-2"] == iso_or_name]
    if langs.empty:
        langs = language_info[language_info["Name"] == iso_or_name]

    return langs.iloc[0]


def calculate_distance():
    if "Distance" in SIGMORPHON_2019_results.columns:
        SIGMORPHON_2019_results.drop(columns=["Distance"])

    distances = []
    for row in SIGMORPHON_2019_results.iterrows():
        source = get_language(row[1]["Source Language"])
        target = get_language(row[1]["Target Language"])
        if source["Language family"] == target["Language family"]:
            if source["Subfamily"] == target["Subfamily"]:
                distances.append("Closely related")
            else:
                distances.append("Distantly related")
        else:
            distances.append("Unrelated")

    SIGMORPHON_2019_results["Distance"] = distances


def calculate_improvements():
    SIGMORPHON_2019_results["Accuracy Improvement"] = SIGMORPHON_2019_results["Best Transfer Accuracy"] - SIGMORPHON_2019_results["Baseline Accuracy"]
    SIGMORPHON_2019_results["Levenshtein Improvement"] = SIGMORPHON_2019_results["Best Transfer Levenshtein"] - SIGMORPHON_2019_results["Baseline Levenshtein"]


def calculate():
    calculate_distance()
    calculate_improvements()
    SIGMORPHON_2019_results[COLUMN_ORDER].to_csv(SIGMORPHON_2019_results_filename, index=False)
