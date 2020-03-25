import numpy as np

from .general import *
from .unimorph import POS
from .alignment import InflectionShapes

COLUMN_ORDER = [
    "Source Language",
    "Target Language",
    "Distance",
    "Baseline Accuracy",
    "Best Transfer Accuracy",
    "Accuracy Improvement vs Baseline",
    "Best 2018 Accuracy",
    "Accuracy Improvement vs 2018",
    "Best Accuracy Team",
    "Baseline Levenshtein",
    "Best Transfer Levenshtein",
    "Levenshtein Improvement",
    "Best Levenshtein Team",
    "POS distribution similarity"
] + [f"{pos} category overlap" for pos in POS] + [
    "Inflection shape similarity"
]


def calculate_distance():
    distances = []
    for row in SIGMORPHON_2019_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])
        if source["Language family"] == target["Language family"]:
            if source["Subfamily"] == target["Subfamily"]:
                distances.append("Closely related")
            else:
                distances.append("Distantly related")
        else:
            distances.append("Unrelated")

    SIGMORPHON_2019_results["Distance"] = distances


def calculate_category_overlap(pos):
    category_overlaps = []
    for row in SIGMORPHON_2019_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])

        source_categories_raw = source[f"{pos} categories"]
        target_categories_raw = target[f"{pos} categories"]

        # pandas empty cells default to NaN
        if type(source_categories_raw) == float:
            source_categories = set()
        else:
            source_categories = set(source_categories_raw.split(";"))
        if type(target_categories_raw) == float:
            target_categories = set()
        else:
            target_categories = set(target_categories_raw.split(";"))

        try:
            overlap = len(source_categories & target_categories)/len(source_categories | target_categories)
        except ZeroDivisionError:
            overlap = np.nan

        category_overlaps.append(round(overlap, 2))

    SIGMORPHON_2019_results[f"{pos} category overlap"] = category_overlaps


def calculate_improvements():
    global SIGMORPHON_2019_results

    SIGMORPHON_2019_results["Accuracy Improvement vs Baseline"] = SIGMORPHON_2019_results["Best Transfer Accuracy"] \
                                                                  - SIGMORPHON_2019_results["Baseline Accuracy"]
    SIGMORPHON_2019_results["Levenshtein Improvement"] = SIGMORPHON_2019_results["Best Transfer Levenshtein"] \
                                                         - SIGMORPHON_2019_results["Baseline Levenshtein"]

    accuracy_2018_list = []
    accuracy_improvement_vs_2018 = []
    for row in SIGMORPHON_2019_results.iterrows():
        accuracy_2018 = SIGMORPHON_2018_results.loc[row[1]["Target Language"]]["Low-resource accuracy"]
        accuracy_2018_list.append(accuracy_2018)
        accuracy_improvement_vs_2018.append(row[1]["Best Transfer Accuracy"] - accuracy_2018)
    SIGMORPHON_2019_results["Best 2018 Accuracy"] = accuracy_2018_list
    SIGMORPHON_2019_results["Accuracy Improvement vs 2018"] = accuracy_improvement_vs_2018

    SIGMORPHON_2019_results = SIGMORPHON_2019_results.round({
        "Accuracy Improvement vs Baseline": 1,
        "Accuracy Improvement vs 2018": 1,
        "Levenshtein Improvement": 2
    })


def calculate_proportions(d):
    total = sum(d.values())
    return dict([(key, value/total) for key, value in d.items()])


def calculate_POS_distribution_overlaps():
    POS_distribution_overlaps = []
    for row in SIGMORPHON_2019_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])

        source_POS = calculate_proportions(dict([(pos, source[f"{pos} examples"]) for pos in POS]))
        target_POS = calculate_proportions(dict([(pos, target[f"{pos} examples"]) for pos in POS]))

        distribution_overlap = round(sum(min(source_POS[pos], target_POS[pos]) for pos in POS), 3)
        POS_distribution_overlaps.append(distribution_overlap)

    SIGMORPHON_2019_results["POS distribution similarity"] = POS_distribution_overlaps


def calculate_inflection_shape_similarity():
    inflection_shape_similarities = []
    for row in SIGMORPHON_2019_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])

        source_prevs = [source[f"{shape} prevalence"] for shape in InflectionShapes._fields]
        target_prevs = [target[f"{shape} prevalence"] for shape in InflectionShapes._fields]

        inflection_shape_similarity = 1 - sum([abs(sp-tp) for sp, tp in zip(source_prevs, target_prevs)])/(sum(source_prevs) + sum(target_prevs))
        inflection_shape_similarities.append(round(inflection_shape_similarity, 3))

    SIGMORPHON_2019_results["Inflection shape similarity"] = inflection_shape_similarities


def calculate():
    calculate_distance()
    calculate_improvements()
    for pos in POS:
        calculate_category_overlap(pos)
    calculate_POS_distribution_overlaps()
    calculate_inflection_shape_similarity()
    SIGMORPHON_2019_results[COLUMN_ORDER].to_csv(SIGMORPHON_2019_results_filename, index=False)
