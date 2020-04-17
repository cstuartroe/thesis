import numpy as np

from .general import *
from .unimorph import POS
from .alignment import InflectionShapes, levenshtein

COLUMN_ORDER = [
    "Source Language",
    "Target Language"] + flatten([[
    f"Baseline Accuracy ({arch})",
    f"Transfer Accuracy ({arch})",
    f"Accuracy Improvement ({arch})",
    f"Baseline Levenshtein ({arch})",
    f"Transfer Levenshtein ({arch})",
    f"Levenshtein Improvement ({arch})"
] for arch in ARCHES]) + [
    "POS distribution similarity",
    "Distance",
    "Inflection shape similarity"
] + [f"{pos} category overlap" for pos in POS]


def split_langs(dirname):
    l = dirname.split("-")
    for i in range(len(l)-1):
        if l[i] == "crimean":
            l = [f"{l[i]}-{l[i+1]}"] + l[i+2:]
    if l[0] == "none":
        l[0] = None
    return l


def read_file(filename):
    lines = []
    with open(filename, "r") as fh:
        for line in fh.readlines():
            lemma, word, tags = line.replace("\n", " ").split("\t")
            lines.append((lemma, word, tags))
    return lines


def create_rows():
    global my_results
    my_results = pd.DataFrame(columns=COLUMN_ORDER[:2+6*len(ARCHES)])

    for arch in ARCHES:
        for dirname in os.listdir("model/tag-" + arch):
            langs = split_langs(dirname)
            if len(langs) == 1:
                continue
            elif langs[0] is None:
                continue

            baseline_predictions_filename = f"model/tag-{arch}/none-{langs[1]}/predictions.txt"
            transfer_predictions_filename = f"model/tag-{arch}/{dirname}/predictions.txt"
            gold_filename = f"conll2018/task1/all/{langs[1]}-test"
            if not os.path.exists(transfer_predictions_filename):
                continue

            baseline_guesses = read_file(baseline_predictions_filename)
            transfer_guesses = read_file(transfer_predictions_filename)
            answers = read_file(gold_filename)

            assert(len(baseline_guesses) == len(answers))
            assert(len(transfer_guesses) == len(answers))

            correct_baseline_guesses, cumulative_baseline_levenshtein = 0, 0
            correct_transfer_guesses, cumulative_transfer_levenshtein = 0, 0

            for (lemma1, baseline_guess, tags1), (lemma2, transfer_guess, tags2), (lemma3, answer, tags3) in zip(baseline_guesses, transfer_guesses, answers):
                assert(lemma1 == lemma2)
                assert(lemma2 == lemma3)
                assert(tags1 == tags2)
                assert(tags2 == tags3)

                if baseline_guess == answer:
                    correct_baseline_guesses += 1
                cumulative_baseline_levenshtein += levenshtein(baseline_guess, answer)

                if transfer_guess == answer:
                    correct_transfer_guesses += 1
                cumulative_transfer_levenshtein += levenshtein(transfer_guess, answer)

            row = {
                f"Baseline Accuracy ({arch})": correct_baseline_guesses/len(answers),
                f"Transfer Accuracy ({arch})": correct_transfer_guesses/len(answers),
                f"Baseline Levenshtein ({arch})": cumulative_baseline_levenshtein/len(answers),
                f"Transfer Levenshtein ({arch})": cumulative_transfer_levenshtein/len(answers)
            }



            rows = my_results.loc[
                (my_results["Source Language"] == langs[0].replace("-", " ").title()) &
                (my_results["Target Language"] == langs[1].replace("-", " ").title())
            ]

            if len(rows) == 0:
                row["Source Language"] = langs[0].replace("-", " ").title()
                row["Target Language"] = langs[1].replace("-", " ").title()
                my_results = my_results.append(row, ignore_index=True)
            elif len(rows) == 1:
                rows.update(row)
            else:
                raise KeyError

        my_results[f"Accuracy Improvement ({arch})"] = my_results[f"Transfer Accuracy ({arch})"] - my_results[f"Baseline Accuracy ({arch})"]
        my_results[f"Levenshtein Improvement ({arch})"] = my_results[f"Transfer Levenshtein ({arch})"] - my_results[f"Baseline Levenshtein ({arch})"]

        my_results = my_results.round({
            f"Baseline Accuracy ({arch})": 3,
            f"Transfer Accuracy ({arch})": 3,
            f"Accuracy Improvement ({arch})": 3,
            f"Baseline Levenshtein ({arch})": 2,
            f"Transfer Levenshtein ({arch})": 2,
            f"Levenshtein Improvement ({arch})": 2
        })


def calculate_distance():
    distances = []
    for row in my_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])
        if source["Language family"] == target["Language family"]:
            if source["Subfamily"] == target["Subfamily"]:
                distances.append("Closely related")
            else:
                distances.append("Distantly related")
        else:
            distances.append("Unrelated")

    my_results["Distance"] = distances


def calculate_category_overlap(pos):
    category_overlaps = []
    for row in my_results.iterrows():
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

    my_results[f"{pos} category overlap"] = category_overlaps


def calculate_proportions(d):
    total = sum(d.values())
    return dict([(key, value/total) for key, value in d.items()])


def calculate_POS_distribution_overlaps():
    POS_distribution_overlaps = []
    for row in my_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])

        source_POS = calculate_proportions(dict([(pos, source[f"{pos} examples"]) for pos in POS]))
        target_POS = calculate_proportions(dict([(pos, target[f"{pos} examples"]) for pos in POS]))

        distribution_overlap = round(sum(min(source_POS[pos], target_POS[pos]) for pos in POS), 3)
        POS_distribution_overlaps.append(distribution_overlap)

    my_results["POS distribution similarity"] = POS_distribution_overlaps


def calculate_inflection_shape_similarity():
    inflection_shape_similarities = []
    for row in my_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])

        source_prevs = [source[f"{shape} prevalence"] for shape in InflectionShapes._fields]
        target_prevs = [target[f"{shape} prevalence"] for shape in InflectionShapes._fields]

        inflection_shape_similarity = 1 - sum([abs(sp-tp) for sp, tp in zip(source_prevs, target_prevs)])/(sum(source_prevs) + sum(target_prevs))
        inflection_shape_similarities.append(round(inflection_shape_similarity, 3))

    my_results["Inflection shape similarity"] = inflection_shape_similarities


def calculate():
    create_rows()
    calculate_distance()
    for pos in POS:
        calculate_category_overlap(pos)
    calculate_POS_distribution_overlaps()
    calculate_inflection_shape_similarity()
    my_results[COLUMN_ORDER].to_csv(my_results_filename, index=False)