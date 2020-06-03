import numpy as np
from tabulate import tabulate

from .general import *
from .timer import Timer
from .unimorph import POS
from .alignment import InflectionShapes, levenshtein
from .graphing import graph_one

POS.remove("DET")

row_timer = Timer.get("rows")

COLUMN_ORDER = [
    "Source Language",
    "Target Language",
    "Distance",
    "POS distribution similarity",
    "Inflection shape similarity",
    "Fusion similarity"] + [f"{pos} category overlap" for pos in POS] + flatten([[
    f"Source Language Baseline Accuracy ({arch})",
    f"Baseline Accuracy ({arch})",
    f"Transfer Accuracy ({arch})",
    f"Accuracy Improvement ({arch})",
    f"Source Language Baseline Levenshtein ({arch})",
    f"Baseline Levenshtein ({arch})",
    f"Transfer Levenshtein ({arch})",
    f"Levenshtein Improvement ({arch})"
] for arch in ARCHES])


def split_langs(dirname):
    l = dirname.split("-")
    for i in range(len(l)-1):
        if l[i] == "crimean":
            l = l[:i] + [f"{l[i]}-{l[i+1]}"] + l[i+2:]
    if l[0] == "none":
        l[0] = ""
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
            row_timer.task("setup")
            langs = split_langs(dirname)
            if len(langs) == 1:
                continue
            # elif langs[0] is None:
            #     continue

            # baseline_predictions_filename = f"model/tag-{arch}/none-{langs[1]}/predictions.txt"
            transfer_predictions_filename = f"model/tag-{arch}/{dirname}/predictions.txt"
            gold_filename = f"conll2018/task1/all/{langs[1]}-test"
            if not os.path.exists(transfer_predictions_filename):
                continue

            row_timer.task("reading files")

            # baseline_guesses = read_file(baseline_predictions_filename)
            transfer_guesses = read_file(transfer_predictions_filename)
            answers = read_file(gold_filename)

            # assert(len(baseline_guesses) == len(answers))
            assert(len(transfer_guesses) == len(answers))

            # correct_baseline_guesses, cumulative_baseline_levenshtein = 0, 0
            correct_transfer_guesses, cumulative_transfer_levenshtein = 0, 0

            for (lemma1, guess, tags1), (lemma2, answer, tags2) in zip(transfer_guesses, answers):
                row_timer.task("counting guesses")
                assert(lemma1 == lemma2)
                assert(tags1 == tags2)

                if guess == answer:
                    correct_transfer_guesses += 1

                row_timer.task("levenshteining")
                cumulative_transfer_levenshtein += levenshtein(guess, answer)

            row_timer.task("setting rows")

            row = {
                # f"Baseline Accuracy ({arch})": correct_baseline_guesses/len(answers),
                f"Transfer Accuracy ({arch})": correct_transfer_guesses/len(answers),
                # f"Baseline Levenshtein ({arch})": cumulative_baseline_levenshtein/len(answers),
                f"Transfer Levenshtein ({arch})": cumulative_transfer_levenshtein/len(answers)
            }

            cond = (my_results["Source Language"] == langs[0].replace("-", " ").title()) &\
                   (my_results["Target Language"] == langs[1].replace("-", " ").title())
            rows = my_results.loc[cond]

            if len(rows) == 0:
                row["Source Language"] = langs[0].replace("-", " ").title()
                row["Target Language"] = langs[1].replace("-", " ").title()
                my_results = my_results.append(row, ignore_index=True)
            elif len(rows) == 1:
                for key, value in row.items():
                    my_results.loc[cond, key] = value
            else:
                raise KeyError

        row_timer.task("extra stats")

        baseline_accuracies = []
        baseline_levenshteins = []
        source_accuracies = []
        source_levenshteins = []
        for row in my_results.iterrows():
            if row[1]["Source Language"] == "":
                baseline_accuracies.append(0)
                baseline_levenshteins.append(0)
                source_accuracies.append(0)
                source_levenshteins.append(0)
            else:
                baseline_row = my_results.loc[(my_results["Source Language"] == "") &
                                              (my_results["Target Language"] == row[1]["Target Language"])]
                baseline_accuracies.append(baseline_row.iloc[0][f"Transfer Accuracy ({arch})"])
                baseline_levenshteins.append(baseline_row.iloc[0][f"Transfer Levenshtein ({arch})"])

                source_row = my_results.loc[(my_results["Source Language"] == "") &
                                              (my_results["Target Language"] == row[1]["Source Language"])]
                source_accuracies.append(source_row.iloc[0][f"Transfer Accuracy ({arch})"])
                source_levenshteins.append(source_row.iloc[0][f"Transfer Levenshtein ({arch})"])

        my_results[f"Baseline Accuracy ({arch})"] = baseline_accuracies
        my_results[f"Baseline Levenshtein ({arch})"] = baseline_levenshteins
        my_results[f"Source Language Baseline Accuracy ({arch})"] = source_accuracies
        my_results[f"Source Language Baseline Levenshtein ({arch})"] = source_levenshteins
        my_results[f"Accuracy Improvement ({arch})"] = my_results[f"Transfer Accuracy ({arch})"] - my_results[f"Baseline Accuracy ({arch})"]
        my_results[f"Levenshtein Improvement ({arch})"] = my_results[f"Baseline Levenshtein ({arch})"] - my_results[f"Transfer Levenshtein ({arch})"]

        row_timer.task("rounding")

        my_results = my_results.round({
            f"Baseline Accuracy ({arch})": 3,
            f"Transfer Accuracy ({arch})": 3,
            f"Accuracy Improvement ({arch})": 3,
            f"Baseline Levenshtein ({arch})": 2,
            f"Transfer Levenshtein ({arch})": 2,
            f"Levenshtein Improvement ({arch})": 2
        })

    row_timer.report()

    my_results = my_results[my_results["Source Language"] != ""].sort_values(["Target Language", "Source Language"])


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


def calculate_fusion_similarity():
    fusion_similarities = []
    for row in my_results.iterrows():
        source = get_language_by_name(row[1]["Source Language"])
        target = get_language_by_name(row[1]["Target Language"])

        file_pattern = "csv/fusion/%s.csv"
        source_fusion_filename = file_pattern % source["ISO 639-2"]
        target_fusion_filename = file_pattern % target["ISO 639-2"]

        with open(source_fusion_filename, "r") as fh:
            source_raw_rows = [r.strip().split(",") for r in fh.readlines()]
        with open(target_fusion_filename, "r") as fh:
            target_raw_rows = [r.strip().split(",") for r in fh.readlines()]

        fusion_sum, fusion_diff = 0, 0

        for i in range(1, len(source_raw_rows)):
            for j in range(1, len(source_raw_rows)):
                s, t = float(source_raw_rows[i][j]), float(target_raw_rows[i][j])
                fusion_sum += abs(s) + abs(t)
                fusion_diff += abs(s - t)

        if fusion_diff > fusion_sum:
            print(source_raw_rows)
            print(target_raw_rows)
            print(fusion_diff, fusion_sum)
            raise ValueError
        fusion_similarities.append(round(1 - fusion_diff/fusion_sum, 3))

    my_results["Fusion similarity"] = fusion_similarities


def calculate():
    print("Creating rows...")
    create_rows()
    print("Calculating distance...")
    calculate_distance()
    for pos in POS:
        print(f"Calculating {pos} overlap...")
        calculate_category_overlap(pos)
    print("Calculating distribution overlaps...")
    calculate_POS_distribution_overlaps()
    print("Calculating inflection shape similarity...")
    calculate_inflection_shape_similarity()
    print("Calculating fusion similarities...")
    calculate_fusion_similarity()
    my_results[COLUMN_ORDER].to_csv(my_results_filename, index=False)


def overall_model_performance():
    def f(vbl):
        if "Levenshtein" in vbl:
            return lambda n: round(n, 2)
        else:
            return lambda n: str(round(n*100, 1)) + "\\%"

    for response_vbl in ["Baseline Accuracy (%s)", "Baseline Levenshtein (%s)"]:
        for arch in ["hard", "soft"]:
            results = {}
            for row in my_results.iterrows():
                results[row[1]["Target Language"]] = row[1][response_vbl % arch]

            mu, sigma = np.average(list(results.values())), np.std(list(results.values()))

            fn = f(response_vbl)
            print(response_vbl[:-5] if arch == "hard" else "", end=" & ")
            print(f"{arch} & {fn(mu)} & {fn(sigma)} \\\\\n\\hline")

    for response_vbl in ["Transfer Accuracy (%s)", "Transfer Levenshtein (%s)"]:
        for arch in ["hard", "soft"]:
            results = []
            for row in my_results.iterrows():
                results.append(row[1][response_vbl % arch])

            mu, sigma = np.average(results), np.std(results)

            fn = f(response_vbl)
            print(response_vbl[:-5] if arch == "hard" else "", end=" & ")
            print(f"{arch} & {fn(mu)} & {fn(sigma)}\\\\\n\\hline")


def accuracy_avgs():
    for st in ["Source", "Target"]:
        for response_vbl in ["Accuracy Improvement (hard)", "Levenshtein Improvement (hard)",
                             "Baseline Accuracy (hard)", "Transfer Accuracy (hard)"]:
            source_lang_improvements = {}
            for row in my_results.iterrows():
                if row[1]["Distance"] == "Unrelated":
                    sl = row[1][st + " Language"]
                    source_lang_improvements[sl] = source_lang_improvements.get(sl, []) + [row[1][response_vbl]]

            source_lang_improvement_avgs = []
            for source_lang, improvements in source_lang_improvements.items():
                mu = round(sum(improvements)/len(improvements), 3)
                sigma = round(np.std(improvements), 3)
                source_lang_improvement_avgs.append((source_lang, mu, sigma))

            source_lang_improvement_avgs.sort(key=lambda x: x[1])

            print(f"{response_vbl} by {st} Language")
            print(tabulate([("Language", "mean", "stdev")] + source_lang_improvement_avgs))

            for i in range(len(source_lang_improvement_avgs)):
                for j in range(i+1, len(source_lang_improvement_avgs)):
                    lang1, mu1, sigma1 = source_lang_improvement_avgs[i]
                    lang2, mu2, sigma2 = source_lang_improvement_avgs[j]
                    if (mu1 + 1.96*sigma1) <= (mu2 - 1.96*sigma2):
                        print(f"{lang1} < {lang2}")
                    elif (mu1 - 1.96*sigma1) >= (mu2 + 1.96*sigma2):
                        print(f"{lang2} < {lang1}")


def source_language_effects():
    for arch in ["hard", "soft"]:
        sl_accuracies = {}
        sl_improvements = {}
        for row in my_results.iterrows():
            sl = row[1]["Source Language"]
            sl_accuracies[sl] = row[1][f"Source Language Baseline Accuracy ({arch})"]
            sl_improvements[sl] = sl_improvements.get(sl, [])
            sl_improvements[sl].append(row[1][f"Accuracy Improvement ({arch})"])

        sl_mus, sl_sigmas = {}, {}
        for sl, l in sl_improvements.items():
            sl_mus[sl] = sum(l)/len(l)
            sl_sigmas[sl] = np.std(l)

        for row in my_results.iterrows():
            sl = row[1]["Source Language"]
            tl = row[1]["Target Language"]
            improvement = row[1][f"Accuracy Improvement ({arch})"]
            if improvement < sl_mus[sl] - 1.96*sl_sigmas[sl]:
                print(f"{sl} -> {tl} improvement was {improvement} << {sl} avg of {sl_mus[sl]}")
            if improvement > sl_mus[sl] + 1.96*sl_sigmas[sl]:
                print(f"{sl} -> {tl} improvement was {improvement} >> {sl} avg of {sl_mus[sl]}")

        sample_mu = sum(sl_mus.values())/len(sl_mus)
        sample_sigma = np.std(list(sl_mus.values()))

        sl_accuracies_list = []
        sl_improvements_list = []
        table = []
        for sl in sl_accuracies.keys():
            sl_accuracies_list.append(sl_accuracies[sl])
            sl_improvements_list.append(sl_mus[sl])
            table.append((sl, sl_accuracies[sl], sl_mus[sl], sl_sigmas[sl]))

            if sl_mus[sl] < sample_mu - 1.96*sample_sigma:
                print(sl, "< avg")
            elif sl_mus[sl] > sample_mu + 1.96*sample_sigma:
                print(sl, "> avg")

        graph_one(sl_accuracies_list, sl_improvements_list,
                  title=f"Source Language Baseline Accuracy ({arch}) vs. Average Target\nLanguage Accuracy Improvement ({arch}) by Source Language",
                  xlabel=f"Source Language Baseline Accuracy ({arch})",
                  ylabel=f"Average Target Language Accuracy Improvement ({arch})",
                  langset_name="Source Languages")

        table.sort(key=lambda row: row[2])

        print(r"\begin{tabular}{|c|c|c|c|}")
        print("\\hline")
        print(f"Source Language & SL Baseline Accuracy & \\multicolumn{{2}}{{|c|}}{{TL Accuracy Improvement ({arch})}} \\\\")
        print(f"& ({arch}) & $\\mu$ & $\\sigma$ \\\\")
        print("\\hline\n\\hline")
        for sl, acc, mu, sigma in table:
            print(f"{sl} & {round(acc,3)} & {round(mu,3)} & {round(sigma,3)}\\\\\n\\hline")
        print(r"\end{tabular}")


def extra_stats():
    # overall_model_performance()
    # accuracy_avgs()
    source_language_effects()