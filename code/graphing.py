from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from .general import *
from .timer import Timer

random.seed(2019)

GRAPH_TIMER = Timer.get("graph")

dot_color = '#3f6fcf'
line_color = '#ffdd22'


def permutation_test(list1, list2):
    """Performs a permutation test to assess confidence of correlation"""
    r = np.corrcoef(list1, list2)[0, 1]
    random_rs = []
    for i in range(10000):
        random.shuffle(list1)
        random.shuffle(list2)
        random_rs.append(np.corrcoef(list1, list2)[0, 1])

    num_rs_greater = len([random_r for random_r in random_rs if abs(random_r) > abs(r)])
    return num_rs_greater/10000


def graph():
    metrics = [
        "Accuracy Improvement vs 2018",
        "Accuracy Improvement vs Baseline",
        "Levenshtein Improvement",
        "Best Transfer Accuracy",
        "Best 2018 Accuracy"
    ]

    explanatories = [
        "N category overlap",
        "V category overlap",
        "ADJ category overlap",
        "POS distribution similarity",
        "Genealogical distance",
        "Inflection shape similarity"
    ]

    turkic_languages = list(language_info[language_info["Language family"] == "Turkic"]["Name"])

    langsets = {
        "all language pairs": SIGMORPHON_2019_results,
        "distantly related and unrelated language pairs": SIGMORPHON_2019_results[SIGMORPHON_2019_results["Distance"] != "Closely related"],
        "Turkic languages": SIGMORPHON_2019_results[SIGMORPHON_2019_results["Target Language"].isin(turkic_languages)],
        "non-Turkic languages": SIGMORPHON_2019_results[~SIGMORPHON_2019_results["Target Language"].isin(turkic_languages)]
    }

    langsets["non-Turkic distantly related and unrelated language pairs"] = langsets["non-Turkic languages"][langsets["non-Turkic languages"]["Distance"] != "Closely related"]

    genealogical_numeric = {
        "Closely related": 2,
        "Distantly related": 1,
        "Unrelated": 0
    }

    for polarity in ['', 'in']:
        folder = f"images/generated/{polarity}significant/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if filename != ".gitkeep" and (os.path.isfile(file_path) or os.path.islink(file_path)):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    for i, metric in enumerate(metrics):
        print(f"{i+1}/{len(metrics)} {metric}")
        for explanatory in tqdm(explanatories):
            for langset_name, langset in langsets.items():
                GRAPH_TIMER.task("Setup")
                title = f"{metric} vs. {explanatory}\nin {langset_name}"
                plt.title(title)
                plt.xlabel(explanatory)
                plt.ylabel(metric)

                xs = []
                ys = []
                for row in langset.iterrows():
                    if explanatory == "Genealogical distance":
                        x = genealogical_numeric[row[1]["Distance"]]
                    else:
                        x = row[1][explanatory]
                    y = row[1][metric]
                    if (not pd.isna(x)) and not (pd.isna(y)):
                        xs.append(x)
                        ys.append(y)

                GRAPH_TIMER.task("corrcoef")
                r = np.corrcoef(ys, xs)[0][1]

                GRAPH_TIMER.task("Plotting")
                plt.plot(xs, ys, '.', color=dot_color)

                GRAPH_TIMER.task("polyfit")
                m, b = np.polyfit(xs, ys, 1)
                plt.plot([0, max(xs)], [b, b + m*max(xs)], '-', color=line_color)

                # print(f"{metric} ~= {round(m,2)}*({explanatory}) + {round(b,2)}")

                GRAPH_TIMER.task("permtest")
                p = permutation_test(xs, ys)

                GRAPH_TIMER.task("Saving")
                plt.figtext(.7, .01, f"n={len(xs)} r={round(r, 2)} p={round(p, 3)}")
                plt.figtext(.1, .01, f"y = {round(m, 2)}x + {round(b, 2)}")

                plt.savefig(f"images/generated/{'in' if p>.05 else ''}significant"
                            f"/{metric.replace(' ', '_')}_vs_{explanatory.replace(' ', '_')}"
                            f"_{langset_name.replace(' ', '_')}.png")
                plt.close()

    GRAPH_TIMER.report()
