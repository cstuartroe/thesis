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
    try:
        r = np.corrcoef(list1, list2)[0, 1]
    except TypeError as e:
        print(list1)
        print(list2)
        raise e
    random_rs = []
    for i in range(10000):
        random.shuffle(list1)
        random.shuffle(list2)
        random_rs.append(np.corrcoef(list1, list2)[0, 1])

    num_rs_greater = len([random_r for random_r in random_rs if abs(random_r) > abs(r)])
    return num_rs_greater/10000


GENEALOGICAL_NUMERIC = {
    "Closely related": 2,
    "Distantly related": 1,
    "Unrelated": 0
}

METRICS = flatten([
    (f"Accuracy Improvement ({arch})", f"Levenshtein Improvement ({arch})",
     f"Transfer Accuracy ({arch}", f"Transfer Levenshtein ({arch}") for arch in ARCHES
])

EXPLANATORIES = [
    "N category overlap",
    "V category overlap",
    "ADJ category overlap",
    "POS distribution similarity",
    "Genealogical similarity",
    "Inflection shape similarity",
    "Fusion similarity"
]

langsets = {
    "all language pairs": my_results,
    "distantly related and unrelated language pairs": my_results[my_results["Distance"] != "Closely related"],
    "closely related language pairs": my_results[my_results["Distance"] == "Closely related"]
}


def graph_one(xs, ys, title, xlabel, ylabel, langset_name):
    GRAPH_TIMER.task("Setup")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    GRAPH_TIMER.task("corrcoef")
    try:
        r = np.corrcoef(ys, xs)[0][1]
    except TypeError as e:
        print(ys)
        print(xs)
        raise e

    GRAPH_TIMER.task("Plotting")
    plt.plot(xs, ys, '.', color=dot_color)

    GRAPH_TIMER.task("polyfit")
    m, b = np.polyfit(xs, ys, 1)
    plt.plot([0, max(xs)], [b, b + m * max(xs)], '-', color=line_color)

    # print(f"{metric} ~= {round(m,2)}*({explanatory}) + {round(b,2)}")

    GRAPH_TIMER.task("permtest")
    p = permutation_test(xs, ys)

    GRAPH_TIMER.task("Saving")
    plt.figtext(.7, .01, f"n={len(xs)} r={round(r, 2)} p={round(p, 3)}")
    plt.figtext(.1, .01, f"y = {round(m, 2)}x + {round(b, 2)}")

    filename = f"images/generated/{'in' if p > .05 else ''}significant" + \
                f"/{ylabel.replace(' ', '_')}_vs_{xlabel.replace(' ', '_')}" + \
                f"_{langset_name.replace(' ', '_')}.png"

    plt.savefig(filename)
    plt.close()


def graph():
    for polarity in ['', 'in']:
        folder = f"images/generated/{polarity}significant/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if filename != ".gitkeep" and (os.path.isfile(file_path) or os.path.islink(file_path)):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    pairs = [(m, e) for m in METRICS for e in EXPLANATORIES]

    for metric, explanatory in tqdm(pairs):
        for langset_name, langset in langsets.items():
            xs = []
            ys = []
            for row in langset.iterrows():
                if explanatory == "Genealogical similarity":
                    x = GENEALOGICAL_NUMERIC[row[1]["Distance"]]
                else:
                    x = row[1][explanatory]
                y = row[1][metric]
                if (not pd.isna(x)) and not (pd.isna(y)):
                    xs.append(x)
                    ys.append(y)

            graph_one(xs=xs, ys=ys,
                      title=f"{metric} vs. {explanatory}\nin {langset_name}",
                      xlabel=explanatory, ylabel=metric, langset_name=langset_name)

    GRAPH_TIMER.report()
