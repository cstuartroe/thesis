from matplotlib import pyplot as plt
import numpy as np
import random

from .general import *


dot_color = '#3f6fcf'
line_color = '#ffdd22'


def get_2018_low_acc(lang_name):
    return SIGMORPHON_2018_results.loc[lang_name]["Low-resource accuracy"]


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
    distant_langs = SIGMORPHON_2019_results[SIGMORPHON_2019_results["Distance"] != "Closely related"]

    for metric in "Accuracy Improvement", "Levenshtein Improvement":
        for explanatory in "N category overlap", "V category overlap":
            title = f"{metric} vs. {explanatory}\nin distantly related and unrelated languages"
            plt.title(title)
            plt.xlabel(explanatory)
            plt.ylabel(metric)

            xs = []
            ys = []
            for row in distant_langs.iterrows():
                x = row[1][explanatory]
                y = row[1][metric]
                if (not pd.isna(x)) and not (pd.isna(y)):
                    xs.append(x)
                    ys.append(y)

            r = np.corrcoef(ys, xs)[0][1]

            plt.plot(xs, ys, '.', color=dot_color)

            m, b = np.polyfit(xs, ys, 1)
            plt.plot([0, max(xs)], [b, b + m*max(xs)], '-', color=line_color)

            plt.figtext(.8, .01, f"r={round(r, 2)} p={round(permutation_test(xs, ys), 3)}")

            plt.savefig(f"images/generated/{metric} vs {explanatory}.png")
            plt.close()
