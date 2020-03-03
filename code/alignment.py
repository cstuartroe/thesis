from collections import namedtuple
from tqdm import tqdm

from .general import *

InflectionShapes = namedtuple('InflectionShapes', ['prefix', 'infix', 'alternation', 'suffix'])
PAD_CHR = '$'


def hamming_distance(s1, s2):
    assert(len(s1) == len(s2))
    # lol code golfing it up
    return sum([c1 != c2 for c1, c2 in zip(s1, s2)])


def sigmorphon2016align(s1, s2):
    min_dist = len(s1)+len(s2)
    min_align = None, None
    for shift in range(len(s1) + len(s2)):
        s1pad = PAD_CHR*max(shift-len(s1), 0) + s1 + PAD_CHR*max(len(s2)-shift, 0)
        s2pad = PAD_CHR*max(len(s1)-shift, 0) + s2 + PAD_CHR*max(shift-len(s2), 0)
        dist = hamming_distance(s1pad, s2pad)
        if dist < min_dist:
            min_dist = dist
            min_align = s1pad, s2pad
    return min_align


def levenshtein_align(s1, s2):
    diffs = [[(0, 0, 0)]]

    for j in range(1, len(s2)+1):
        diffs[-1].append((0, -1, j))

    for i in range(1, len(s1)+1):
        diffs.append([(-1, 0, i)])

        for j in range(1, len(s2)+1):
            x, y, d = 0, -1, diffs[i][j - 1][2] + 1
            if diffs[i - 1][j][2] + 1 < d:
                x, y, d = -1, 0, diffs[i - 1][j][2] + 1

            sub_cost = 0 if s1[i-1] == s2[j-1] else 1.5
            if diffs[i - 1][j - 1][2] + sub_cost < d:
                x, y, d = -1, -1, diffs[i-1][j-1][2] + sub_cost

            diffs[-1].append((x, y, d))

    i, j = len(s1) - 1, len(s2) - 1
    s1pad, s2pad = "", ""
    while i > -1 or j > -1:
        x, y, _ = diffs[i+1][j+1]
        s1pad = (PAD_CHR if x == 0 else s1[i]) + s1pad
        s2pad = (PAD_CHR if y == 0 else s2[j]) + s2pad
        i, j = i+x, j+y

    return s1pad, s2pad


def get_inflection_shapes(lemma, form):
    lemmapad, formpad = levenshtein_align(lemma, form)

    prefix, infix, alternation, suffix = False, False, False, False
    z = list(zip(lemmapad, formpad))

    i = 0
    while PAD_CHR in z[i]:
        prefix = True
        i += 1
    z = z[i:]

    i = -1
    while PAD_CHR in z[i]:
        suffix = True
        i -= 1
    if i != -1:
        z = z[:i+1]

    for c1, c2 in z:
        if PAD_CHR in (c1, c2):
            infix = True
        elif c1 != c2:
            alternation = True

    return InflectionShapes(prefix, infix, alternation, suffix)


def get_inflection_shape_prevalences(lang_name):
    prevs = [0]*len(InflectionShapes._fields)

    triplets = get_language_training_data(lang_name)
    for triplet in triplets:
        lemma, form = triplet["lemma"], triplet["inflected_form"]
        has_shapes = get_inflection_shapes(lemma, form)
        for i, has_shape in enumerate(has_shapes):
            if has_shape:
                prevs[i] += 1

    return list(map(lambda x: x/len(triplets), prevs))


def pull():
    for shape in InflectionShapes._fields:
        language_info[f"{shape} prevalence"] = 0.0

    for index, row in tqdm(list(language_info.iterrows())):
        prevs = get_inflection_shape_prevalences(row["Name"])
        for i in range(len(prevs)):
            language_info.loc[index, f"{InflectionShapes._fields[i]} prevalence"] = prevs[i]

    language_info.to_csv(language_info_filename, index=False)
