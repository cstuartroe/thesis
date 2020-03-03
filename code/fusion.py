from tqdm import tqdm
import csv
import random

from .general import *
from .alignment import levenshtein_diffs
from .taginfo import UNIMORPH_CATEGORIES, UNIMORPH_TAGTYPES, TAGTYPE_INDICES, \
    UNIMORPH_POS_MAP, UNIMORPH_CONFLICT_RESOLUTION
from .timer import Timer

random.seed(2020)

FUSION_TIMER = Timer.get("fusion")


def levenshtein(s1, s2):
    diffs = levenshtein_diffs(s1, s2)
    return diffs[-1][-1][-1]


def set_tagtype(tagvec, tagtype, tag, lang_name):
    """For whatever reason, determiner part of speech is never explicitly marked in this dataset.
       Words with a definiteness annotation and no explicit part of speech are regarded here as
       determiners, but other parts of speech override it.

       tagvec: A reference to the array where a tag vector is being recorded in place.
       tagtype: The category of tag, e.g., part of speech, tense, case
       tag: The value to be set
       lang_name: Used for language-specific conflict resolution, based on idiosyncracies in how
                  each language has been encoded in this dataset
    """

    if tagtype == "Definiteness" and tagvec[0] is None:
        set_tagtype(tagvec, "POS", "DET", lang_name)

    i = TAGTYPE_INDICES[tagtype]

    if tagvec[i] is not None and tagvec[i] != tag and tagvec[i] != "DET":
        if lang_name in UNIMORPH_CONFLICT_RESOLUTION:
            for ordering in (tag, tagvec[i]), (tagvec[i], tag):
                if ordering in UNIMORPH_CONFLICT_RESOLUTION[lang_name]:
                    tagvec[i] = UNIMORPH_CONFLICT_RESOLUTION[lang_name][ordering]
                    return

        raise ValueError(f"Warning: setting {tagtype} to {tag}, already set to {tagvec[i]}")

    tagvec[i] = tag


def encode_tags(taglist, lang_name):
    """Take a list of tags in arbitrary order and convert to a vector where each position corresponds
       to a particular category of tag, e.g., tagvec[0] is part of speech, tagvec[1] is grammatical person,
       etc. Many tags are liable to be ignored.
    """
    tagvec = [None]*len(UNIMORPH_CATEGORIES)
    for tag in taglist:
        if tag in UNIMORPH_TAGTYPES:
            tagtype = UNIMORPH_TAGTYPES[tag]
            set_tagtype(tagvec, tagtype, tag, lang_name)

        if tag in UNIMORPH_POS_MAP:
            pos = UNIMORPH_POS_MAP[tag]
            set_tagtype(tagvec, "POS", pos, lang_name)

    return tagvec


def tag_diffs(encoded_tags1, encoded_tags2):
    diffs = []
    for i, tagtype in enumerate(UNIMORPH_CATEGORIES.keys()):
        if encoded_tags1[i] != encoded_tags2[i]:
            diffs.append(tagtype)
    return diffs


def get_encoded_forms(lang_name):
    """Produce a dictionary of dictionaries, which indexes all inflected forms primarily by part of speech,
       and secondarily by lemma. Each entry records a tag vector and the inflected form. e.g.,
       {
           "V": {
               "wish": [
                   ("wished", ["V", "PST", ...]),
                   ...
               ],
               ...
           },
           ...
       }
    """
    triplets = get_language_training_data(lang_name)
    by_lemma = {}

    for triplet in triplets:
        lemma = triplet["lemma"]
        form = triplet["inflected_form"]

        try:
            tagvec = encode_tags(triplet["unimorph_tags"].split(";"), lang_name)

        except ValueError as e:
            if lang_name not in {"Kurmanji", "Sorani"}:
                print(lang_name, triplet)
                raise e
            continue

        if lemma not in by_lemma:
            by_lemma[lemma] = []

        by_lemma[lemma].append((form, tagvec))

    return by_lemma


def get_fusion(lang_name):
    FUSION_TIMER.task("Encoded forms")

    encoded_forms = get_encoded_forms(lang_name)

    # by individual tagtype, list of levenshtein differences of forms which vary only on this tagtype
    single_diff_distances = {}

    # by pair of tagtypes, list of levenshtein differences of forms which vary only on these two tagtypes
    double_diff_distances = {}

    FUSION_TIMER.task("Diffing")

    form_pairs = []
    total_pairs = 0
    for lemma, formlist in encoded_forms.items():
        for i in range(len(formlist)):
            for j in range(i+1, len(formlist)):
                total_pairs += 1
                form1, tagvec1 = formlist[i]
                form2, tagvec2 = formlist[j]
                diffs = tag_diffs(tagvec1, tagvec2)

                if len(diffs) in [1, 2]:
                    form_pairs.append((form1, form2, diffs))

    FUSION_TIMER.task("Shuffling")

    random.shuffle(form_pairs)
    form_pairs = form_pairs[:10**5]

    for form1, form2, diffs in form_pairs:
        FUSION_TIMER.task("Levenshtein")
        ldist = levenshtein(form1, form2)

        FUSION_TIMER.task("Dict setting")

        if len(diffs) == 1:
            single_diff_distances[diffs[0]] = single_diff_distances.get(diffs[0], []) + [ldist]

        elif len(diffs) == 2:
            for ordering in (diffs[0], diffs[1]), (diffs[1], diffs[0]):
                double_diff_distances[ordering] = double_diff_distances.get(ordering, []) + [ldist]

    FUSION_TIMER.task("Averaging")

    single_diff_distance_averages = {}
    for tagtype, dists in single_diff_distances.items():
        single_diff_distance_averages[tagtype] = sum(dists)/len(dists)

    double_diff_distance_averages = {}
    for pair, dists in double_diff_distances.items():
        double_diff_distance_averages[pair] = sum(dists) / len(dists)

    FUSION_TIMER.task("Constructing fusion grid")

    fusion_grid = []
    for tagtype1 in UNIMORPH_CATEGORIES:
        fusion_grid.append([])
        for tagtype2 in UNIMORPH_CATEGORIES:
            if (tagtype1, tagtype2) in double_diff_distance_averages \
              and tagtype1 in single_diff_distance_averages and tagtype2 in single_diff_distance_averages:
                ddd = double_diff_distance_averages[(tagtype1, tagtype2)]
                sda_sum = single_diff_distance_averages[tagtype1] + single_diff_distance_averages[tagtype2]
                fusion_coefficient = round(1 - (ddd / sda_sum), 3)
            else:
                fusion_coefficient = 0
            fusion_grid[-1].append(fusion_coefficient)

    return fusion_grid


def pull():
    for index, row in tqdm(list(language_info.iterrows())):
        fusion_grid = get_fusion(row["Name"])
        FUSION_TIMER.task("to csv")
        with open(f"csv/fusion/{row['ISO 639-2']}.csv", "w") as fh:
            writer = csv.writer(fh)
            writer.writerows(fusion_grid)

    FUSION_TIMER.report()
