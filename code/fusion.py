from tqdm import tqdm

from .general import *
from .alignment import levenshtein_diffs

from .taginfo import UNIMORPH_CATEGORIES, UNIMORPH_TAGTYPES, TAGTYPE_INDICES, \
    UNIMORPH_POS_MAP, UNIMORPH_CONFLICT_RESOLUTION


def levenshtein(s1, s2):
    diffs = levenshtein_diffs(s1, s2)
    return diffs[-1][-1][-1]


def set_tagtype(tagvec, tagtype, tag, lang_name):
    i = TAGTYPE_INDICES[tagtype]
    if tagvec[i] is not None and tagvec[i] != tag and tagvec[i] != "DET":
        if lang_name in UNIMORPH_CONFLICT_RESOLUTION:
            if (tag, tagvec[i]) in UNIMORPH_CONFLICT_RESOLUTION[lang_name]:
                tagvec[i] = UNIMORPH_CONFLICT_RESOLUTION[lang_name][(tag, tagvec[i])]
                return

            if (tagvec[i], tag) in UNIMORPH_CONFLICT_RESOLUTION[lang_name]:
                tagvec[i] = UNIMORPH_CONFLICT_RESOLUTION[lang_name][(tagvec[i], tag)]
                return

        raise ValueError(f"Warning: setting {tagtype} to {tag}, already set to {tagvec[i]}")

    tagvec[i] = tag


def encode_tags(taglist, lang_name):
    tagvec = [None]*len(UNIMORPH_CATEGORIES)
    for tag in taglist:
        if tag in UNIMORPH_TAGTYPES:
            tagtype = UNIMORPH_TAGTYPES[tag]
            set_tagtype(tagvec, tagtype, tag, lang_name)
            if tagtype == "Definiteness" and tagvec[0] is None:
                set_tagtype(tagvec, "POS", "DET", lang_name)

        if tag in UNIMORPH_POS_MAP:
            pos = UNIMORPH_POS_MAP[tag]
            set_tagtype(tagvec, "POS", pos, lang_name)


def tag_diffs(encoded_tags1, encoded_tags2):
    return sum([t1 == t2 for t1, t2 in zip(encoded_tags1, encoded_tags2)])


def get_fusion(lang_name):
    triplets = get_language_training_data(lang_name)

    for triplet in triplets:
        lemma = triplet["lemma"]
        form = triplet["inflected_form"]

        try:
            tags = encode_tags(triplet["unimorph_tags"].split(";"), lang_name)

        except ValueError as e:
            if lang_name not in {"Kurmanji", "Sorani"}:
                print(lang_name, triplet)
                raise e


def pull():
    for index, row in (list(language_info.iterrows())):
        get_fusion(row["Name"])

    # language_info.to_csv(language_info_filename, index=False)
