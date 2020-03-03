from tqdm import tqdm

from .general import *

UNIMORPH_CATEGORIES = {
    "POS": {"DET", "V", "N", "ADJ"},
    "Person": {"0", "1", "2", "3", "4"},
    "Polarity": {"POS", "NEG"},
    "Politeness": {"INFM", "FORM", "COL", "LIT", "POL"},
    "Number": {"SG", "DU", "PL"},
    "Tense": {"PST", "PRS", "FUT", "RMT", "PRES", "PAST", "HYP"},  # PRES=PRS, PAST=PST, HYP are nonstandard
    "Mood": {"IND", "SBJV", "IMP", "COND", "POT", "OPT", "ADM", "SUBJ"},  # SUBJ=SBJV is nonstandard
    "Case": {"NOM", "ACC", "GEN", "DAT", "INS", "VOC", "ABL", "ESS", "ERG", "ABS", "COM", "TERM", "TRANS", "PRT",
             "PRIV", "FRML",
             "LOC", "INST", "NOM/ACC", "NOM/ACC/DAT", "DAT/GEN",  # LOC, INST=INS are nonstandard
             "AT+ABL", "AT+ALL", "AT+ESS", "IN+ABL", "IN+ALL", "IN+ESS"},
    "Definiteness": {"DEF", "INDF", "NDEF", "DEF/INDF"},  # NDEF=INDF is nonstandard
    "Aspect": {"IPFV", "PRF", "PFV", "PROG", "HAB"},
    "Gender": {"MASC", "FEM", "NEUT",
               'BANTU1', 'BANTU2', 'BANTU3', 'BANTU4', 'BANTU5', 'BANTU6', 'BANTU7', 'BANTU8', 'BANTU9', 'BANTU10',
               'BANTU11', 'BANTU14', 'BANTU15', 'BANTU17'},  # NAKH
    "Animacy": {"ANIM", "INAN"},
    "Voice": {"ACT", "PASS"},
    "Finiteness": {"FIN", "NFIN"},
    "Comparison": {"CMPR", "SPRL"},
    "Interrogativity": {"DECL", "INTR"},  # INTR is supposed to mean intransitive, but it's used in Turkish here as INT
    "Possession": {'PSSB10', 'PSSB8', 'PSS2PF', 'PSS3PF', 'PSSB9', 'PSSB1', 'PSSB7', 'PSSB11', 'PSS1PE', 'PSS1P',
                   'PSSB5', 'PSS2SF', 'PSS2P', 'PSS1PI', 'PSSB17', 'PSSB15', 'PSS3SM', 'PSS3SF', 'PSS1S', 'PSS2S',
                   'PSS3S', 'PSSB14', 'PSSB2', 'PSSB3', 'PSS3PM', 'PSS3P', 'PSSB4', 'PSS2PM', 'PSS2SM', 'PSSB6'},
    # "Language-specific":
}

UNIMORPH_IGNORE = {
    'LGSPEC1', 'LGSPEC2', 'LGSPEC3', 'LGSPEC4', 'LGSPEC5', 'LGSPEC6', 'LGSPEC7', 'LGSPEC8', 'LGSPEC9', 'LGSPEC10',
    'LGSPEC11', 'LGSPEC13',

    "ARGABS1", "ARGABS2", "ARGABS3",
    "ARGABSPL", "ARGABSSG",
    "ARGERG1", "ARGERG2", "ARGERG3",
    "ARGERGFEM", "ARGERGMASC",
    "ARGERGPL", "ARGERGSG",
    "ARGIO1", "ARGIO2", "ARGIO3",
    "ARGIOFEM", "ARGIOINFM", "ARGIOMASC", "ARGIOPL", "ARGIOSG", "ARGERGINFM", "ARGABSINFM",

    "PSSD"
}


def reverse_dict(d):
    out = {}
    for key, iterable in d.items():
        for v in iterable:
            if v in out:
                raise ValueError(f"Duplicate: {v} is in both {key} and {out[v]}")
            out[v] = key
    return out


UNIMORPH_TAGTYPES = reverse_dict(UNIMORPH_CATEGORIES)

TAGTYPE_INDICES = dict([(tagtype, i) for i, tagtype in enumerate(UNIMORPH_CATEGORIES.keys())])

UNIMORPH_POS_MAP = {
    "V.CONV": "V",
    "V.CVB": "V",
    "V.MSDR": "V",
    "V.NFIN": "V",
    "V.PTCP": "V",
}

UNIMORPH_CONFLICT_RESOLUTION = {
    "Zulu": {
        ("RMT", "PST"): "RMT"
    },
    "North Frisian": {
        ("RMT", "PST"): "RMT"
    },
    "Swahili": {
        ("IND", "COND"): "COND"
    },
    "Cornish": {
        ("IND", "COND"): "COND"
    },
    "Adyghe": {
        ("DEF", "NDEF"): "DEF"
    },
    "Breton": {
        ("IPFV", "HAB"): "IPFV/HAB",
        ("IPFV", "PROG"): "IPFV/PROG"
    },
    "Sorani": {
        ("2", "3"): "2/3"
    }
}


assert(len(set(UNIMORPH_POS_MAP.keys()) & UNIMORPH_IGNORE) == 0)
KNOWN_TAGS = set(UNIMORPH_POS_MAP.keys()) | UNIMORPH_IGNORE
for s in UNIMORPH_CATEGORIES.values():
    if len(s & KNOWN_TAGS) != 0:
        raise ValueError("Duplicated tag: " + str(s))
    KNOWN_TAGS = KNOWN_TAGS | s


def get_tags(lang_name):
    triplets = get_language_training_data(lang_name)
    alltags = {}

    for triplet in triplets:
        lemma = triplet["lemma"]
        tags = set(triplet["unimorph_tags"].split(";"))
        for tag in tags:
            alltags[tag] = alltags.get(tag, 0) + 1

    return alltags


def print_knowns():
    alltags = {}
    unknown_pairs = []
    for index, row in tqdm(list(language_info.iterrows())):
        for tag, occurrences in get_tags(row["Name"]).items():
            alltags[tag] = alltags.get(tag, 0) + occurrences
            if tag not in KNOWN_TAGS:
                unknown_pairs.append((row["Name"], tag))

    for pair in sorted(unknown_pairs, key=lambda x: x[1]):
        print(*pair)

    print()

    for tag, occurrences in sorted(list(alltags.items()), key=lambda x: x[1]):
        if tag not in KNOWN_TAGS:
            print(tag, occurrences)

    print()

    for tag in KNOWN_TAGS:
        if tag not in alltags:
            print(tag, 0)

    for tag, occurrences in sorted(list(alltags.items()), key=lambda x: x[1]):
        if tag in KNOWN_TAGS:
            print(tag, occurrences)

    print("{\"" + '", "'.join(sorted([tag for tag in alltags if tag.startswith("IN+")])) + "\"}")
