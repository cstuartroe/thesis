import sys
import os
import re

from .general import flatten
from crosslingual_inflection_baseline.src import train, sigmorphon19_task1_decode

language_groups = [
    ["turkish", "bashkir", "crimean-tatar", "uzbek"],
    ["spanish", "portuguese", "italian", "romanian"],
    ["finnish", "estonian"],
    ["georgian"],
    ["navajo"],
    ["arabic", "hebrew"],
    ["danish", "swedish"],
    ["czech", "slovak"],
    ["quechua"],
    ["zulu"],
    ["basque"]
]


def get_target_languages(source_language):
    if source_language is None:
        yield from flatten(language_groups)

    else:
        source_group_index = None
        source_language_index = None
        source_group_size = None
        for group_index, group in enumerate(language_groups):
            if source_language in group:
                source_group_index = group_index
                source_language_index = group.index(source_language)
                source_group_size = len(group)

        for group_index, group in enumerate(language_groups):
            for language_index, language in enumerate(group):
                if ((group_index - source_group_index + language_index - source_language_index) % source_group_size == 0) != (group_index == source_group_index):
                    yield language


ARCH = "hard"

source_langs = [(None if lang == "none" else lang) for lang in sys.argv[2:]]
if "all" in source_langs:
    source_langs = flatten(language_groups)


def main():
    mt = train.Multitrainer(langs=set(flatten(language_groups)), pretrain_epochs=10, train_epochs=20, arch=ARCH)

    for lang in flatten(language_groups):
        for suffix in ["train-high", "train-medium", "dev", "test"]:
            f = f"conll2018/task1/all/{lang}-{suffix}"
            if not os.path.exists(f):
                print(f, "does not exist")

    pairs = []
    for source_lang in source_langs:
        for target_lang in get_target_languages(source_lang):
            mt.train_pair(source_lang, target_lang)
            sigmorphon19_task1_decode.main(ARCH, source_lang, target_lang, test=False)
            correct_guesses, total_guesses = sigmorphon19_task1_decode.main(ARCH, source_lang, target_lang)
            pairs.append((source_lang, target_lang, correct_guesses, total_guesses))

    for source_lang, target_lang, correct_guesses, total_guesses in pairs:
        print(f"{source_lang} -> {target_lang}: {correct_guesses}/{total_guesses} ({round(correct_guesses*100/total_guesses)}%)")


def flush():
    for tagdir in os.listdir("model"):
        for pairdir in os.listdir(os.path.join("model", tagdir)):
            model_files = [f for f in os.listdir(os.path.join("model", tagdir, pairdir))
                           if re.fullmatch(r"model.nll_.+epoch_\d+", f)]
            model_files.sort(key=lambda f: int(f.split("_")[-1]))
            for model in model_files[:-1]:
                os.remove(os.path.join("model", tagdir, pairdir, model))