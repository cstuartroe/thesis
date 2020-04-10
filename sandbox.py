import sys

from crosslingual_inflection_baseline.src import train

languages = ["spanish", "portuguese", "turkish", "azeri", "finnish", "estonian", "georgian", "navajo"]

source_lang = sys.argv[1] if len(sys.argv) > 1 else None

mt = train.Multitrainer(langs=set(languages), pretrain_epochs=200, train_epochs=100)

for target_lang in languages:
    if source_lang != target_lang:
        mt.train_pair(source_lang, target_lang)
