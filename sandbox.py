import sys

from crosslingual_inflection_baseline.src import train

languages = ["spanish", "portuguese", "turkish", "azeri", "finnish", "estonian", "georgian", "navajo"]

source_lang = sys.argv[1] if len(sys.argv) > 1 else None

for target_lang in languages:
    if source_lang != target_lang:
        train.main(source_lang, target_lang, pretrain_epochs=1, train_epochs=1)
