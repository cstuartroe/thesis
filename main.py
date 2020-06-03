import sys

from code import WALS, unimorph, calculate, graphing, alignment, taginfo, fusion, modeling
from crosslingual_inflection_baseline.src import train

DISPATCH = {
    "WALS": WALS.pull,
    "unimorph": unimorph.pull,
    "align": alignment.pull,
    "fusion": fusion.pull,
    "knowntags": taginfo.print_knowns,
    "calculate": calculate.calculate,
    "graph": graphing.graph,
    "model": modeling.main,
    "flush": modeling.flush,
    "stats": calculate.extra_stats
}


def all():
    WALS.pull()
    unimorph.pull()
    alignment.pull()
    fusion.pull()
    calculate.calculate()
    # graphing.graph()


DISPATCH["all"] = all


if __name__ == "__main__":
    for command in sys.argv[1:]:
        if command in DISPATCH:
            DISPATCH[command]()
        else:
            print(f"Unknown command: {command}")

    if len(sys.argv) == 1:
        print("Usage: python main.py <command>")
        print("The available commands are:")
        for command in DISPATCH.keys():
            print("    " + command)
