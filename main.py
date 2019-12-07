import sys

from code import WALS, unimorph, calculate, graphing

DISPATCH = {
    "WALS": WALS.pull,
    "unimorph": unimorph.pull,
    "calculate": calculate.calculate,
    "graph": graphing.graph
}

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