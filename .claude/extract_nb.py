import json
import sys

path = sys.argv[1]
out = sys.argv[2]
nb = json.load(open(path, encoding="utf-8"))
with open(out, "w", encoding="utf-8") as f:
    for i, cell in enumerate(nb["cells"]):
        f.write(f"##### CELL {i} [{cell['cell_type']}] #####\n")
        f.write("".join(cell["source"]))
        f.write("\n\n")
print("done", len(nb["cells"]), "cells")
