"""Builds notebooks/kaggle_demo_only.ipynb: a tiny, self-contained notebook that ONLY runs the demo (seconds, no training)."""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = "/kaggle/working/genre-demo"
FILES = ["src/genre/__init__.py", "src/genre/config.py", "src/genre/utils.py", "src/genre/data_mel.py",
         "src/genre/evaluate.py", "src/genre/models_improved.py", "scripts/04_demo.py"]
md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
code = lambda t: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}
cells = [md("""
# Genre-Extraction: demo only (seconds, no training)
**Before running** (this is what makes it work):
1. *Add Input* → **Your Work / Notebook Output** → pick the saved version of your training notebook (it contains `cnn.weights.h5`, `mel64x128.npz`, `metrics.json`), **or** attach a Dataset where you uploaded those files.
2. Accelerator can stay on *None* (CPU is enough). The GTZAN audio dataset is **not** needed: the feature cache is used.
3. Run all cells. The script finds the files by itself, so no paths to type.
"""), code(f"""
import os
os.makedirs("{D}/src/genre", exist_ok=True); os.makedirs("{D}/scripts", exist_ok=True)
""")]
for rel in FILES:
    cells.append(code(f"%%writefile {D}/{rel}\n" + open(os.path.join(ROOT, rel)).read()))
cells += [
    md("## Files found (if this prints nothing, attach the training notebook's output as an Input)"),
    code('!find /kaggle/input /kaggle/working -name "cnn.weights.h5" -o -name "mel64x128.npz" -o -name "metrics.json" 2>/dev/null | head -20'),
    md("## 1. Recompute accuracy on the 60 unseen test songs and list songs (`--n 60` lists all of them)"),
    code(f"!cd {D} && python scripts/04_demo.py --n 12"),
    md("## 2. Your own audio (edit the path: Add Input / Upload a .wav, then use the path shown in the sidebar)"),
    code(f"!cd {D} && python scripts/04_demo.py --n 0 --audio /kaggle/input/YOUR_DATASET/my_song.wav"),
    code(f"import glob\nfrom IPython.display import Image, display\nfor f in sorted(glob.glob('{D}/outputs/demo/*.png')): display(Image(f))"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "kaggle_demo_only.ipynb")
json.dump(nb, open(out, "w"), indent=1); print("wrote", out, "cells:", len(cells))
