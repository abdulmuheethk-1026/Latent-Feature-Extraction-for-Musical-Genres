"""Builds notebooks/kaggle_genre_extraction.ipynb (self-contained: it writes every source file with %%writefile)."""
import json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KAGGLE_DIR = "/kaggle/working/genre-extraction-repro"
FILES = (["src/genre/" + f for f in ("__init__.py", "config.py", "utils.py", "data_paper.py", "data_mel.py", "evaluate.py",
                                     "models_paper.py", "models_improved.py", "training.py", "experiment.py")]
         + ["scripts/01_reproduce_paper.py", "scripts/02_train_improved.py", "scripts/03_tune_optuna.py", "scripts/04_demo.py"])


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}


cells = [md("""
# Genre-Extraction on Kaggle — reproduction (Obj. 1) + improved model & tuning (Obj. 2)
**Before running:** *Add Data* → "GTZAN Dataset - Music Genre Classification"; Accelerator → GPU.
Sections: 0 setup · 1 write source files · 2 smoke test · 3 reproduce paper · 4 improved models · 5 tuning · 6 results/zip.
"""), code("""
import os, sys, subprocess
os.makedirs("/kaggle/working/genre-extraction-repro/src/genre", exist_ok=True)
os.makedirs("/kaggle/working/genre-extraction-repro/scripts", exist_ok=True)
import tensorflow as tf
print("TF", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
try:
    import optuna
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"])   # needs Internet ON
""")]
for rel in FILES:
    src = open(os.path.join(ROOT, rel)).read()
    cells.append(code(f"%%writefile {KAGGLE_DIR}/{rel}\n{src}"))

RUN = f"cd {KAGGLE_DIR} && python"
cells += [
    md("## 2. Smoke test (≈ 1–2 min; also reveals the GTZAN folder that was auto-detected)"),
    code(f"!{RUN} scripts/01_reproduce_paper.py --quick --out outputs/smoke"),
    md("## 3. Objective 1 — reproduce the paper\n`random_segment` = original (leaky) protocol; `song` = same models, honest split."),
    code(f"!{RUN} scripts/01_reproduce_paper.py --split random_segment --out outputs/paper_leaky"),
    code(f"!{RUN} scripts/01_reproduce_paper.py --split song --out outputs/paper_song"),
    md("## 4. Objective 2 — better front-end (log-mel) + better models, song-level split"),
    code(f"!{RUN} scripts/02_train_improved.py --model cnn --out outputs/cnn"),
    code(f"!{RUN} scripts/02_train_improved.py --model conv_dsae --out outputs/conv_dsae"),
    md("## 5. Objective 2 — hyper-parameter tuning (Optuna)\nAbout 20–40 min on a T4 for 20 trials. Lower `--trials` if short on time."),
    code(f"!{RUN} scripts/03_tune_optuna.py --model conv_dsae --trials 20 --epochs 30 --out outputs/tune"),
    code(f"!{RUN} scripts/02_train_improved.py --model conv_dsae --params outputs/tune/best_params.json --epochs 80 --out outputs/conv_dsae_tuned"),
    md("## 6. Results"),
    code(f"""
import json, glob
from IPython.display import Image, display
for f in sorted(glob.glob("{KAGGLE_DIR}/outputs/*/metrics.json")):
    m = json.load(open(f)); t = m.get("test") or m.get("table1")
    print(f.split("outputs/")[1], "->", json.dumps(t)[:300])
for p in ["paper_leaky/fig4_confusion_dsae_test.png", "paper_leaky/fig5b_pca_dsae_64.png",
          "conv_dsae/confusion_test.png", "conv_dsae/pca_embedding_test.png"]:
    fp = "{KAGGLE_DIR}/outputs/" + p
    if os.path.exists(fp): display(Image(fp))
"""),
    code(f"""
import shutil
shutil.make_archive("/kaggle/working/genre_outputs", "zip", "{KAGGLE_DIR}/outputs")   # download from the Output tab
"""),
]
cells += [
    md("## 7. Live demo for your teacher (needs outputs/cnn/cnn.weights.h5 from section 4; takes seconds)\n"
       "Re-computes accuracy on the 60 unseen test songs, shows individual songs, and can classify YOUR audio file."),
    code(f"!{RUN} scripts/04_demo.py --weights outputs/cnn/cnn.weights.h5 --n 12"),
    code(f"# put your own .wav in the notebook (Add Input / Upload), then edit the path below\n"
         f"!{RUN} scripts/04_demo.py --weights outputs/cnn/cnn.weights.h5 --n 0 --audio /kaggle/input/YOUR_DATASET/my_song.wav"),
    code(f"import glob\nfrom IPython.display import Image, display\n"
         f"for f in sorted(glob.glob('{KAGGLE_DIR}/outputs/demo/*.png')): display(Image(f))"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "kaggle_genre_extraction.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "cells:", len(cells))
