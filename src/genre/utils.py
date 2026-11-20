import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import GENRES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def find_gtzan_root(explicit=None, genres=GENRES):
    """Return the folder that directly contains <genre>/ sub-folders full of .wav files (any nesting depth)."""
    def ok(d):
        # folders must exist AND contain .wav files (this dataset also ships Data/images_original full of PNGs)
        return all(os.path.isdir(os.path.join(d, g)) and
                   any(f.lower().endswith(".wav") for f in os.listdir(os.path.join(d, g))) for g in genres)

    if explicit and explicit != "auto":
        if ok(explicit):
            return explicit
        raise FileNotFoundError(f"{explicit} does not contain the folders {genres}")
    seen = []
    for base in ("/kaggle/input", "./genres", "./data", "."):
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if ok(dirpath):                      # check BEFORE anything else; no depth limit
                return dirpath
            if len(seen) < 25 and dirpath.count(os.sep) <= 6:
                seen.append(dirpath)
            if "outputs" in dirnames:            # never descend into our own results
                dirnames.remove("outputs")
    raise FileNotFoundError(
        f"No folder containing {genres} found. Folders seen (first 25):\n  " + "\n  ".join(seen) +
        "\nPass --data-root <folder that contains classical/, jazz/, metal/, pop/>")


def list_wavs(root, genre):
    d = os.path.join(root, genre)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".wav")]


def song_split(song_genre, val_frac=0.15, test_frac=0.15, seed=42, n_folds=0, fold=0):
    """Stratified split on SONGS (never on segments). Returns (train, val, test) song indices."""
    idx = np.arange(len(song_genre))
    song_genre = np.asarray(song_genre)
    if n_folds and n_folds > 1:
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        rest, test = list(skf.split(idx, song_genre))[fold]
        tr, va = train_test_split(rest, test_size=val_frac, stratify=song_genre[rest], random_state=seed)
    else:
        rest, test = train_test_split(idx, test_size=test_frac, stratify=song_genre, random_state=seed)
        tr, va = train_test_split(rest, test_size=val_frac / (1 - test_frac),
                                  stratify=song_genre[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(test)


def make_splits(X, y, song, mode="song", seed=42, n_folds=0, fold=0):
    """mode='random_segment' -> what the original repo did (leaky); mode='song' -> leak-free."""
    X, y, song = np.asarray(X), np.asarray(y), np.asarray(song)
    n = len(X)
    if mode == "random_segment":
        perm = np.random.RandomState(seed).permutation(n)
        a, b = int(0.75 * n), int(0.875 * n)          # 6000 / 1000 / 1000 for n = 8000
        parts = {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}
    elif mode == "song":
        n_songs = song.max() + 1
        sg = np.zeros(n_songs, dtype=int)
        sg[song] = y
        tr, va, te = song_split(sg, seed=seed, n_folds=n_folds, fold=fold)
        parts = {k: np.where(np.isin(song, s))[0] for k, s in zip(("train", "val", "test"), (tr, va, te))}
    else:
        raise ValueError(mode)
    return {k: (X[i], y[i], song[i]) for k, i in parts.items()}


def leakage(splits):
    """Fraction of test segments whose parent song also appears in the training set."""
    tr_songs = np.unique(splits["train"][2])
    return float(np.isin(splits["test"][2], tr_songs).mean())
