"""Exact preprocessing of the original repo (softmaxautoencoder.py::load_data).

For each 30 s GTZAN clip: librosa.load (22.05 kHz, mono) -> first 600 000 samples -> reshape(15000, 40)
-> mean over the 40 axis (average pooling, pool=40 => 'artificial 551 Hz') -> 20 consecutive
500-sample segments. 400 songs x 20 = 8000 examples in R^500.
"""
import os
import warnings
import numpy as np

from .config import GENRES
from .utils import list_wavs, make_splits, leakage

N_SAMPLES, POOL, SEG_LEN, N_SEG = 600_000, 40, 500, 20


def load_paper_dataset(root, genres=GENRES, cache=None):
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if d["X"].size:                                   # ignore empty/corrupt caches
            return d["X"], d["y"], d["song"], list(d["names"])
    import librosa
    X, y, song, names = [], [], [], []
    for gi, g in enumerate(genres):
        for path in list_wavs(root, g):
            try:
                audio, _ = librosa.load(path)               # defaults sr=22050, mono (as original)
            except Exception as e:                           # e.g. jazz.00054.wav is corrupt in the Kaggle copy
                warnings.warn(f"skipping unreadable file {path}: {e}")
                continue
            audio = audio[:N_SAMPLES]
            if len(audio) < N_SAMPLES:
                audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
            pooled = audio.reshape(N_SAMPLES // POOL, POOL).mean(axis=1)
            sid = len(names)
            names.append(os.path.basename(path))
            for j in range(N_SEG):
                X.append(pooled[SEG_LEN * j: SEG_LEN * (j + 1)])
                y.append(gi)
                song.append(sid)
    if not names:
        raise RuntimeError(f"No .wav files could be read under {root}. Is this the genres_original folder?")
    X = np.asarray(X, dtype=np.float32)
    y, song = np.asarray(y, dtype=np.int32), np.asarray(song, dtype=np.int32)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, X=X, y=y, song=song, names=np.asarray(names))
    return X, y, song, names


def prepare_paper(root, cache=None, split="random_segment", seed=42, n_folds=0, fold=0):
    X, y, song, names = load_paper_dataset(root, cache=cache)
    splits = make_splits(X, y, song, mode=split, seed=seed, n_folds=n_folds, fold=fold)
    print(f"[data] {len(names)} songs -> {len(X)} segments | split={split} | "
          f"test-segments whose song is also in train: {leakage(splits):.1%}")
    return splits
