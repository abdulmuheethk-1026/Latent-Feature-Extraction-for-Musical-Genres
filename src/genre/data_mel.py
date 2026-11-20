"""Improved front-end: log-mel spectrogram segments (3 s, 50 % overlap) instead of 551 Hz average-pooled waveform."""
import os
import warnings
import numpy as np

from .config import GENRES
from .utils import list_wavs, make_splits, leakage

SR, SEG_SEC, HOP_SEC = 22050, 3.0, 1.5
N_FFT, HOP, N_MELS, N_FRAMES = 1024, 512, 64, 128


def song_to_segments(path, librosa):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=30.0)
    seg, step = int(SR * SEG_SEC), int(SR * HOP_SEC)
    out = []
    for start in range(0, max(len(y) - seg, 0) + 1, step):
        chunk = y[start:start + seg]
        if len(chunk) < seg:
            chunk = np.pad(chunk, (0, seg - len(chunk)))
        m = librosa.feature.melspectrogram(y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=20)
        m = librosa.power_to_db(m, ref=1.0, top_db=80.0)      # absolute level kept (loudness is a genre cue)
        out.append(m[:, :N_FRAMES].astype(np.float32))        # (64, 130) -> (64, 128)
    return out


def load_mel_dataset(root, genres=GENRES, cache=None):
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if d["X"].size:                                   # ignore empty/corrupt caches
            return d["X"], d["y"], d["song"], list(d["names"])
    import librosa
    X, y, song, names = [], [], [], []
    for gi, g in enumerate(genres):
        for path in list_wavs(root, g):
            try:
                segs = song_to_segments(path, librosa)
            except Exception as e:
                warnings.warn(f"skipping unreadable file {path}: {e}")
                continue
            sid = len(names)
            names.append(os.path.basename(path))
            for s in segs:
                X.append(s)
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


def prepare_mel(root, cache=None, split="song", seed=42, n_folds=0, fold=0):
    X, y, song, names = load_mel_dataset(root, cache=cache)
    splits = make_splits(X, y, song, mode=split, seed=seed, n_folds=n_folds, fold=fold)
    mu, sd = splits["train"][0].mean(), splits["train"][0].std() + 1e-6      # train-only statistics
    splits = {k: (((a - mu) / sd)[..., None].astype(np.float32), yy, ss) for k, (a, yy, ss) in splits.items()}
    print(f"[data] {len(names)} songs -> {len(X)} segments | split={split} | "
          f"test-segments whose song is also in train: {leakage(splits):.1%}")
    return splits
