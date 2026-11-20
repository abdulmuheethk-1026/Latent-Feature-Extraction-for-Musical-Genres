"""Live demo of the trained mel-CNN (no training, takes seconds once features are cached).

  python scripts/04_demo.py --weights outputs/cnn/cnn.weights.h5 --n 10
  python scripts/04_demo.py --weights outputs/cnn/cnn.weights.h5 --audio my_song.wav other.wav

What it does
 1. Re-creates the exact song-level split (seed 42) and the train-only standardisation.
 2. Loads the weights and RE-COMPUTES accuracy on the 60 unseen test songs; compares with the saved metrics.json.
 3. Shows N random test songs: true genre, predicted genre, confidence.
 4. For every --audio file: predicts its genre and saves a picture (spectrogram + probabilities).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genre.config import GENRES
from genre.data_mel import load_mel_dataset, song_to_segments
from genre.evaluate import softmax
from genre.utils import find_gtzan_root, make_splits


def locate(path, filename):
    """Return `path` if it exists, otherwise search /kaggle/input, /kaggle/working and . for `filename`."""
    if path and os.path.exists(path):
        return path
    for base in ("/kaggle/input", "/kaggle/working", "."):
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            if filename in files:
                found = os.path.join(dirpath, filename)
                print(f"[locate] {filename} -> {found}")
                return found
    raise FileNotFoundError(f"{filename} not found. Attach the notebook output / dataset that contains it "
                            f"(Add Input), or train it first (scripts/02_train_improved.py --model cnn).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/cnn/cnn.weights.h5")
    ap.add_argument("--data-root", default="auto")
    ap.add_argument("--cache", default="outputs/cache/mel64x128.npz")
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 1) same data, same split, same normalisation as training
    a.weights = locate(a.weights, os.path.basename(a.weights))
    cache = a.cache
    if not os.path.exists(cache):
        try:
            cache = locate(cache, os.path.basename(cache))
        except FileNotFoundError:
            pass                                   # fall back to rebuilding features from the audio
    root = None if os.path.exists(cache) else find_gtzan_root(a.data_root)   # audio dataset only needed without a cache
    X, y, song, names = load_mel_dataset(root, cache=cache)
    sp = make_splits(X, y, song, mode="song", seed=a.seed)
    mu, sd = sp["train"][0].mean(), sp["train"][0].std() + 1e-6
    Xte, yte, ste = sp["test"]
    Xte = ((Xte - mu) / sd)[..., None].astype(np.float32)

    # 2) model
    from genre.models_improved import build_cnn
    model = build_cnn((64, 128, 1))
    model.load_weights(a.weights)
    print(f"loaded {a.weights}\n")

    probs = softmax(model.predict(Xte, batch_size=256, verbose=0))
    seg_acc = float((probs.argmax(1) == yte).mean())
    ids = np.unique(ste)
    P = np.stack([probs[ste == i].mean(0) for i in ids])
    T = np.array([yte[ste == i][0] for i in ids])
    song_acc = float((P.argmax(1) == T).mean())
    print(f"Unseen TEST songs: {len(ids)} | segment accuracy {seg_acc:.3f} | song accuracy {song_acc:.3f}")
    mj = os.path.join(os.path.dirname(a.weights), "metrics.json")
    if os.path.exists(mj):
        saved = json.load(open(mj))["test"]
        print(f"saved in metrics.json:             segment accuracy {saved['segment_acc']:.3f} | song accuracy {saved['song_acc']:.3f}")

    # 3) a few songs, one by one
    rng = np.random.RandomState(0)
    pick = rng.choice(len(ids), size=min(a.n, len(ids)), replace=False)
    print(f"\n{'song file':<22}{'true':<11}{'predicted':<11}{'confidence':<11}")
    for k in pick:
        p = P[k]
        mark = "OK " if p.argmax() == T[k] else "WRONG"
        print(f"{names[ids[k]]:<22}{GENRES[T[k]]:<11}{GENRES[p.argmax()]:<11}{p.max():<11.1%}{mark}")

    # 4) your own audio
    for path in a.audio:
        segs = np.stack(song_to_segments(path, __import__("librosa")))
        x = ((segs - mu) / sd)[..., None].astype(np.float32)
        p = softmax(model.predict(x, verbose=0)).mean(0)
        print(f"\n{os.path.basename(path)}: {len(segs)} windows -> {GENRES[p.argmax()]} ({p.max():.1%})  | " +
              ", ".join(f"{g} {v:.1%}" for g, v in zip(GENRES, p)))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].imshow(segs[0], origin="lower", aspect="auto"); ax[0].set_title("log-mel spectrogram (first 3 s)")
        ax[0].set_xlabel("time frames"); ax[0].set_ylabel("mel band")
        ax[1].bar(GENRES, p); ax[1].set_ylim(0, 1); ax[1].set_title(f"prediction: {GENRES[p.argmax()]} ({p.max():.0%})")
        fig.tight_layout()
        fp = os.path.join(a.out, "demo_" + os.path.splitext(os.path.basename(path))[0] + ".png")
        fig.savefig(fp, dpi=120); plt.close(fig)
        print("saved", fp)


if __name__ == "__main__":
    main()
