import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA

from .config import GENRES, GENRE_COLORS


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def song_level(probs, y, song):
    """Average segment probabilities per song -> one prediction per song."""
    ids = np.unique(song)
    P = np.stack([probs[song == i].mean(axis=0) for i in ids])
    T = np.array([y[song == i][0] for i in ids])
    return T, P.argmax(1)


def summarize(logits, y, song, genres=GENRES, name=""):
    probs = softmax(logits)
    pred = probs.argmax(1)
    st, sp = song_level(probs, y, song)
    rep = classification_report(y, pred, target_names=genres, output_dict=True, zero_division=0)
    res = {"segment_acc": float(accuracy_score(y, pred)),
           "segment_macro_f1": float(f1_score(y, pred, average="macro")),
           "song_acc": float(accuracy_score(st, sp)),
           "n_segments": int(len(y)), "n_songs": int(len(st)),
           "per_class": {g: {k: float(rep[g][k]) for k in ("precision", "recall", "f1-score")} for g in genres}}
    print(f"\n== {name} ==  segment acc {res['segment_acc']:.3f} | macro-F1 {res['segment_macro_f1']:.3f} | "
          f"song acc {res['song_acc']:.3f} ({res['n_songs']} songs)")
    print(classification_report(y, pred, target_names=genres, digits=3, zero_division=0))
    return res, pred


def plot_confusion(y, pred, path, title="Normalized confusion matrix", genres=GENRES):
    cm = confusion_matrix(y, pred, labels=range(len(genres))).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=45)
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)
    for i in range(len(genres)):
        for j in range(len(genres)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pca(Z, y, title, path, genres=GENRES):
    Z2 = PCA(n_components=2).fit_transform(Z.reshape(len(Z), -1))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for k in reversed(range(len(genres))):                      # same draw order as original plots
        m = y == k
        ax.scatter(Z2[m, 0], Z2[m, 1], c=GENRE_COLORS[k], alpha=.5, edgecolor="k", s=18, label=genres[k].capitalize())
    ax.set_title(title); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_history(hist, path, title="Training curves"):
    keys = [k for k in hist if hist[k]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in keys:
        (axes[0] if "loss" in k else axes[1]).plot(hist[k], label=k)
    axes[0].set_title("loss"); axes[1].set_title("accuracy")
    for a in axes:
        a.legend(fontsize=7); a.set_xlabel("epoch")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
