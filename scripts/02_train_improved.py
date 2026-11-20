"""Train the improved models on log-mel spectrograms with a leak-free SONG-level split.

  python scripts/02_train_improved.py --model cnn --out outputs/cnn
  python scripts/02_train_improved.py --model conv_dsae --out outputs/conv_dsae
  python scripts/02_train_improved.py --model conv_dsae --folds 5 --fold 0      # 5-fold CV (run folds 0..4)
  python scripts/02_train_improved.py --model conv_dsae --params outputs/tune/best_params.json   # tuned
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np

from genre.data_mel import prepare_mel
from genre.evaluate import plot_confusion, plot_history, plot_pca, save_json
from genre.experiment import embeddings, train_eval
from genre.utils import find_gtzan_root, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="conv_dsae", choices=["cnn", "conv_dsae"])
    ap.add_argument("--data-root", default="auto")
    ap.add_argument("--out", default="outputs/improved")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--recon-weight", type=float, default=1.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--params", default=None, help="JSON from 03_tune_optuna.py (overrides the flags above)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epochs = 2
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)

    p = dict(lr=a.lr, base=a.base, latent=a.latent, dropout=a.dropout)
    if a.model == "conv_dsae":
        p.update(cls_weight=a.cls_weight, recon_weight=a.recon_weight)
    batch, augment = a.batch, not a.no_augment
    if a.params:
        tuned = json.load(open(a.params))
        batch, augment = tuned.pop("batch", batch), tuned.pop("spec_aug", augment)
        p.update({k: v for k, v in tuned.items() if k in ("lr", "base", "latent", "dropout", "cls_weight", "recon_weight")})
    print("hyper-parameters:", p, "| batch", batch, "| spec_aug", augment)

    root = find_gtzan_root(a.data_root)
    cache = os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "cache", "mel64x128.npz")
    data = prepare_mel(root, cache=cache, split="song", seed=a.seed, n_folds=a.folds, fold=a.fold)

    r = train_eval(a.model, p, data, epochs=a.epochs, batch=batch, augment=augment, patience=a.patience, seed=a.seed, verbose=2)
    Xte, yte, _ = data["test"]
    plot_confusion(yte, r["test_pred"], os.path.join(a.out, "confusion_test.png"), f"{a.model}: test (song-level split)")
    plot_pca(embeddings(r["model"], Xte), yte, f"PCA of 64-d {a.model} embedding (test songs)", os.path.join(a.out, "pca_embedding_test.png"))
    plot_history(r["history"], os.path.join(a.out, "curves.png"), a.model)
    save_json({"model": a.model, "params": p, "batch": batch, "spec_aug": augment, "fold": a.fold, "folds": a.folds,
               "val": r["val"], "test": r["test"]}, os.path.join(a.out, "metrics.json"))
    try:
        r["model"].save_weights(os.path.join(a.out, f"{a.model}.weights.h5"))
    except Exception as e:
        print("could not save weights:", e)
    print(f"\nFINAL TEST  segment acc {r['test']['segment_acc']:.3f} | song acc {r['test']['song_acc']:.3f}")


if __name__ == "__main__":
    main()
