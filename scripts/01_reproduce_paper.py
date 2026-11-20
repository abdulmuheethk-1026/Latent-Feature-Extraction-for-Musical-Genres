"""Reproduce the paper / repo: baseline NN, vanilla AE (400 ep), Deep Softmax Autoencoder (200 ep), Tables 1-2, Figs 1,4,5,6.

  python scripts/01_reproduce_paper.py --data-root auto --out outputs/paper
  python scripts/01_reproduce_paper.py --split song        # same models, leak-free song-level split (honest numbers)
  python scripts/01_reproduce_paper.py --quick             # 3-epoch smoke test
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import tensorflow as tf

from genre.data_paper import prepare_paper
from genre.evaluate import plot_confusion, plot_history, plot_pca, save_json, summarize
from genre.models_paper import build_baseline_nn, build_dsae, build_vanilla_ae
from genre.training import predict_logits
from genre.utils import find_gtzan_root, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="auto")
    ap.add_argument("--out", default="outputs/paper")
    ap.add_argument("--split", default="random_segment", choices=["random_segment", "song"])
    ap.add_argument("--epochs", type=int, default=200)       # baseline + DSAE (repo: num_epochs=200)
    ap.add_argument("--ae-epochs", type=int, default=400)    # vanilla AE (repo encoder.py: num_epochs=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--cls-weight", type=float, default=0.1)  # repo: classificationweight = 0.1
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epochs = a.ae_epochs = 3
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)

    root = find_gtzan_root(a.data_root)
    print("GTZAN root:", root)
    splits = prepare_paper(root, cache=os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "cache", "paper_500d.npz"),
                           split=a.split, seed=a.seed)
    (Xtr, ytr, _), (Xva, yva, sva), (Xte, yte, ste) = splits["train"], splits["val"], splits["test"]
    perm = np.random.RandomState(a.seed).permutation(len(Xtr))        # shuffle ONCE like the repo (fit(shuffle=False))
    Xtr, ytr = Xtr[perm], ytr[perm]
    common = dict(batch_size=a.batch, epochs=a.epochs, shuffle=False, verbose=0)
    results, hists = {}, {}

    # 1) two-layer baseline (tfsoftmax.py)
    base = build_baseline_nn()
    h = base.fit(Xtr, ytr, validation_data=(Xva, yva), **common)
    hists["baseline"] = h.history

    # 2) vanilla autoencoder (encoder.py)
    vae = build_vanilla_ae()
    h = vae.fit(Xtr, Xtr, validation_data=(Xva, Xva), batch_size=a.batch, epochs=a.ae_epochs, shuffle=False, verbose=0)
    hists["vanilla_ae"] = h.history

    # 3) Deep Softmax Autoencoder (softmaxautoencoder.py)
    dsae = build_dsae(cls_weight=a.cls_weight)
    h = dsae.fit(Xtr, {"recon": Xtr, "logits": ytr}, validation_data=(Xva, {"recon": Xva, "logits": yva}), **common)
    hists["dsae"] = {k: v for k, v in h.history.items() if "loss" in k or "accuracy" in k}

    # ---- Table 1: accuracies
    acc = lambda m, X, y: float((predict_logits(m, X).argmax(1) == y).mean())
    table1 = {"two_layer_nn": {s: acc(base, X, y) for s, (X, y) in {"train": (Xtr, ytr), "dev": (Xva, yva), "test": (Xte, yte)}.items()},
              "deep_softmax_ae": {s: acc(dsae, X, y) for s, (X, y) in {"train": (Xtr, ytr), "dev": (Xva, yva), "test": (Xte, yte)}.items()}}
    print("\nTable 1 (classification accuracy)  paper: NN 52.0/38.1/36.4  DSAE 94.9/64.1/65.3")
    for k, v in table1.items():
        print(f"  {k:18s} train {v['train']:.3f}  dev {v['dev']:.3f}  test {v['test']:.3f}")

    # ---- Table 2 + Fig 4 : per-class metrics, confusion matrix on test
    res, pred = summarize(predict_logits(dsae, Xte), yte, ste, name="Deep Softmax AE / test (Table 2)")
    plot_confusion(yte, pred, os.path.join(a.out, "fig4_confusion_dsae_test.png"))
    results.update(table1=table1, table2_dsae_test=res)

    # ---- Figs 1,5,6: PCA visualisations on dev+test (2000 examples, as in the repo's *_dev.csv)
    Xv, yv = np.concatenate([Xva, Xte]), np.concatenate([yva, yte])
    plot_pca(Xv, yv, "PCA of 500-Dim Raw Audio Data", os.path.join(a.out, "fig1_pca_raw.png"))
    plot_pca(tf.keras.Model(vae.input, vae.get_layer("embedding").output).predict(Xv, verbose=0), yv,
             "PCA of 64-Dim Vanilla Autoencoder Bottleneck", os.path.join(a.out, "fig5a_pca_vanilla_ae.png"))
    plot_pca(tf.keras.Model(dsae.input, dsae.get_layer("embedding").output).predict(Xv, verbose=0), yv,
             "PCA of 64-Dim Deep Softmax AE Bottleneck", os.path.join(a.out, "fig5b_pca_dsae_64.png"))
    plot_pca(predict_logits(dsae, Xv), yv, "PCA of 4-Dim Deep Softmax AE Logits", os.path.join(a.out, "fig6b_pca_dsae_logits.png"))
    plot_pca(predict_logits(base, Xv), yv, "PCA of 4-Dim Two Layer NN Logits", os.path.join(a.out, "fig6a_pca_nn_logits.png"))
    plot_history(hists["dsae"], os.path.join(a.out, "fig3_dsae_curves.png"), "Deep Softmax AE")

    save_json(results, os.path.join(a.out, "metrics.json"))
    print("\nSaved figures + metrics.json to", a.out)


if __name__ == "__main__":
    main()
