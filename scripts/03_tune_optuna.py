"""Hyper-parameter search (Optuna / TPE) with a leak-free song-level split. Objective = validation SONG accuracy.

  python scripts/03_tune_optuna.py --model conv_dsae --trials 20 --epochs 30
  python scripts/03_tune_optuna.py --model cnn       --trials 20 --epochs 30
  python scripts/03_tune_optuna.py --model paper_dsae --trials 30 --epochs 120   # tunes the ORIGINAL 500-d MLP model
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
try:
    import optuna
except ImportError:
    sys.exit("pip install optuna")

from genre.data_mel import prepare_mel
from genre.data_paper import prepare_paper
from genre.evaluate import save_json
from genre.experiment import train_eval
from genre.utils import find_gtzan_root, set_seed


def suggest(trial, kind):
    if kind in ("cnn", "conv_dsae"):
        p = dict(lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
                 base=trial.suggest_categorical("base", [16, 32, 48]),
                 latent=trial.suggest_categorical("latent", [32, 64, 128]),
                 dropout=trial.suggest_float("dropout", 0.1, 0.5))
        if kind == "conv_dsae":
            p["cls_weight"] = trial.suggest_float("cls_weight", 0.1, 5.0, log=True)
        extra = dict(batch=trial.suggest_categorical("batch", [32, 64, 128]),
                     spec_aug=trial.suggest_categorical("spec_aug", [True, False]))
    else:   # paper_dsae (500-d raw-audio MLP)
        p = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                 latent=trial.suggest_categorical("latent", [32, 64, 128]),
                 dropout=trial.suggest_float("dropout", 0.0, 0.5),
                 cls_weight=trial.suggest_float("cls_weight", 0.05, 5.0, log=True),
                 recon_activation=trial.suggest_categorical("recon_activation", ["relu", "linear"]))
        extra = dict(batch=trial.suggest_categorical("batch", [128, 256, 512]), spec_aug=False)
    return p, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="conv_dsae", choices=["cnn", "conv_dsae", "paper_dsae"])
    ap.add_argument("--data-root", default="auto")
    ap.add_argument("--out", default="outputs/tune")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--final-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    root = find_gtzan_root(a.data_root)
    cdir = os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "cache")
    if a.model == "paper_dsae":
        data = prepare_paper(root, cache=os.path.join(cdir, "paper_500d.npz"), split="song", seed=a.seed)
    else:
        data = prepare_mel(root, cache=os.path.join(cdir, "mel64x128.npz"), split="song", seed=a.seed)

    def objective(trial):
        p, extra = suggest(trial, a.model)
        r = train_eval(a.model, p, data, epochs=a.epochs, batch=extra["batch"], augment=extra["spec_aug"],
                       patience=a.patience, seed=a.seed, eval_test=False)
        trial.set_user_attr("val_segment_acc", r["val"]["segment_acc"])
        return r["val"]["song_acc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    study.optimize(objective, n_trials=a.trials)
    best = dict(study.best_params)
    print("\nBEST val song-acc", study.best_value, best)
    save_json(best, os.path.join(a.out, "best_params.json"))
    study.trials_dataframe().to_csv(os.path.join(a.out, "trials.csv"), index=False)

    # retrain with the best config for longer, report TEST once
    p, extra = {}, {}
    for k, v in best.items():
        (extra if k in ("batch", "spec_aug") else p)[k] = v
    r = train_eval(a.model, p, data, epochs=a.final_epochs, batch=extra.get("batch", 64),
                   augment=extra.get("spec_aug", False), patience=12, seed=a.seed, verbose=0)
    save_json({"best_params": best, "val": r["val"], "test": r["test"]}, os.path.join(a.out, "final_metrics.json"))
    print(f"\nFINAL (best params) TEST segment acc {r['test']['segment_acc']:.3f} | song acc {r['test']['song_acc']:.3f}")


if __name__ == "__main__":
    main()
