"""One function used by both the training script and the Optuna tuner."""
import numpy as np
import tensorflow as tf

from . import models_improved as MI, models_paper as MP
from .evaluate import summarize
from .training import fit_model, predict_logits
from .utils import set_seed

AE_KINDS = {"conv_dsae", "paper_dsae"}


def build_model(kind, p, input_shape):
    if kind == "cnn":
        return MI.build_cnn(input_shape, **p)
    if kind == "conv_dsae":
        return MI.build_conv_dsae(input_shape, **p)
    if kind == "paper_dsae":
        return MP.build_dsae(input_dim=input_shape[0], **p)
    if kind == "paper_nn":
        return MP.build_baseline_nn(input_dim=input_shape[0], **p)
    raise ValueError(kind)


def train_eval(kind, params, data, epochs, batch=64, augment=False, patience=10, seed=42, verbose=0, eval_test=True):
    """params -> model kwargs (lr, base, latent, dropout, cls_weight ...). Returns dict with metrics + models."""
    tf.keras.backend.clear_session()
    set_seed(seed)
    (Xtr, ytr, _), (Xva, yva, sva), (Xte, yte, ste) = data["train"], data["val"], data["test"]
    ae = kind in AE_KINDS
    model = build_model(kind, params, Xtr.shape[1:])
    hist, best_val_seg = fit_model(model, Xtr, ytr, Xva, yva, ae, epochs, batch,
                                   augment=augment and Xtr.ndim == 4, patience=patience, verbose=verbose)
    val_res, _ = summarize(predict_logits(model, Xva), yva, sva, name=f"{kind} / val")
    if eval_test:                       # the tuner passes False: test data is never touched while searching
        test_res, test_pred = summarize(predict_logits(model, Xte), yte, ste, name=f"{kind} / TEST")
    else:
        test_res, test_pred = None, None
    return {"model": model, "history": hist, "val": val_res, "test": test_res, "test_pred": test_pred,
            "best_val_segment_acc": best_val_seg}


def embeddings(model, X):
    try:
        emb = tf.keras.Model(model.input, model.get_layer("embedding").output)
        return emb.predict(X, batch_size=256, verbose=0)
    except ValueError:
        return predict_logits(model, X)
