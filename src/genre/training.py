import numpy as np
import tensorflow as tf
from tensorflow import keras


def predict_logits(model, X, bs=256):
    out = model.predict(X, batch_size=bs, verbose=0)
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (list, tuple)):
        return out[model.output_names.index("logits")]
    return out


def spec_augment(x, max_f=10, max_t=16):
    """One frequency mask + one time mask (zero == train mean because inputs are standardised)."""
    n_mels, n_t = x.shape[0], x.shape[1]
    f = tf.random.uniform([], 0, max_f + 1, tf.int32)
    f0 = tf.random.uniform([], 0, n_mels - max_f, tf.int32)
    t = tf.random.uniform([], 0, max_t + 1, tf.int32)
    t0 = tf.random.uniform([], 0, n_t - max_t, tf.int32)
    fi, ti = tf.range(n_mels), tf.range(n_t)
    fm = tf.cast(tf.logical_not((fi >= f0) & (fi < f0 + f)), x.dtype)
    tm = tf.cast(tf.logical_not((ti >= t0) & (ti < t0 + t)), x.dtype)
    return x * fm[:, None, None] * tm[None, :, None]


def make_dataset(X, y, batch, augment=False, ae=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(min(len(X), 20000), reshuffle_each_iteration=True)

    def f(x, yy):
        xi = spec_augment(x) if augment else x
        return (xi, {"recon": x, "logits": yy}) if ae else (xi, yy)

    return ds.map(f, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


class BestValAcc(keras.callbacks.Callback):
    """Tracks validation accuracy itself (metric names differ between Keras 2/3), keeps best weights, early-stops."""

    def __init__(self, Xva, yva, patience=10):
        super().__init__()
        self.Xva, self.yva, self.patience = Xva, yva, patience
        self.accs, self.best, self.best_w, self.wait = [], -1.0, None, 0

    def on_epoch_end(self, epoch, logs=None):
        acc = float((predict_logits(self.model, self.Xva).argmax(1) == self.yva).mean())
        self.accs.append(acc)
        if acc > self.best:
            self.best, self.best_w, self.wait = acc, self.model.get_weights(), 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_w is not None:
            self.model.set_weights(self.best_w)


def fit_model(model, Xtr, ytr, Xva, yva, ae, epochs, batch, augment=False, patience=10, lr_patience=4, verbose=0):
    ds = make_dataset(Xtr, ytr, batch, augment=augment, ae=ae)
    val = (Xva, {"recon": Xva, "logits": yva}) if ae else (Xva, yva)
    cb = BestValAcc(Xva, yva, patience)
    cbs = [cb, keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=lr_patience, min_lr=1e-6)]
    h = model.fit(ds, validation_data=val, epochs=epochs, callbacks=cbs, verbose=verbose)
    hist = {k: v for k, v in h.history.items() if "loss" in k or "accuracy" in k}
    hist["val_seg_acc"] = cb.accs
    return hist, cb.best
