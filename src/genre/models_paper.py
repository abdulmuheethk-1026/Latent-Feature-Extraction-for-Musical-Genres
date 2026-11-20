"""Exact architectures from the original repo, ported from TF1 (tf.contrib / placeholders) to tf.keras.

softmaxautoencoder.py  -> build_dsae        (Deep Softmax Autoencoder, the paper's final model)
encoder.py             -> build_vanilla_ae  (reconstruction-only autoencoder, 400 epochs in the repo)
tfsoftmax.py           -> build_baseline_nn (500 -> 128 tanh -> 4)

Faithfulness notes
* Xavier-uniform init == Keras glorot_uniform (default); biases start at zero (default).
* Original dropout used keep_prob=0.8  ->  rate 0.2, applied after the 2nd encoder layer and 2nd decoder layer only.
* Original decoder ends with ReLU (so it can never output negative samples although the waveform is in [-1, 1]).
  Kept as default for reproduction; `recon_activation="linear"` is offered for the tuned variant.
* Original loss (code, not the paper's Eq. 3):  mean((x - x_hat)^2) + 0.1 * mean(softmax_cross_entropy)
"""
from tensorflow.keras import layers, Model, optimizers, losses


def _encode(x, latent, dropout):
    x = layers.Dense(256, activation="sigmoid", name="enc1")(x)
    x = layers.Dense(192, activation="tanh", name="enc2")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", name="enc3")(x)
    return layers.Dense(latent, activation="relu", name="embedding")(x)


def _decode(z, input_dim, dropout, recon_activation):
    x = layers.Dense(128, activation="sigmoid", name="dec1")(z)
    x = layers.Dense(192, activation="sigmoid", name="dec2")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu", name="dec3")(x)
    return layers.Dense(input_dim, activation=recon_activation, name="recon")(x)


def build_baseline_nn(input_dim=500, n_classes=4, lr=1e-4, hidden=128):
    inp = layers.Input((input_dim,), name="x")
    h = layers.Dense(hidden, activation="tanh", name="hidden")(inp)
    logits = layers.Dense(n_classes, name="logits")(h)
    m = Model(inp, logits, name="baseline_nn")
    m.compile(optimizers.Adam(lr), loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["sparse_categorical_accuracy"])
    return m


def build_vanilla_ae(input_dim=500, latent=64, dropout=0.2, lr=1e-4, recon_activation="relu"):
    inp = layers.Input((input_dim,), name="x")
    z = _encode(inp, latent, dropout)
    recon = _decode(z, input_dim, dropout, recon_activation)
    m = Model(inp, recon, name="vanilla_ae")
    m.compile(optimizers.Adam(lr), loss="mse")
    return m


def build_dsae(input_dim=500, latent=64, n_classes=4, dropout=0.2, lr=1e-4,
               recon_weight=1.0, cls_weight=0.1, recon_activation="relu"):
    inp = layers.Input((input_dim,), name="x")
    z = _encode(inp, latent, dropout)
    recon = _decode(z, input_dim, dropout, recon_activation)
    c = layers.Dense(32, activation="tanh", name="cls1")(z)
    c = layers.Dense(16, activation="tanh", name="cls2")(c)
    logits = layers.Dense(n_classes, name="logits")(c)
    m = Model(inp, {"recon": recon, "logits": logits}, name="deep_softmax_autoencoder")
    m.compile(optimizers.Adam(lr),
              loss={"recon": "mse", "logits": losses.SparseCategoricalCrossentropy(from_logits=True)},
              loss_weights={"recon": recon_weight, "logits": cls_weight},
              metrics={"logits": ["sparse_categorical_accuracy"]})
    return m
