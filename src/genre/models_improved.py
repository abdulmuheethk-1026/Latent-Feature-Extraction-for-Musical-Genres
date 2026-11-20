"""Better front-end + back-end:  log-mel (64 x 128 x 1) -> small VGG-style CNN.

build_cnn        : pure classifier, 64-d 'embedding' layer before the softmax.
build_conv_dsae  : the paper's idea (autoencoder + classifier on the SAME 64-d code) but convolutional.
                   Trained as a *denoising* AE when SpecAugment is on (input masked, target clean).
"""
from tensorflow.keras import layers, Model, optimizers, losses


def _conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def build_cnn(input_shape=(64, 128, 1), n_classes=4, base=32, latent=64, dropout=0.3, lr=1e-3, **_):
    inp = layers.Input(input_shape, name="x")
    x = inp
    for f in (base, 2 * base, 4 * base, 4 * base):
        x = _conv_block(x, f)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout / 2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    z = layers.Dense(latent, activation="relu", name="embedding")(x)
    z = layers.Dropout(dropout)(z)
    logits = layers.Dense(n_classes, name="logits")(z)
    m = Model(inp, logits, name="mel_cnn")
    m.compile(optimizers.Adam(lr), loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["sparse_categorical_accuracy"])
    return m


def build_conv_dsae(input_shape=(64, 128, 1), n_classes=4, base=32, latent=64, dropout=0.3, lr=1e-3,
                    recon_weight=1.0, cls_weight=1.0, **_):
    inp = layers.Input(input_shape, name="x")
    x = inp
    for f in (base, 2 * base, 4 * base, 4 * base):          # 64x128 -> 4x8
        x = _conv_block(x, f)
        x = layers.MaxPooling2D(2)(x)
    h, w, c = input_shape[0] // 16, input_shape[1] // 16, 4 * base
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    z = layers.Dense(latent, activation="relu", name="embedding")(x)
    # classifier head: same shape as the paper (64 -> 32 -> 16 -> classes, tanh)
    k = layers.Dense(32, activation="tanh")(z)
    k = layers.Dense(16, activation="tanh")(k)
    logits = layers.Dense(n_classes, name="logits")(k)
    # decoder
    d = layers.Dense(h * w * c, activation="relu")(z)
    d = layers.Reshape((h, w, c))(d)
    for f in (4 * base, 2 * base, base, max(base // 2, 8)):  # 4x8 -> 64x128
        d = layers.UpSampling2D(2)(d)
        d = _conv_block(d, f)
    recon = layers.Conv2D(1, 3, padding="same", name="recon")(d)
    m = Model(inp, {"recon": recon, "logits": logits}, name="conv_deep_softmax_autoencoder")
    m.compile(optimizers.Adam(lr),
              loss={"recon": "mse", "logits": losses.SparseCategoricalCrossentropy(from_logits=True)},
              loss_weights={"recon": recon_weight, "logits": cls_weight},
              metrics={"logits": ["sparse_categorical_accuracy"]})
    return m
