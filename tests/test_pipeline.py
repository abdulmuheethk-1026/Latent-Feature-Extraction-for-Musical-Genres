"""Offline smoke tests (no TensorFlow / librosa / dataset needed): librosa is stubbed with synthetic audio.
Run:  python -m pytest -q tests     or     python tests/test_pipeline.py
"""
import os, sys, types, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from scipy.io import wavfile


def _install_fake_librosa():
    lb = types.ModuleType("librosa")
    def load(path, sr=22050, mono=True, duration=None):
        s, y = wavfile.read(path)
        y = y.astype(np.float32) / 32768.0
        return (y[: int(sr * duration)] if duration else y), sr
    feat = types.SimpleNamespace(
        melspectrogram=lambda y, sr, n_fft, hop_length, n_mels, fmin: np.abs(np.random.rand(n_mels, 1 + len(y) // hop_length)).astype(np.float32))
    lb.load, lb.feature = load, feat
    lb.power_to_db = lambda m, ref=1.0, top_db=80.0: 10 * np.log10(m + 1e-10)
    sys.modules["librosa"] = lb


def _make_fake_gtzan(root, n_per_genre=10):
    for g in ["classical", "jazz", "metal", "pop"]:
        os.makedirs(os.path.join(root, g))
        for i in range(n_per_genre):
            y = (np.random.randn(661_500) * 3000).astype(np.int16)
            wavfile.write(os.path.join(root, g, f"{g}.{i:05d}.wav"), 22050, y)


def test_all():
    _install_fake_librosa()
    from genre.data_paper import prepare_paper
    from genre.data_mel import prepare_mel
    from genre.utils import find_gtzan_root, leakage
    from genre.evaluate import summarize, plot_pca, plot_confusion

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "genres_original")
        _make_fake_gtzan(root)
        assert find_gtzan_root("auto" if False else root) == root

        # paper pipeline: 40 songs x 20 segments x 500 dims
        sp = prepare_paper(root, cache=os.path.join(tmp, "p.npz"), split="random_segment")
        n = sum(len(v[0]) for v in sp.values())
        assert n == 40 * 20 and sp["train"][0].shape[1] == 500
        assert leakage(sp) > 0.9, "random segment split must leak songs"
        sp2 = prepare_paper(root, cache=os.path.join(tmp, "p.npz"), split="song")
        assert leakage(sp2) == 0.0
        tr, va, te = (set(np.unique(sp2[k][2])) for k in ("train", "val", "test"))
        assert not (tr & va) and not (tr & te) and not (va & te)

        # mel pipeline: 19 segments per 30 s song, (64,128,1)
        sm = prepare_mel(root, cache=os.path.join(tmp, "m.npz"), split="song")
        assert sm["train"][0].shape[1:] == (64, 128, 1)
        assert sum(len(v[0]) for v in sm.values()) == 40 * 19
        assert leakage(sm) == 0.0
        # train-only standardisation
        assert abs(sm["train"][0].mean()) < 1e-3

        # metrics / plots
        Xte, yte, ste = sm["test"]
        logits = np.random.randn(len(yte), 4)
        res, pred = summarize(logits, yte, ste, name="stub")
        assert 0 <= res["song_acc"] <= 1
        plot_confusion(yte, pred, os.path.join(tmp, "c.png"))
        plot_pca(np.random.randn(len(yte), 8), yte, "t", os.path.join(tmp, "p.png"))
        assert os.path.exists(os.path.join(tmp, "p.png"))
    print("all offline tests passed")


if __name__ == "__main__":
    test_all()
