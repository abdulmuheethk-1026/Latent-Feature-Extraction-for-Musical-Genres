# Pipeline smoke test

[`test_pipeline.py`](test_pipeline.py) generates synthetic audio and stubs librosa to check feature shapes, song-split isolation, leakage and evaluation outputs without training a model or downloading GTZAN.

From the repository root, with the project dependencies available, run:

```bash
python tests/test_pipeline.py
```
