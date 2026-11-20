# Genre package

This package implements both experiment pipelines and the shared evaluation logic.

| Modules | Purpose |
| --- | --- |
| `config.py`, `utils.py` | Defaults, dataset discovery, splitting and leakage checks |
| `data_paper.py`, `data_mel.py` | Waveform segments and log-mel windows, with feature caching |
| `models_paper.py`, `models_improved.py` | Paper architectures and improved convolutional models |
| `training.py`, `experiment.py` | Shared fitting and experiment orchestration |
| `evaluate.py` | Segment and song metrics and evaluation plots |

Run these modules through the [experiment scripts](../../scripts/README.md). To exercise the data and evaluation code with synthetic inputs, run `python tests/test_pipeline.py` from the repository root.
