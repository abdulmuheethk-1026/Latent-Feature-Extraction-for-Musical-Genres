# Experiment scripts

These are the command-line entry points for reproducing the paper, training improved models, tuning and running the demo. Run all commands from the repository root after installing [dependencies](../requirements.txt). Replace `/path/to/genres_original` with the GTZAN WAV directory.

```bash
# Reproduce the original random-segment protocol and compare it with a song split.
python scripts/01_reproduce_paper.py --data-root "/path/to/genres_original" --split random_segment --out outputs/paper_leaky
python scripts/01_reproduce_paper.py --data-root "/path/to/genres_original" --split song --out outputs/paper_song

# Train either of the improved models on a song split.
python scripts/02_train_improved.py --data-root "/path/to/genres_original" --model cnn --out outputs/cnn
python scripts/02_train_improved.py --data-root "/path/to/genres_original" --model conv_dsae --out outputs/conv_dsae

# Optional Optuna search, scored on validation song accuracy.
python scripts/03_tune_optuna.py --data-root "/path/to/genres_original" --model conv_dsae --trials 20 --out outputs/tune

# Evaluate unseen songs and classify your own WAV using the default CNN.
python scripts/04_demo.py --weights outputs/cnn/cnn.weights.h5 --cache outputs/cache/mel64x128.npz --n 12 --audio "my_song.wav"
```

Add `--quick` to either training script for a short smoke run. Training writes metrics and plots under the selected `--out` path. The demo needs matching weights and training features. If the cache is unavailable, pass `--data-root` to rebuild it.
