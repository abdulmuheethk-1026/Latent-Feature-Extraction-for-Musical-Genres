# Kaggle notebooks

These notebooks provide self-contained Kaggle workflows that write out the Python files they use.

- [Training notebook](kaggle_genre_extraction.ipynb) runs the reproduction, improved models and optional tuning. Import it into Kaggle, attach GTZAN with the WAV files, enable a GPU and run the setup cells first. Save the outputs before ending the session.
- [Demo notebook](kaggle_demo_only.ipynb) evaluates a trained CNN and classifies a custom clip. Attach the saved training output or a dataset containing `cnn.weights.h5` and `mel64x128.npz`. Include `metrics.json` for comparison with the saved scores. Run the cells in order. CPU is sufficient.

Use the [generator commands](../tools/README.md) after changing the source files embedded in the notebooks.
