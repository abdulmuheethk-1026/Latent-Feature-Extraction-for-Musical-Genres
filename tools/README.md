# Notebook generators

These tools rebuild the self-contained Kaggle notebooks from the Python source and scripts. Use them after changing code embedded in the notebooks so the Kaggle and local workflows stay in sync.

Run from the repository root with Python:

```bash
python tools/build_kaggle_notebook.py
python tools/build_demo_notebook.py
```

The commands overwrite the corresponding files in [`notebooks/`](../notebooks/README.md).
