import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

print("Executing notebooks/training.ipynb on GPU...")
notebook_path = "notebooks/training.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "."}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("[SUCCESS] Notebook executed completely with all cell outputs saved in-place!")