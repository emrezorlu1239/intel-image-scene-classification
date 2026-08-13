import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

print("Executing notebook notebooks/training.ipynb...")
with open("notebooks/training.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

os.chdir("notebooks")
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("Notebook executed successfully.")
finally:
    os.chdir("..")
    with open("notebooks/training.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook outputs saved to notebooks/training.ipynb.")
