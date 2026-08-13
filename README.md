# Intel Image Classification — TinyVGG CNN from Scratch

A PyTorch implementation of a **TinyVGG** convolutional neural network trained from
scratch to classify natural scene images into 6 categories:
`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

The model is deliberately simple (built only with `nn.Conv2d`, `nn.ReLU`,
`nn.MaxPool2d`, and a single `nn.Linear` layer) to demonstrate the full
end-to-end deep learning workflow — from data loading and a custom training loop
to evaluation, visualization, and model export.

## Dataset

- **Source:** [kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Structure:** `data/seg_train/seg_train/` (14,034 images) and `data/seg_test/seg_test/` (3,000 images)
- **Classes (6):** buildings, forest, glacier, mountain, sea, street
- **License:** The dataset is released under the Intel / MIT license
  ([Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
  — see the dataset page for the exact license text). Please review it before
  redistributing or using the data commercially.

Download it with the Kaggle API:

```bash
kaggle datasets download -d puneet6060/intel-image-classification
# extract into ./data/ so that data/seg_train/seg_train/ and data/seg_test/seg_test/ exist
```

## Model Architecture — TinyVGG

The network follows the classic TinyVGG design: two convolutional blocks, each
made of **Conv → ReLU → Conv → ReLU → MaxPool**, followed by a flatten +
linear classifier.

```
Input: (batch, 3, 64, 64)  RGB image resized to 64x64

Block 1: Conv2d(3 -> hidden) -> ReLU -> Conv2d(hidden -> hidden) -> ReLU -> MaxPool2d(2)
Block 2: Conv2d(hidden -> hidden) -> ReLU -> Conv2d(hidden -> hidden) -> ReLU -> MaxPool2d(2)

Classifier: Flatten -> Linear(hidden * 16 * 16 -> 6)
```

- **Activation:** ReLU
- **Pooling:** 2x2 MaxPool with stride 2 (64x64 → 32x32 → 16x16)
- **Loss:** `nn.CrossEntropyLoss`
- **Optimizer:** Adam (lr=0.001)

## Results

Two hyperparameter configurations were compared (5 epochs each, batch size 32,
image size 64x64). Test accuracy is on the 3,000-image test split.

| Model | Hidden Units | Epochs | Batch Size | LR | Best Test Accuracy |
|-------|-------------|--------|-----------|-----|--------------------|
| TinyVGG-10 | 10 | 5 | 32 | 0.001 | **76.67%** |
| TinyVGG-32 | 32 | 5 | 32 | 0.001 | **81.52%** |

`TinyVGG-32` (hidden_units=32) achieved the best test accuracy of **81.52%** at
epoch 5 and its weights are saved to `results/model.pth`.

### Loss & Accuracy Curves

![Loss and accuracy curves](results/loss_accuracy_curves.png)

### Confusion Matrix (best model)

![Confusion matrix](results/confusion_matrix.png)

More artifacts can be found in the `results/` folder:
`misclassified_examples.png` shows incorrectly classified samples.

## Project Structure

```
intel-image-classification/
├── notebooks/
│   └── training.ipynb        # Full walkthrough: model, training, plots
├── src/
│   ├── data_setup.py         # ImageFolder -> DataLoader creation + transforms
│   ├── model_builder.py      # TinyVGG architecture definition
│   ├── engine.py             # train_step / test_step / train loops
│   ├── utils.py              # Model saving + plotting helpers
│   └── train.py              # CLI entry point
├── results/                  # Plots + best model weights (model.pth)
├── data/                     # Dataset (ignored by git)
├── requirements.txt
└── README.md
```

## Installation

Requires Python 3.10+.

```bash
# 1. Clone / enter the project
cd intel-image-classification

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# Note: for CUDA support, install PyTorch from the official index first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Usage

### From the CLI

```bash
# Default training (hidden_units=32, 5 epochs)
python src/train.py

# Custom hyperparameters
python src/train.py --hidden_units 32 --epochs 10 --batch_size 64 --lr 0.0005

# Headless mode (no plots)
python src/train.py --no_plots
```

Outputs are written to `results/`:
- `model.pth` — best model weights
- `loss_accuracy_curves.png`, `confusion_matrix.png`, `misclassified_examples.png`

### From the notebook

Open and run all cells:

```bash
jupyter notebook notebooks/training.ipynb
```

## Acknowledgments

This project was first prototyped on Kaggle: https://www.kaggle.com/code/emrezorlu1239/intel-image-classification-tinyvgg-cnn
