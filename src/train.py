"""
train.py
========
Main training script runnable from the command line.
Combines the src/data_setup, model_builder, engine, and utils modules.

Usage examples:
    # Run with default settings (hidden_units=32, epochs=5)
    python train.py

    # Run with custom hyperparameters
    python train.py --hidden_units 10 --epochs 10 --batch_size 64 --lr 0.0005

    # Help menu
    python train.py --help
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Add the src/ folder itself to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_setup import create_dataloaders
from model_builder import TinyVGG
from engine import train
from utils import save_model, plot_loss_curves, plot_confusion_matrix, plot_misclassified


def get_device() -> torch.device:
    """Returns the available device (CUDA if GPU compatible, CPU otherwise)."""
    if torch.cuda.is_available():
        try:
            # CUDA kernel smoke test
            test_conv = nn.Conv2d(1, 1, 1).cuda()
            _ = torch.zeros(1, 1, 4, 4).cuda()
            _ = test_conv(_)
            del test_conv
            return torch.device("cuda")
        except Exception:
            print("[WARNING] GPU present but CUDA kernel incompatibility detected. Using CPU.")
    return torch.device("cpu")


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Intel Image Classification - TinyVGG Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data path arguments
    parser.add_argument(
        "--train_dir",
        type=str,
        default=os.path.join("data", "seg_train", "seg_train"),
        help="Path to the training data folder"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.path.join("data", "seg_test", "seg_test"),
        help="Path to the test data folder"
    )

    # Hyperparameter arguments
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=32,
        help="Number of filters in the convolutional layers"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Image resizing size (img_size x img_size)"
    )

    # Output arguments
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Folder where results (model, plots) will be saved"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model.pth",
        help="Name of the model file to save"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="Do not generate plots (for headless/CI environments)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- 0. Device Selection ---
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Intel Image Classification - TinyVGG Training")
    print(f"{'='*60}")
    print(f"[INFO] Device      : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU         : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Hidden Units: {args.hidden_units}")
    print(f"[INFO] Epochs      : {args.epochs}")
    print(f"[INFO] Batch Size  : {args.batch_size}")
    print(f"[INFO] LR          : {args.lr}")
    print(f"[INFO] Img Size    : {args.img_size}x{args.img_size}")
    print(f"{'='*60}\n")

    # --- 1. Data Loading ---
    print("[STEP 1] Loading datasets...")
    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    print(f"[INFO] Classes    : {class_names}")
    print(f"[INFO] Train batches: {len(train_loader)} batch × {args.batch_size} = ~{len(train_loader.dataset)} images")
    print(f"[INFO] Test batches: {len(test_loader)} batch × {args.batch_size} = ~{len(test_loader.dataset)} images\n")

    # --- 2. Model Creation ---
    print("[STEP 2] Creating TinyVGG model...")
    torch.manual_seed(42)
    model = TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(device)
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- 3. Loss Function & Optimizer ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 4. Training ---
    print(f"[STEP 3] Training model for {args.epochs} epochs...")
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device
    )

    # --- 5. Report Best Test Accuracy ---
    best_test_acc = max(results["test_acc"])
    best_epoch = results["test_acc"].index(best_test_acc) + 1
    print(f"\n[RESULT] Best test accuracy: %{best_test_acc*100:.2f} (Epoch {best_epoch})")

    # --- 6. Model Saving ---
    print(f"\n[STEP 4] Saving model: {args.results_dir}/{args.model_name}")
    os.makedirs(args.results_dir, exist_ok=True)
    save_model(model=model, target_dir=args.results_dir, model_name=args.model_name)

    # --- 7. Plots ---
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")  # Headless-safe backend

        model_label = f"TinyVGG (hidden_units={args.hidden_units})"

        print(f"\n[STEP 5] Saving training curves...")
        plot_loss_curves(
            results=results,
            model_name=model_label,
            save_path=os.path.join(args.results_dir, "loss_accuracy_curves.png")
        )

        print(f"[STEP 6] Saving confusion matrix...")
        y_true, y_preds = plot_confusion_matrix(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=device,
            model_name=model_label,
            save_path=os.path.join(args.results_dir, "confusion_matrix.png")
        )

        print(f"[STEP 7] Saving misclassified examples...")
        plot_misclassified(
            model=model,
            dataset=test_loader.dataset,
            y_true=y_true,
            y_preds=y_preds,
            class_names=class_names,
            n_samples=6,
            save_path=os.path.join(args.results_dir, "misclassified_examples.png")
        )

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Model    : {args.results_dir}/{args.model_name}")
    print(f"  Plots    : {args.results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
