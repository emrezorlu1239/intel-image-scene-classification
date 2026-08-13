\"\"\"
train.py
========
Main training script runnable from the command line.
Combines src/data_setup, model_builder, engine, and utils modules.
\"\"\"

import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_setup import create_dataloaders
from model_builder import TinyVGG
from engine import train
from utils import save_model, plot_loss_curves, plot_confusion_matrix, plot_misclassified


def get_device(force_gpu: bool = False) -> torch.device:
    \"\"\"Returns the available device (CUDA if available, CPU otherwise).\"\"\"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    elif force_gpu:
        raise RuntimeError("CUDA is required (--force_gpu) but no CUDA-capable device was detected.")
    else:
        print("[INFO] CUDA not available. Using CPU.")
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intel Image Classification - TinyVGG Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train_dir", type=str, default=os.path.join("data", "seg_train", "seg_train"))
    parser.add_argument("--test_dir", type=str, default=os.path.join("data", "seg_test", "seg_test"))
    parser.add_argument("--hidden_units", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="model.pth")
    parser.add_argument("--no_plots", action="store_true", default=False)
    parser.add_argument("--force_gpu", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device(force_gpu=args.force_gpu)
    print(f"\n{'='*60}")
    print("Intel Image Classification - TinyVGG Training")
    print(f"{'='*60}")
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Hidden Units: {args.hidden_units}")
    print(f"[INFO] Epochs      : {args.epochs}")
    print(f"[INFO] Batch Size  : {args.batch_size}")
    print(f"[INFO] LR          : {args.lr}")
    print(f"[INFO] Img Size    : {args.img_size}x{args.img_size}")
    print(f"{'='*60}\n")

    print("[STEP 1] Loading datasets...")
    train_loader, test_loader, class_names = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    print(f"[INFO] Classes     : {class_names}")
    print(f"[INFO] Train count : {len(train_loader.dataset)} images ({len(train_loader)} batches)")
    print(f"[INFO] Test count  : {len(test_loader.dataset)} images ({len(test_loader)} batches)\n")

    print("[STEP 2] Creating TinyVGG model...")
    torch.manual_seed(42)
    model = TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(device)
    print(f"[INFO] Total model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    best_test_acc = max(results["test_acc"])
    best_epoch = results["test_acc"].index(best_test_acc) + 1
    print(f"\n[RESULT] Best test accuracy: {best_test_acc*100:.2f}% (Epoch {best_epoch})")

    print(f"\n[STEP 4] Saving model: {args.results_dir}/{args.model_name}")
    os.makedirs(args.results_dir, exist_ok=True)
    save_model(model=model, target_dir=args.results_dir, model_name=args.model_name)

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")

        model_label = f"TinyVGG (hidden_units={args.hidden_units})"

        print("\n[STEP 5] Saving training curves...")
        plot_loss_curves(
            results=results,
            model_name=model_label,
            save_path=os.path.join(args.results_dir, "loss_accuracy_curves.png")
        )

        print("[STEP 6] Saving confusion matrix...")
        y_true, y_preds = plot_confusion_matrix(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=device,
            model_name=model_label,
            save_path=os.path.join(args.results_dir, "confusion_matrix.png")
        )

        print("[STEP 7] Saving misclassified examples...")
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
    print("Training complete!")
    print(f"  Model    : {args.results_dir}/{args.model_name}")
    print(f"  Plots    : {args.results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
\"\"\