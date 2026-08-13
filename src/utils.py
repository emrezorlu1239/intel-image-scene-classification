"""
utils.py
========
Helper functions for saving models, plotting confusion matrices, and
visualizing training curves.

Usage:
    from src.utils import save_model, plot_loss_curves, plot_confusion_matrix
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from typing import Dict, List


def save_model(
    model: nn.Module,
    target_dir: str,
    model_name: str = "model.pth"
) -> None:
    """
    Saves a PyTorch model's state_dict to disk.

    Args:
        model (nn.Module): PyTorch model to save.
        target_dir (str): Path of the folder where the model is saved.
        model_name (str): Name of the file to save. Default: "model.pth".

    Example:
        save_model(model=best_model, target_dir="results", model_name="model.pth")
    """
    # Create the target folder (if missing)
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Build the model file path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "Model name must end with '.pth' or '.pt'."
    model_save_path = target_dir_path / model_name

    # Save the model
    torch.save(obj=model.state_dict(), f=model_save_path)
    print(f"[INFO] Model saved: {model_save_path}")


def plot_loss_curves(
    results: Dict[str, List[float]],
    model_name: str = "Model",
    save_path: str = None
) -> None:
    """
    Plots the training and test loss/accuracy curves.

    Args:
        results (Dict[str, List[float]]): Dictionary returned by the engine.train() function.
            Anahtarlar: "train_loss", "train_acc", "test_loss", "test_acc"
        model_name (str): Model name shown in the plot title.
        save_path (str, optional): File path where the plot is saved.
            If None, the plot is only displayed.

    Example:
        plot_loss_curves(results=results_32, model_name="TinyVGG-32", save_path="results/curves.png")
    """
    epochs = range(1, len(results["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, results["train_loss"], label="Train Loss", color="blue", marker="o")
    ax1.plot(epochs, results["test_loss"], label="Test Loss", color="red", linestyle="--", marker="s")
    ax1.set_title(f"{model_name} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(epochs, [acc * 100 for acc in results["train_acc"]], label="Train Acc", color="blue", marker="o")
    ax2.plot(epochs, [acc * 100 for acc in results["test_acc"]], label="Test Acc", color="red", linestyle="--", marker="s")
    ax2.set_title(f"{model_name} - Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Training curves saved: {save_path}")

    plt.show()


def plot_confusion_matrix(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    model_name: str = "Model",
    save_path: str = None
) -> None:
    """
    Plots a confusion matrix from the model predictions.

    Args:
        model (nn.Module): Trained model to evaluate.
        dataloader (DataLoader): Test data loader.
        class_names (List[str]): List of class names.
        device (torch.device): Device used for computation.
        model_name (str): Model name shown in the plot title.
        save_path (str, optional): File path where the plot is saved.

    Example:
        plot_confusion_matrix(
            model=best_model,
            dataloader=test_loader,
            class_names=class_names,
            device=device,
            save_path="results/confusion_matrix.png"
        )
    """
    model.eval()
    y_true, y_preds = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            y_true.extend(y.numpy())
            y_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Confusion Matrix - {model_name}")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved: {save_path}")

    plt.show()

    return y_true, y_preds


def plot_misclassified(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    y_true: List[int],
    y_preds: List[int],
    class_names: List[str],
    n_samples: int = 6,
    save_path: str = None
) -> None:
    """
    Visualizes misclassified image examples.

    Args:
        model (nn.Module): Model (not required for plotting; kept for future extension).
        dataset: Test dataset (ImageFolder format).
        y_true (List[int]): List of true labels.
        y_preds (List[int]): List of predicted labels.
        class_names (List[str]): Class names.
        n_samples (int): Number of examples to show. Default: 6.
        save_path (str, optional): File path where the plot is saved.
    """
    misclassified_idx = [i for i, (t, p) in enumerate(zip(y_true, y_preds)) if t != p]

    nrows = (n_samples + 2) // 3
    plt.figure(figsize=(12, 4 * nrows))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(misclassified_idx[:n_samples]):
        img, true_label = dataset[idx]
        pred_label = y_preds[idx]

        # Convert the normalized tensor into a displayable image
        img_np = img.numpy().transpose((1, 2, 0))
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        plt.subplot(nrows, 3, i + 1)
        plt.imshow(img_np)
        plt.title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            color="red"
        )
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Misclassified examples saved: {save_path}")

    plt.show()
