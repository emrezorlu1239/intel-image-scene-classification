"""
engine.py
=========
train_step, test_step, and train functions for TinyVGG model training.

Usage:
    from src.engine import train
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=5,
        device=device
    )
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Trains the model for a single epoch.

    Args:
        model (nn.Module): PyTorch model to train.
        dataloader (DataLoader): Training data loader.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        device (torch.device): Device on which computation is performed (CPU/GPU).

    Returns:
        Tuple[float, float]: (average training loss, average training accuracy)
    """
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Zero the gradients
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Update the weights
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Average over batches
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates the model on the test data.

    Args:
        model (nn.Module): PyTorch model to evaluate.
        dataloader (DataLoader): Test data loader.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device on which computation is performed (CPU/GPU).

    Returns:
        Tuple[float, float]: (average test loss, average test accuracy)
    """
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Average over batches
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Trains and tests the model for the given number of epochs.

    Prints and records training and test metrics at the end of each epoch.

    Args:
        model (nn.Module): PyTorch model to train.
        train_dataloader (DataLoader): Training data loader.
        test_dataloader (DataLoader): Test data loader.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        loss_fn (nn.Module): Loss function.
        epochs (int): Number of training epochs.
        device (torch.device): Device on which computation is performed (CPU/GPU).

    Returns:
        Dict[str, List[float]]: Dictionary of the training history:
            {
                "train_loss": [...],
                "train_acc": [...],
                "test_loss": [...],
                "test_acc": [...]
            }
    """
    # Dictionary to track results
    results: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Print the result of each epoch
        print(
            f"Epoch: {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: %{train_acc*100:.2f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: %{test_acc*100:.2f}"
        )

        # Record the results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
