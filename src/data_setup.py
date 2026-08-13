"""
data_setup.py
=============
Functions for preparing DataLoaders for the Intel Image Classification dataset.

Usage:
    from src.data_setup import create_dataloaders
    train_loader, test_loader, class_names = create_dataloaders(
        train_dir="data/seg_train/seg_train",
        test_dir="data/seg_test/seg_test",
        batch_size=32
    )
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    img_size: int = 64,
    num_workers: int = 0
):
    """
    Creates training and test DataLoaders for the Intel Image Classification dataset.

    Normalization: ImageNet statistics are used.
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    Args:
        train_dir (str): Path to the training data folder (ImageFolder format).
        test_dir (str): Path to the test data folder (ImageFolder format).
        batch_size (int): Number of samples per DataLoader iteration. Default: 32.
        img_size (int): Size to which images are resized. Default: 64.
        num_workers (int): Number of subprocesses used by the DataLoader. Default: 0.

    Returns:
        tuple: (train_loader, test_loader, class_names)
            - train_loader (DataLoader): Training data loader.
            - test_loader (DataLoader): Test data loader.
            - class_names (list[str]): List of class names.
    """
    # Common transform pipeline (Resize + ToTensor + Normalize)
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create datasets with ImageFolder
    train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

    class_names = train_data.classes

    # DataLoaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader, class_names
