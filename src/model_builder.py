"""
model_builder.py
================
PyTorch module class defining the TinyVGG CNN architecture.

Architecture (following the structure from lessons 10-13):
  Block 1: Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d
  Block 2: Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d
  Classifier: Flatten -> Linear

Usage:
    from src.model_builder import TinyVGG
    model = TinyVGG(input_shape=3, hidden_units=32, output_shape=6)
"""

import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    """
    TinyVGG CNN architecture for Intel Image Classification.

    Structure:
      - Conv Block 1: Conv2d(3, hidden_units, 3) -> ReLU -> Conv2d -> ReLU -> MaxPool2d(2)
      - Conv Block 2: Conv2d(hidden_units, hidden_units, 3) -> ReLU -> Conv2d -> ReLU -> MaxPool2d(2)
      - Classifier: Flatten -> Linear(hidden_units * 16 * 16, output_shape)

    NOTE: This architecture is designed for 64x64 input images.
         You must update the Linear layer input size for different dimensions.

    Args:
        input_shape (int): Number of input channels (3 for RGB).
        hidden_units (int): Number of filters in the convolutional layers.
        output_shape (int): Number of classes (6 for the Intel dataset).
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # First convolution block
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second convolution block
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier layer
        # 64x64 input -> MaxPool x2 -> 16x16 output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor, shape: (batch, channels, height, width).

        Returns:
            torch.Tensor: Raw logit outputs, shape: (batch, output_shape).
        """
        return self.classifier(self.block_2(self.block_1(x)))
