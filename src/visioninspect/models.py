from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TinySegmenter(nn.Module):
    """A compact encoder-decoder suited to edge inspection deployments."""

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


def classical_anomaly_map(image: np.ndarray) -> np.ndarray:
    """High-pass OpenCV baseline with a NumPy fallback."""
    try:
        import cv2

        smooth = cv2.GaussianBlur(image, (11, 11), 0)
    except ImportError:
        padded = np.pad(image, 2, mode="reflect")
        smooth = np.zeros_like(image)
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                smooth[row, column] = padded[row : row + 5, column : column + 5].mean()
    score = np.abs(image - smooth)
    return np.clip(score / 0.42, 0, 1)


def model_size_mb(model: nn.Module) -> float:
    return (
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1e6
    )
