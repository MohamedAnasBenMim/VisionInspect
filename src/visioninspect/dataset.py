from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class Sample:
    image: np.ndarray
    mask: np.ndarray
    label: int
    defect_type: str


def _disk(mask: np.ndarray, center: tuple[int, int], radius: int) -> None:
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2] = 1.0


def _line(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], width: int) -> None:
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    xs = np.linspace(start[0], end[0], steps).astype(int)
    ys = np.linspace(start[1], end[1], steps).astype(int)
    for x, y in zip(xs, ys, strict=True):
        y0, y1 = max(0, y - width), min(mask.shape[0], y + width + 1)
        x0, x1 = max(0, x - width), min(mask.shape[1], x + width + 1)
        mask[y0:y1, x0:x1] = 1.0


def generate_sample(seed: int, size: int = 64, defective: bool | None = None) -> Sample:
    """Generate a deterministic textured surface and pixel-accurate defect mask."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    texture = 0.52 + 0.035 * np.sin(x / 3.6) + 0.025 * np.cos(y / 6.1)
    texture += rng.normal(0, 0.018, (size, size))
    image = np.clip(texture, 0, 1).astype(np.float32)
    mask = np.zeros_like(image)
    is_defective = bool(seed % 2) if defective is None else defective
    defect_type = "normal"

    if is_defective:
        if seed % 4 in (1, 2):
            defect_type = "scratch"
            x0 = int(rng.integers(7, 18))
            y0 = int(rng.integers(8, size - 18))
            _line(mask, (x0, y0), (size - 8, min(size - 5, y0 + int(rng.integers(8, 22)))), 1)
            image = np.clip(image - mask * float(rng.uniform(0.30, 0.48)), 0, 1)
        else:
            defect_type = "dent"
            center = (int(rng.integers(15, size - 15)), int(rng.integers(15, size - 15)))
            _disk(mask, center, int(rng.integers(5, 9)))
            distance = np.hypot(x - center[0], y - center[1])
            image = np.clip(image - mask * (0.24 + 0.12 * np.cos(distance / 3)), 0, 1)

    return Sample(image.astype(np.float32), mask, int(is_defective), defect_type)


class SyntheticInspectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, count: int, seed: int = 7, augment: bool = False) -> None:
        self.samples = [generate_sample(seed + index) for index in range(count)]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image, mask = sample.image.copy(), sample.mask.copy()
        if self.augment and index % 3 == 0:
            image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
        if self.augment and index % 5 == 0:
            image = np.clip(image * 1.06 - 0.02, 0, 1)
        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(mask[None]).float(),
            torch.tensor(sample.label, dtype=torch.long),
        )


def split_seeds(total: int, seed: int = 17) -> tuple[list[int], list[int], list[int]]:
    """Return leakage-safe 60/20/20 train, validation, and test seed splits."""
    values = np.arange(seed, seed + total)
    np.random.default_rng(seed).shuffle(values)
    train_end, validation_end = int(total * 0.6), int(total * 0.8)
    return (
        values[:train_end].tolist(),
        values[train_end:validation_end].tolist(),
        values[validation_end:].tolist(),
    )
