from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SyntheticInspectionDataset
from .metrics import EvaluationReport, calibrate_threshold, evaluate_predictions
from .models import TinySegmenter


@dataclass(frozen=True, slots=True)
class TrainingResult:
    report: EvaluationReport
    losses: list[float]
    model: TinySegmenter


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum()
    return 1 - (2 * intersection + 1) / (probabilities.sum() + target.sum() + 1)


def train_model(seed: int = 11, epochs: int = 20, samples: int = 96) -> TrainingResult:
    set_reproducible(seed)
    train = SyntheticInspectionDataset(samples, seed=seed, augment=True)
    validation = SyntheticInspectionDataset(32, seed=seed + 5_000)
    test = SyntheticInspectionDataset(40, seed=seed + 10_000)
    loader = DataLoader(
        train, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    model = TinySegmenter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    binary_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for images, masks, _ in loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = binary_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        losses.append(round(epoch_loss / len(loader), 6))

    model.eval()
    validation_images = torch.stack([validation[index][0] for index in range(len(validation))])
    validation_masks = np.stack(
        [validation[index][1].numpy()[0] for index in range(len(validation))]
    )
    images = torch.stack([test[index][0] for index in range(len(test))])
    masks = np.stack([test[index][1].numpy()[0] for index in range(len(test))])
    labels = np.array([int(test[index][2]) for index in range(len(test))])
    with torch.inference_mode():
        validation_scores = torch.sigmoid(model(validation_images)).numpy()[:, 0]
        score_maps = torch.sigmoid(model(images)).numpy()[:, 0]
    threshold = calibrate_threshold(validation_masks, validation_scores)
    report = evaluate_predictions(masks, score_maps, labels, threshold=threshold)
    return TrainingResult(report, losses, model)


def save_training_result(result: TrainingResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), output / "visioninspect.pt")
    (output / "training-report.json").write_text(
        json.dumps({"metrics": result.report.to_dict(), "losses": result.losses}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    result = train_model()
    save_training_result(result, Path("artifacts"))
    print(json.dumps(result.report.to_dict(), indent=2))
