from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    image_auroc: float
    pixel_auroc: float
    precision: float
    recall: float
    dice: float
    threshold: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = targets.astype(bool).ravel()
    scores = scores.astype(float).ravel()
    positive, negative = scores[targets], scores[~targets]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUROC requires positive and negative examples")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def evaluate_predictions(
    masks: np.ndarray,
    score_maps: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationReport:
    if masks.shape != score_maps.shape:
        raise ValueError("masks and score maps must have identical shapes")
    flattened = score_maps.reshape(len(score_maps), -1)
    top_k = max(1, flattened.shape[1] // 20)
    image_scores = np.partition(flattened, -top_k, axis=1)[:, -top_k:].mean(axis=1)
    predicted = score_maps >= threshold
    truth = masks.astype(bool)
    true_positive = np.count_nonzero(predicted & truth)
    false_positive = np.count_nonzero(predicted & ~truth)
    false_negative = np.count_nonzero(~predicted & truth)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    return EvaluationReport(
        image_auroc=round(binary_auroc(labels, image_scores), 6),
        pixel_auroc=round(binary_auroc(truth, score_maps), 6),
        precision=round(float(precision), 6),
        recall=round(float(recall), 6),
        dice=round(float(dice), 6),
        threshold=threshold,
    )


def calibrate_threshold(masks: np.ndarray, score_maps: np.ndarray) -> float:
    """Select a pixel threshold on validation data by maximum Dice score."""
    truth = masks.astype(bool)
    best_threshold, best_dice = 0.5, -1.0
    lower = max(0.01, float(np.quantile(score_maps, 0.5)))
    upper = min(0.99, float(np.quantile(score_maps, 0.999)))
    for threshold in np.linspace(lower, upper, 80):
        predicted = score_maps >= threshold
        true_positive = np.count_nonzero(predicted & truth)
        false_positive = np.count_nonzero(predicted & ~truth)
        false_negative = np.count_nonzero(~predicted & truth)
        dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
        if dice > best_dice:
            best_threshold, best_dice = float(threshold), float(dice)
    return round(best_threshold, 6)
