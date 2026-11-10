from pathlib import Path

import numpy as np
import pytest
import torch

from visioninspect.dataset import SyntheticInspectionDataset, generate_sample, split_seeds
from visioninspect.inference import benchmark_onnx, benchmark_torch, export_onnx
from visioninspect.metrics import binary_auroc, calibrate_threshold, evaluate_predictions
from visioninspect.models import TinySegmenter, classical_anomaly_map, model_size_mb
from visioninspect.training import dice_loss, save_training_result, set_reproducible, train_model


def test_generator_is_reproducible_and_mask_aligned() -> None:
    first, second = generate_sample(21), generate_sample(21)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.label == 1
    assert first.image[first.mask == 1].mean() < first.image[first.mask == 0].mean()
    assert generate_sample(22).defect_type == "normal"


def test_dataset_augmentation_keeps_shapes() -> None:
    image, mask, label = SyntheticInspectionDataset(6, augment=True)[0]
    assert image.shape == mask.shape == (1, 64, 64)
    assert label.item() in (0, 1)


def test_split_has_no_leakage() -> None:
    train, validation, test = split_seeds(50)
    assert len(train) == 30 and len(validation) == len(test) == 10
    assert not (set(train) & set(validation) | set(train) & set(test) | set(validation) & set(test))


def test_segmenter_preserves_resolution_and_is_compact() -> None:
    model = TinySegmenter()
    assert model(torch.zeros(2, 1, 64, 64)).shape == (2, 1, 64, 64)
    assert model_size_mb(model) < 0.1


def test_classical_map_is_normalized() -> None:
    scores = classical_anomaly_map(generate_sample(23).image)
    assert scores.shape == (64, 64)
    assert 0 <= scores.min() <= scores.max() <= 1


def test_metrics_match_perfect_predictions() -> None:
    masks = np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]]])
    scores = masks * 0.9 + (1 - masks) * 0.1
    report = evaluate_predictions(masks, scores, np.array([0, 1]), threshold=0.5)
    assert report.image_auroc == report.pixel_auroc == 1
    assert report.precision == report.recall == report.dice == 1
    assert report.to_dict()["threshold"] == 0.5


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(np.zeros((2, 2)), np.zeros((2, 3)), np.array([0, 1]))


def test_threshold_calibration_finds_separation() -> None:
    masks = np.array([[[0, 0], [0, 1]]], dtype=float)
    scores = np.array([[[0.1, 0.2], [0.3, 0.9]]])
    threshold = calibrate_threshold(masks, scores)
    assert 0.3 < threshold <= 0.9


def test_auroc_rejects_single_class() -> None:
    with pytest.raises(ValueError):
        binary_auroc(np.ones(3), np.arange(3))


def test_dice_loss_improves_for_correct_logits() -> None:
    target = torch.tensor([[[[1.0, 0.0]]]])
    assert dice_loss(torch.tensor([[[[5.0, -5.0]]]]), target) < dice_loss(
        torch.zeros_like(target), target
    )


def test_seed_controls_torch() -> None:
    set_reproducible(42)
    first = torch.rand(3)
    set_reproducible(42)
    torch.testing.assert_close(first, torch.rand(3))


def test_training_export_and_inference_benchmark(tmp_path: Path) -> None:
    result = train_model(seed=3, epochs=1, samples=16)
    assert len(result.losses) == 1
    save_training_result(result, tmp_path)
    assert (tmp_path / "visioninspect.pt").exists()
    model_path = export_onnx(result.model, tmp_path / "visioninspect.onnx")
    image = generate_sample(101).image
    assert benchmark_torch(result.model, image, runs=2) >= 0
    assert benchmark_onnx(model_path, image, runs=2) >= 0
