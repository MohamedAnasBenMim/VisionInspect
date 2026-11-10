from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import SyntheticInspectionDataset, generate_sample
from .inference import benchmark_onnx, benchmark_torch, export_onnx
from .metrics import calibrate_threshold, evaluate_predictions
from .models import classical_anomaly_map, model_size_mb
from .training import train_model


def main() -> None:
    test = SyntheticInspectionDataset(40, seed=10_011)
    validation = SyntheticInspectionDataset(32, seed=5_011)
    masks = np.stack([test[index][1].numpy()[0] for index in range(len(test))])
    labels = np.array([int(test[index][2]) for index in range(len(test))])
    started = time.perf_counter()
    baseline_maps = np.stack([classical_anomaly_map(sample.image) for sample in test.samples])
    baseline_validation = np.stack(
        [classical_anomaly_map(sample.image) for sample in validation.samples]
    )
    baseline_latency = (time.perf_counter() - started) * 1000 / len(test)
    validation_masks = np.stack(
        [validation[index][1].numpy()[0] for index in range(len(validation))]
    )
    baseline_threshold = calibrate_threshold(validation_masks, baseline_validation)
    baseline = evaluate_predictions(masks, baseline_maps, labels, threshold=baseline_threshold)

    result = train_model()
    example = generate_sample(50_001).image
    torch_latency = benchmark_torch(result.model, example)
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    torch.save(result.model.state_dict(), output / "visioninspect.pt")
    onnx_path = export_onnx(result.model, output / "visioninspect.onnx")
    onnx_latency = benchmark_onnx(onnx_path, example)
    report = {
        "dataset": {"train_images": 96, "test_images": 40, "resolution": "64x64"},
        "classical_baseline": {**baseline.to_dict(), "latency_ms": round(baseline_latency, 3)},
        "tiny_segmenter": {
            **result.report.to_dict(),
            "torch_latency_ms": round(torch_latency, 3),
            "onnx_latency_ms": round(onnx_latency, 3),
            "model_size_mb": round(model_size_mb(result.model), 4),
        },
    }
    Path("benchmarks/results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
