from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .models import TinySegmenter


def export_onnx(model: TinySegmenter, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 1, 64, 64),
        destination,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=True,
    )
    return destination


def benchmark_torch(model: TinySegmenter, image: np.ndarray, runs: int = 40) -> float:
    tensor = torch.from_numpy(image[None, None]).float()
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(tensor)
        started = time.perf_counter()
        for _ in range(runs):
            model(tensor)
    return (time.perf_counter() - started) * 1000 / runs


def benchmark_onnx(model_path: Path, image: np.ndarray, runs: int = 40) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    value = image[None, None].astype(np.float32)
    for _ in range(5):
        session.run(None, {"image": value})
    started = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"image": value})
    return (time.perf_counter() - started) * 1000 / runs
