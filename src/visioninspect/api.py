from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from .models import TinySegmenter

app = FastAPI(title="VisionInspect", version="1.0.0")
model = TinySegmenter()
checkpoint = Path(os.getenv("VISIONINSPECT_MODEL", "artifacts/visioninspect.pt"))
if checkpoint.exists():
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
model.eval()


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parents[2] / "dashboard/index.html")


@app.post("/inspect")
async def inspect(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image")
    image = Image.open(io.BytesIO(await file.read())).convert("L").resize((64, 64))
    pixels = np.asarray(image, dtype=np.float32) / 255
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(pixels[None, None])))[0, 0].numpy()
    maximum, mean = float(scores.max()), float(scores.mean())
    return {
        "defective": maximum >= 0.45,
        "confidence": round(maximum, 4),
        "defect_area_pct": round(float(np.mean(scores >= 0.45) * 100), 3),
        "mean_anomaly_score": round(mean, 4),
    }
