# VisionInspect Tiny Segmenter — Model Card

## Intended use

Educational and portfolio-grade segmentation of visible surface scratches and dents. The model demonstrates the engineering path from dataset preparation through optimized serving.

## Data

Training and evaluation currently use deterministic 64×64 synthetic textured surfaces. Seed ranges are separated between training, validation, and test generation. The validation set calibrates the segmentation threshold; the test set is used once for final reporting.

## Metrics

The committed benchmark reports image AUROC, pixel AUROC, precision, recall, Dice, model size, eager PyTorch latency, and ONNX Runtime latency. Results apply only to the included generator and local CPU environment.

## Limitations and risks

Synthetic textures do not represent the full variation of industrial optics, materials, contamination, motion blur, or rare defects. A high benchmark score must not be interpreted as readiness for automated rejection decisions. Real deployment requires representative data, lot-separated validation, uncertainty monitoring, reviewer escalation, and safety acceptance criteria.
