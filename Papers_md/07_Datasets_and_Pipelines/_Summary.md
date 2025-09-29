# Category 07: Datasets and Pipelines

Key resources for datasets, benchmarks, and processing pipelines.

---

## Technicolor Light‑Field Dataset and Processing Pipeline
- Dataset: multi‑view light‑field video captured by a synchronized 4x4 camera rig; provides calibrated intrinsics/extrinsics and processed pseudo‑rectified images.
- Pipeline: color homogenization (black level, bias/gain maps, aperture alignment, color correction), calibration with SBA, pseudo‑rectification, multi‑resolution depth estimation with ZNCC, and novel‑view rendering from 3D point reprojection.
- Outcome: a solid benchmark and complete pipeline that enables research on wide‑baseline light‑field video, with GPU‑friendly depth estimation and high‑quality results.
