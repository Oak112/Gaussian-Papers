# Category 06: Compression and Streaming

Works on rate‑aware representations and transmission for dynamic Gaussians.

---

## 4DGC: Rate‑Aware 4D Gaussian Compression for Streamable FVV
- Representation: keyframe with full 3DGS plus a compact multi‑resolution motion grid to propagate Gaussians to later frames; sparse compensated Gaussians for newly revealed content.
- End‑to‑end compression during training: differentiable quantization, tiny implicit entropy models, and a rate‑distortion objective to balance quality and bitrate.
- Two‑stage training: first motion and its entropy model, then compensated Gaussians.
- Results: up to 16x or more bitrate reduction at similar quality; faster training/codec compared to prior art; adjustable quality via lambda.
