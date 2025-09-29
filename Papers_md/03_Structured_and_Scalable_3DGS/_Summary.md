# Category 03: Structured and Scalable 3DGS

Papers on structured representations, anchor hierarchies, and scalable training for large scenes.

---

## 1. Scaffold‑GS: Structured 3D Gaussians for View‑Adaptive Rendering
- Represent the scene by sparse voxel‑aligned anchors. Each visible anchor decodes k neural Gaussians on the fly conditioned on view distance and direction.
- Anchor growing uses accumulated gradients to add anchors where needed; pruning removes low‑contribution anchors. Two‑stage filtering (frustum and opacity) speeds rendering.
- Results: comparable or better PSNR/SSIM/LPIPS vs 3D‑GS, similar or faster FPS, and 4x–10x lower storage across multiple datasets.
