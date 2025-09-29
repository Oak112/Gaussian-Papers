# Category 05: Tracking, Deformation, and Monocular Dynamic GS

Papers targeting motion tracking, non‑rigid deformation, or monocular inputs for dynamic Gaussian rendering.

---

## Deformable 3D Gaussians for Monocular Dynamic Scenes
- Canonical 3D Gaussian set plus an MLP deformation field that outputs per‑time deltas for position, rotation, and scale; joint optimization through differentiable splatting.
- Annealing smoothing for robustness to pose noise; real‑time rendering with strong quality on synthetic and real data.

## Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis
- Keep per‑Gaussian appearance constant over time and optimize only 6‑DoF motion per frame; analysis‑by‑synthesis with physics‑motivated local regularizers.
- Delivers strong dense 6‑DoF tracking and high‑FPS rendering without external correspondences.

## LocalDyGS: Adaptive Local Implicit Feature Decoupling
- Decompose a large dynamic scene into many local spaces defined by seed points; generate temporary temporal Gaussians on demand; global 4D residual for dynamics and adaptive seed growth.
- Robust to large‑range motions (e.g., sports) while remaining fast and compact.

## E‑D3DGS: Per‑Gaussian Embedding‑Based Deformation
- Define deformation as a function of a learnable per‑Gaussian embedding and a temporal embedding, rather than of spatial coordinates; coarse‑to‑fine temporal factors and local‑smoothness regularization.
- Sharper dynamic details and competitive speed/size across datasets.
