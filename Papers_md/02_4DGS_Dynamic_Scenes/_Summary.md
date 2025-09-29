# Category 02: 4DGS for Dynamic Scenes

Works that extend Gaussian Splatting to spatiotemporal 4D representations for real‑time dynamic scene rendering.

---

## 1. 4D Gaussian Splatting (4D‑GS)
- Maintain a single canonical 3D Gaussian set plus a lightweight deformation network conditioned by 4D encodings (e.g., K‑Planes/HexPlane) to predict per‑time pose/shape.
- Real‑time rendering with reduced storage; strong results on synthetic and real data.

## 2. 1000+ FPS 4DGS (4DGS‑1K)
- Remove short‑lifespan and low‑contribution Gaussians with a spatio‑temporal variation score; render only active Gaussians using keyframe visibility masks; add light post‑processing.
- Achieves ~9x raster speedup and massive storage reduction with similar quality.

## 3. Fully Explicit Dynamic 3DGS (Ex4DGS)
- Separate static vs dynamic Gaussians; store sparse time parameters only at keyframes for dynamic ones; interpolate in between; progressive training and backtracking pruning.
- Faster training, lower storage, and high FPS with competitive quality.

## 4. 4D Scaffold GS with Dynamic‑Aware Anchor Growing
- Compress capacity into sparse 4D anchors aligned to a grid; each anchor generates a few neural 4D Gaussians; dynamic‑aware growth allocates capacity to under‑reconstructed regions.

## 5. Hybrid 3D–4DGS
- Convert slow‑varying Gaussians to pure 3D while keeping truly dynamic parts as 4D; unified rasterization for both branches; removes periodic opacity resets.

## 6. MEGA: Memory‑Efficient 4DGS
- Decompose color into DC (stored) + AC (predicted by a tiny MLP); entropy‑regularized deformation enlarges single‑Gaussian coverage; FP16 and compressed storage.
- Up to two orders of magnitude smaller with competitive quality/speed.

## 7. SWIFT4D
- Divide‑and‑conquer: learn a per‑Gaussian dynamicness score; apply 4D deformation only to dynamic points with compact 4D hash features; time‑importance pruning replaces opacity resets.
- Minutes‑level training, strong quality and compact models.
