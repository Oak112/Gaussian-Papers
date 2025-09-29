# Category 04: NeRF, Meshes, and Dynamic Baselines

NeRF family methods, explicit grid/mesh baselines, and dynamic benchmarks commonly compared with Gaussians.

---

## 1. Mip‑NeRF (ICCV 2021)
- Anti‑aliasing via cone tracing and integrated positional encoding; single multi‑scale MLP.
- Large improvements on multi‑scale settings versus vanilla NeRF while being faster.

## 2. Mip‑NeRF 360 (CVPR 2022)
- Contracted space for unbounded scenes, proposal‑to‑backbone online distillation, and distortion regularization.
- Strong quality on complex real indoor/outdoor scenes with moderate compute.

## 3. D‑NeRF (CVPR 2021)
- Deformation to a canonical space plus a canonical radiance field; time input with PE and standard volume rendering.
- Enables dynamic scenes but is compute‑heavy.

## 4. K‑Planes (CVPR 2023)
- Factorize space‑time using pairs of 2D planes (xy/xz/yz and xt/yt/zt). Per‑point features are multiplied then decoded linearly or with a small MLP.
- Efficient, interpretable, and competitive across static, dynamic, and appearance‑varying tasks.

## 5. HexPlane (CVPR 2023)
- Six‑plane factorization with multiply‑then‑concat fusion and lightweight decoders; optional SH.
- Orders‑of‑magnitude faster training than classic dynamic NeRF with competitive quality.

## 6. HyperNeRF (TOG 2021)
- Higher‑dimensional template with deformable slicing to handle topology change; combines with spatial deformation.
- Better interpolation and fewer artifacts in topologically varying scenes.

## 7. HyperReel (CVPR 2023)
- Ray‑conditioned sampling network plus a keyframe dynamic tensor representation; few high‑value samples per ray.
- Strong quality, faster rendering, and compact per‑frame memory.

## 8. DyNeRF (CVPR 2022)
- Time is conditioned by a per‑frame latent code. Train on sparse keyframes first, then interpolate and fine‑tune; time‑aware importance sampling.
- High‑quality 6‑DoF video with large compute reduction vs per‑frame NeRF.
