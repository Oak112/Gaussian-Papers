# Category 01: Streaming and Online Training (FVV)

This category collects works on streaming reconstruction, online training, and efficient free‑viewpoint video (FVV) with Gaussians.

---

## 1. 3DGStream: On‑the‑Fly Training of 3D Gaussians for Efficient FVV
- Problem: Offline training and slow rendering make existing methods unsuitable for real‑time streaming.
- Approach: Two‑stage per‑frame pipeline with a Neural Transformation Cache (hash‑MLP that predicts per‑Gaussian motion) and adaptive addition of new Gaussians only where needed; only NTC parameters are carried across frames.
- Results: ~12 s per frame training, ~200 FPS rendering, small incremental storage; competitive quality.

## 2. V3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians
- Idea: Reformat per‑frame Gaussian attributes into standard 2D video tensors, then use hardware video codecs for compression/transport/decoding on mobile.
- Method: Morton reordering for spatial locality, grouped training with residual‑entropy and temporal losses; cross‑platform player reconstructs Gaussians per frame from decoded images.
- Outcome: Mobile‑friendly, highly compact representation with real‑time playback quality on devices.

## 3. Instant Gaussian Stream (IGS)
- Goal: Faster online reconstruction with generalization and reduced error accumulation.
- Method: A generalizable motion network (AGM‑Net) predicts motion for most frames with a single forward pass; periodic keyframes are optimized to correct drift and handle topology changes.
- Result: ~6x faster than 3DGStream with better PSNR; strong cross‑scene generalization.

## 4. S4D: Streaming 4D Reconstruction with 3D Control Points
- Key idea: Local 6‑DoF motion via discrete 3D control points; observable components from multi‑view optical flow, hidden components learned; keyframe refinement to limit drift.
- Benefit: Robust, efficient local motion modeling and fast convergence.

## 5. SwinGS: Sliding Window Gaussian Splatting
- Treat long videos as a stream of small incremental 3D models; Gaussians have explicit lifespans; sliding‑window training with constant‑size slices sent to the client.
- Achieves large reduction in per‑frame update size with competitive quality and FPS.

## 6. Motion Matters (ComGS): Compact Gaussian Streaming
- Replace point‑wise updates with a small set of motion keypoints that drive nearby Gaussians via learned influence fields; error‑aware correction only at keyframes.
- Storage drops by orders of magnitude with comparable quality and high FPS.

## 7. LongSplat: Online 3DGS from Long Sequences
- Gaussian‑Image Representation (GIR) provides a structured 2D grid that maps to dominant Gaussians along rays, enabling incremental updates and redundancy removal.
- Better quality and scalability on long sequences with significant compression.

## 8. Scale‑GS: Scalable Training on Streaming Content
- Multi‑scale anchor hierarchy with on‑demand activation via gradient thresholds; hybrid deform‑plus‑generate, bidirectional adaptive masks, and learnable pruning.
- Delivers high quality, low storage, and fast training and rendering.
