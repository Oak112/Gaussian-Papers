

---

## Page 1

LongSplat: Online Generalizable 3D Gaussian Splatting from Long Sequence
Images
Guichen Huang1,2, Ruoyu Wang3, Xiangjun Gao4, Che Sun2, Yuwei Wu2,1*, Shenghua Gao3,5, Yunde Jia2,1
1Beijing Institute of Technology2Shenzhen MSU-BIT University3Transcengram
4The Hong Kong University of Science and Technology5The University of Hong Kong
Abstract
3D Gaussian Splatting achieves high-fidelity novel view
synthesis, but its application to online long-sequence sce-
narios is still limited. Existing methods either rely on slow
per-scene optimization or fail to provide efficient incremen-
tal updates, hindering continuous performance. In this pa-
per, we propose LongSplat, an online real-time 3D Gaus-
sian reconstruction framework designed for long-sequence
image input. The core idea is a streaming update mech-
anism that incrementally integrates current-view observa-
tions while selectively compressing redundant historical
Gaussians. Crucial to this mechanism is our Gaussian-
Image Representation (GIR), a representation that encodes
3D Gaussian parameters into a structured, image-like 2D
format. GIR simultaneously enables efficient fusion of
current-view and historical Gaussians and identity-aware
redundancy compression. These functions enable online re-
construction and adapt the model to long sequences without
overwhelming memory or computational costs. Further-
more, we leverage an existing image compression method
to guide the generation of more compact and higher-quality
3D Gaussians. Extensive evaluations demonstrate that
LongSplat achieves state-of-the-art efficiency-quality trade-
offs in real-time novel view synthesis, delivering real-time
reconstruction while reducing Gaussian counts by 44%
compared to existing per-pixel Gaussian prediction meth-
ods.
1. Introduction
Growing interest in 3D scene reconstruction and novel
view synthesis has led to rapid advancements in the field,
among which 3D Gaussian splatting(3DGS) [19, 51, 16, 24]
has gained particular attention for its effectiveness. Despite
its impressive rendering speed at inference time, most ex-
isting methods still rely on slow, per-scene optimization for
reconstruction, which can take minutes to hours even for
*Corresponding author.moderately sized environments. This slow optimization is a
significant barrier for applications requiring fast, real-time
perception and response, such as embodied AI and robotics,
where timely adaptation to dynamic environments is essen-
tial. To address these challenges, there is an increasing need
for systems that can process long sequences of visual data in
real-time, dynamically updating with each new frame input
while ensuring high-quality reconstruction.
Recent efforts have aimed to improve reconstruction ef-
ficiency by developing generalizable splatting models that
directly predict 3D Gaussian parameters from images in
a feed-forward manner. These methods [6, 46, 20] sig-
nificantly reduce processing time and perform well under
sparse-view settings. However, their performance often de-
grades when applied to long sequences or dense multi-view
scenarios: the reconstructed Gaussians become increasingly
redundant and noisy, resulting in artifacts such as floating
points and blurred regions. Moreover, memory and com-
putational costs grow rapidly as more views are processed,
making these approaches difficult to scale to real-world ap-
plications involving hundreds of frames. These limitations
arise primarily from two factors: a lack of global histor-
ical Gaussians modeling and the absence of an efficient
incremental update mechanism, both of which are essen-
tial for robust long-term reconstruction. Although some
recent works [41, 42, 43, 56, 20] extending generalizable
3D GS to sequential inputs sets, they still struggle with
incremental updates or rely on fixed-length reconstruction
pipelines, limiting their flexibility and scalability in online
long-sequence scenarios.
In this paper, we propose LongSplat, an online 3DGS
framework designed for real-time, incremental reconstruc-
tion from long-sequence images. Its core innovation lies in
an incremental update mechanism that integrates current-
view observations while selectively compressing redun-
dant historical Gaussian. This mechanism efficiently per-
forms two key operations per frame: (1) Adaptive Com-
pression: selectively compressing accumulated Gaussians
from past views to eliminate redundancy and minimize
storage/rendering costs, and (2) Online Integration: fus-arXiv:2507.16144v1  [cs.CV]  22 Jul 2025

---

## Page 2

ing current-view Gaussians with the historical state. These
strategies aim to mitigate a core limitation of generalizable
3DGS: per-pixel prediction inherently produces dense but
redundant Gaussians. By progressively refining the Gaus-
sian field over time, our method seeks to improve scalabil-
ity and memory usage while enhancing consistency across
views. In addition, the compression mechanism reduces re-
dundancy and offers a potential path toward dynamic scene
modeling, where outdated or redundant elements can be re-
moved in a lightweight, incremental manner without repro-
cessing the entire sequence.
Specifically, we propose Gaussian-Image Representation
(GIR) that projects 3D Gaussian parameters into a struc-
tured 2D image-like format. This representation enables on-
line reconstruction by facilitating the propagation of infor-
mation across views and supporting localized compression.
To enhance cross-view interaction, GIR projects historical
Gaussians into the current frame, enabling feature-level fu-
sion. This fusion not only improves the spatial consistency
of the reconstructed 3D Gaussian field, but also provides a
structured basis for subsequent compression of redundant
historical information. In addition, GIR plays a central role
in localized compression by maintaining the mapping be-
tween 2D projections and their corresponding historical 3D
Gaussians. This identity-aware structure makes 3DGS more
tractable and removes redundant splats accumulated over
time. Such compression not only reduces memory and ren-
dering cost, but also improves visual quality by eliminating
overlapping or outdated Gaussians. Furthermore, we lever-
age GIR’s image-like structure to apply supervision from
ground-truth 3DGS, using an optimized per-scene Gaussian
dataset constructed with existing image compression tech-
niques [11]. This strategy improves both compactness and
fidelity of the learned 3D Gaussians without requiring full
3D loss computation.
Through extensive evaluations, we demonstrate that
LongSplat achieves state-of-the-art efficiency-quality trade-
offs in real-time novel view synthesis. Our method achieves
real-time rendering and reduces Gaussian counts by 44% on
DL3DV[22]. Moreover, LongSplat outperforms the base-
line by 3.6 dB in PSNR on the DL3DV benchmark and
exhibits superior scalability for long-sequence scene recon-
struction. By efficiently processing long-sequence visual
data, LongSplat opens up new possibilities for real-time 3D
perception in applications that require handling extensive
visual inputs. Our contributions can be summarized as fol-
lows:
• We propose LongSplat, a real-time 3D Gaussian re-
construction framework tailored for arbitrary-view,
long-sequence image inputs. By introducing a 3D
Gaussian updating mechanism that selectively com-
presses redundant historical Gaussians and incremen-
tally integrates current-view observations, LongSplatenables scalable, memory-efficient reconstruction and
real-time novel view synthesis.
• We introduce Gaussian-Image Representation(GIR), a
structured 2D representation of 3D Gaussians that en-
ables efficient historical feature fusion, redundancy
compression, lightweight 2D operations, and GIR-
space supervision.
• Extensive experiments show that LongSplat achieves
state-of-the-art real-time novel view synthesis, provid-
ing real-time rendering and reducing Gaussian counts
by 44% compared to existing methods.
2. Related Work
Traditional 3D Gaussian Splatting. Traditional 3D Gaus-
sian Splatting (3DGS) methods [19, 51, 16, 24, 13] have
emerged as a powerful paradigm for high-fidelity novel
view synthesis, leveraging explicit 3D Gaussian primi-
tives to represent scenes. Unlike Neural Radiance Fields
(NeRFs) [26, 3, 4, 12, 5, 23, 38], which rely on com-
putationally intensive ray-marching-based volume render-
ing, 3DGS achieves real-time rendering speeds through tile-
based rasterization of differentiable Gaussian primitives.
These methods optimize Gaussian parameters (e.g., posi-
tion, scale, rotation, and opacity) through per-scene opti-
mization, resulting in high-quality novel view synthesis and
fast rendering. However, the per-scene optimization process
is inherently time-consuming, which significantly limits its
applicability in real-time perception tasks.
Generalizable 3D Gaussian Splatting. Inspired by prior
progress in generalizable NeRFs [50, 21, 14, 7, 49], re-
cent efforts have focused on generalizable 3D Gaussian
Splatting methods that enable feed-forward prediction of
3D Gaussians from input images[30, 33, 27, 2, 47, 18, 10,
36, 52, 44]. Pioneering works such as PixelSplat [6] and
GPS-Gaussian [53] explore feed-forward 3D Gaussian re-
construction using epipolar geometry from just two input
views, achieving fast and high-quality novel view synthesis.
MVS-based methods [8, 46, 20] extends this direction by
leveraging multi-view geometry through cost volumes for
enhanced accuracy and generalization. Adaptive Gaussian
[1] departs from fixed pixel-wise Gaussian representations
by dynamically adapting the distribution and number of 3D
Gaussians based on local geometric complexity.
Other approaches extend these approaches to sequential
inputs: For example, FreeSplat [42, 43] proposes a cross-
view aggregation scheme and a pixel-wise triplet fusion
strategy that jointly optimizes overlapping view regions,
enabling free-view synthesis with geometrically consistent
scene reconstruction. Yet, due to its latent GS represen-
tation, the heavy computational overhead limits scalability

---

## Page 3

𝑰𝒕
Nearest N views
Feature extractor𝑮𝒍𝒐𝒃𝒂𝒍  𝟑𝑫𝑮𝑺  𝑮𝒕−𝟏
updateGussian  Image
Representation
𝑮𝒍𝒐𝒃𝒂𝒍  𝟑𝑫𝑮𝑺  𝑮𝒕Encoder
TransformGI 𝒕−𝟏 → t
BackboneM t
GI t 𝑨𝒅𝒅  𝟑𝑫𝑮𝑺  𝑮𝒕𝑫𝒆𝒍𝒆𝒕𝒆  𝟑𝑫𝑮𝑺  𝑮𝒕−𝟏
𝑭𝒉
𝑭𝒄Selected Gaussian
Deleted Gaussian
Added GaussianFigure 1. Overview of the Longsplat framework. Given an input image sequence {It}T
t=1, our model incrementally constructs a global
3D Gaussian scene representation Ggthrough iterative frame-wise updates. At each timestep t, we extract two complementary feature
streams: (1) a multi-view spatial feature map Fcfrom the current frame and its temporally adjacent neighbors using the DepthSplat
pipeline, providing local geometry and appearance cues; and (2) a historical context feature map Fhby rendering the accumulated global
Gaussians Gg
t−1into a 2D Gaussian-Image Representation (GIR) via differentiable projection. These streams are fused via a transformer-
based module to produce a fused representation Ff, from which we derive an adaptive update mask ˆMtand generate current-frame
Gaussians Gc
t. The global representation Ggis then selectively updated, enabling efficient long-sequence reconstruction with spatial-
temporal consistency.
to long-sequence inputs, making it less suitable for real-
time processing of long-sequence inputs. Long-LRM [56]
leverages a hybrid architecture merging and Gaussian prun-
ing—to process up to 32 views in a single feed-forward
pass, reconstructing entire scenes with performance com-
parable to optimization-based methods. Despite these ad-
vances, its reliance on fixed-length reconstruction limits
flexibility, making it unsuitable for dynamic, open-ended
sequences. Zpressor [41] significantly reduces memory
requirements via anchor-frame propagation while achiev-
ing high reconstruction quality. Compared to such anchor-
frame methods that still rely on per-frame prediction and
fixed-feature transfer, our approach supports dynamic up-
dates to Gaussians across frames, enabling better use of
historical information. As a per-pixel predictor, it is also
compatible with our framework and can serve as a feature
encoder within our fusion pipeline. We propose LongSplat,
an online 3D Gaussian reconstruction framework specifi-
cally designed for long-sequence inputs, supporting scal-
able temporal modeling under streaming and interactive
conditions. Our approach enables real-time editing and
streaming integration without compromising reconstruction
fidelity through 3DGS updating and history view fusion
techniques.
Indoor Scene Reconstruction Indoor scene reconstruction
has been extensively studied through various paradigms, in-
cluding voxel-based methods [34, 29], TSDF fusion [32],and Nerf-based approaches [55, 54, 31, 48]. While these
methods excel in geometric reconstruction, they often lack
the ability to perform photorealistic novel view synthesis.
Recent advancements [25, 35, 17, 15, 28, 45] introduce
3D Gaussian-based representations that improve rendering
speed and quality by avoiding unnecessary spatial computa-
tions, but still rely on external or ground-truth depth maps.
In contrast, our method operates end-to-end without depth
supervision, leveraging photometric losses to achieve ac-
curate 3D Gaussian localization and scalable scene recon-
struction.
3. Method
3.1. Vanilla 3D Gaussian Splatting
The vanilla 3D Gaussian Splatting (3DGS) represents
scenes as a collection of anisotropic Gaussians G=
{µ,Σ, c, α}, where µdenotes position, Σthe covariance
matrix, cthe color, and αthe opacity. The rendering process
follows alpha compositing along each ray:
C(u, v) =X
i∈Sciαii−1Y
j=1(1−αj) (1)
where Srepresents the set of Gaussians sorted by depth.
The Gaussian parameters are optimized by a photometric

---

## Page 4

loss to minimize the difference between renderings and im-
age observations.
3.2. Longsplat Pipeline
Longsplat processes an input image sequence IT
t=1and
iteratively updates the global Gaussian representation Gg.
At each timestep t, the model takes in the current frame
Itand jointly leverages both multi-view spatial features Fc
and historical global context features Fhto produce the
current per-frame Gaussian predictions Gc
tand update the
global scene representation Gg
taccordingly. This process
involves extracting two key feature streams:
(1) Multi-view Spatial Feature Map Fc. To ensure ge-
ometric consistency and accurate depth-aware representa-
tion, we extract multi-view features from the current frame
and its Ntemporally adjacent neighbors. Specifically, we
adopt the feature extraction pipeline from Depthsplat[46],
which produces both dense feature maps and per-pixel raw
Gaussian predictions. These features capture the local 3D
structure and provide a strong prior for the current frame’s
geometry.
(2) Historical Feature Map Fh. To incorporate long-
range temporal information, we introduce a Gaussian-
Image Representation (GIR) that efficiently encodes ac-
cumulated global Gaussians Gg
t−1into the current cam-
era view. Using a differentiable perspective projection
operator Π, we render the historical Gaussians into a
structured, image-aligned 2D format, denoted as ˆGIg
t=
Π(Gg
t−1, Kt, Tt), where KtandTtare the camera intrin-
sics and extrinsics. The projected Gaussian-Image is then
encoded via a shallow CNN to yield the historical context
feature map Fh.
History Fusion. The spatially aligned features Fcand
temporally accumulated features Fhare fused through a
transformer-based module that attends across both streams
to produce an enriched representation Ff. This fused fea-
ture encodes both current appearance and long-term con-
text. From Ff, the module predicts an update mask weight-
ingˆMt∈[0,1]H×W, that determines the compression
weighting for each pixel.
Compressed Module. To determine which Gaussians
should be retained or delete, we apply a thresholding strat-
egy to the predicted soft update weights ˆMt, producing a
binary confidence mask Mt∈ {0,1}H×Wbased on a tun-
able confidence threshold. This binary mask identifies high-
confidence pixels suitable for global storage and further op-
timization. We then use Mtto filter both the enriched fea-
tureFfand its corresponding per-pixel learnable embed-
dings, effectively compressing the Gaussian splatting quan-
tities by discarding uncertain or redundant Gaussians. The
Gussian  Image
RepresentationGlobal GS
Update
3D IoU
Mask Los sHistory Fusion+
𝑮𝒍𝒐𝒃𝒂𝒍  𝟑𝑫𝑮𝑺  𝑮𝒕−𝟏
𝟑𝑫𝑮𝑺  𝑫𝒂𝒕𝒂𝒔𝒆𝒕  𝑮𝒈𝒕GI 𝒕−𝟏→𝒕GI 𝒕
GI 𝒕−𝟏→𝒕
GI 𝒕−𝟏→𝒕 Efficient 3D IoUM 𝒕
Geometry
LossGI 𝒈𝒕→𝒕 GI 𝒕
0.0%0.0%1.0%0.0%
90.0%Figure 2. Overview of the proposed Gaussian-Image Represen-
tation (GIR) and its four core capabilities. GIR encodes per-pixel
Gaussian parameters into a structured 2D image space, enabling
efficient and flexible 3D reasoning. (a)History Fusion: GIR al-
lows temporally consistent fusion of 3D Gaussians across multiple
frames by leveraging shared Gaussian IDs. (b)Global Update:
GIR supports gradient-based updates to the global 3D Gaussian
field from localized image-space errors. (c)Geometry Supervi-
sion: GIR enables pixel-wise geometry loss against ground-truth
3D Gaussians, providing strong spatial supervision. (d)Efficient
3D IoU: By tracking 3D instance masks via Gaussian IDs, GIR
enables differentiable 3D IoU estimation and 3D IoU-based mask
loss.
selected features and embeddings are fed into a lightweight
transformer and a shared Gaussian head to produce the fi-
nal set of compressed per-frame Gaussians Gc
t. These are
then flagged as valid candidates for global scene represen-
tationGg
t, enabling consistent, efficient, and scalable scene
reconstruction over long sequences.
3.3. Gaussian-Image Representation
We propose a Gaussian-Image Representation (GIR)
that encodes per-pixel Gaussian attributes into a structured
2D format. This compact view-aligned representation en-
ables efficient memory usage, supports localized updates,
and bridges the gap between 3D scene modeling and 2D
image-space supervision.
Formally, for each pixel (u, v)in a rendered view, the
Gaussian-Image Gv∈RH×W×10stores the projected
2D position µuv, the upper-triangular components of the
covariance matrix vech (Σuv), opacity αuv, and a unique
Gaussian identifier IDuv:
Gv(u, v) = [µuv,vech(Σuv), αuv, IDuv] (2)
Unlike standard 3D Gaussian Splatting (3DGS), which

---

## Page 5

blends all overlapping Gaussians along a ray, our GIR
adopts a sparse rendering strategy in which each pixel is
associated with only a single dominant Gaussian. We con-
sider two selection methods:
(1) Nearest Rendering. The first visible Gaussian (with
opacity above threshold τ) is selected:
A(r) =Gkwhere k= min {i|αi> τ} (3)
(2) Most-Contributive Rendering. The Gaussian that
contributes most to the final color along the ray is chosen,
based on transmittance-weighted opacity:
k= arg max
i
αii−1Y
j=1(1−αj)
 (4)
These rendering strategies produce a clean and disentan-
gled projection of the 3D Gaussians into 2D space, avoid-
ing transparency-induced blurring and enabling more stable
per-pixel learning. To enable consistent mapping between
image pixels and their 3D counterparts, we also generate
a Gaussian ID map ID∈ZH×W, recording the index of
the selected Gaussian at each pixel. This ID map acts as
a lightweight and deterministic link for downstream tasks
such as pruning, compression, and temporal fusion.
Moreover, GIR facilitates the creation of training data
via a self-supervised bootstrapping process. By first run-
ning per-scene optimization (e.g., via LightGaussian [11])
to obtain high-fidelity 3D Gaussians, we render their cor-
responding Most-Contributive GIRs as supervision tar-
gets. These rendered representations serve as ground truth
for learning to predict Gaussian parameters directly from
new views, enabling the construction of scalable training
datasets without external annotations.
Through this design, GIR provides a unified interface
for both training and inference: it enables 2D convolution-
based processing of 3D attributes, supports efficient view-
wise updates, and grounds learning in a consistent, differ-
entiable projection of the 3D scene.
3.4. Training
Our training framework supervises predicted Gaussians
using both per-view compressed parameters from Light-
Gaussian [11] and rendered RGB images, ensuring con-
sistency in geometry, appearance, and compactness. The
training loss integrates 3D alignment, mask-guided opac-
ity modulation, and photometric supervision across selected
target views.
Image Reconstruction Loss. To provide dense supervi-
sion, we randomly sample a set of target views {It}from
the input sequence and render the current set of predicted
Gaussians Gtinto corresponding RGB images ˆIt. We then
compute a photometric loss between the rendered imageand the ground truth:
Lrgb=X
t∈T∥ˆIt−It∥1 (5)
This loss provides direct gradients for updating Gaussian
attributes, such as color, position, and opacity, and ensures
consistency between the 3D representation and the input
imagery. Moreover, it enables self-supervised refinement
of the predicted confidence mask via opacity modulation
(described below).
Geometric Alignment. In addition to image-level su-
pervision, we enforce consistency between the predicted
Gaussians and the ground-truth compressed set {Ggt
v}. The
3D position alignment loss minimizes spatial deviation:
Lxyz=1
|V|X
v∈V∥µpred
v−µgt
v∥1 (6)
The covariance consistency loss promotes shape alignment:
LΣ=1
|V|X
v∈V∥Σpred
v−Σgt
v∥1 (7)
Together, they form the image-space geometric loss:
Lgeo=Lxyz+λΣLΣ (8)
where λΣ= 0.5.
Mask-Guided Learning. To enable soft pruning of re-
dundant or outdated Gaussians, we introduce a learnable
visibility mask Mt∈[0,1]H×W, which modulates the ren-
dered opacity:
αuv
mod=Mt(u, v)·αuv(9)
This mechanism allows the model to compress unnecessary
Gaussians while retaining informative ones, guided by both
geometric overlap and photometric supervision.
To supervise the mask, we compute a pairwise 3D over-
lap score based on Oriented Bounding Boxes (OBBs) - the
minimal rectangular boxes enclosing each 3D Gaussian el-
lipsoid. Unlike conventional symmetric IoU, we define a
view-asymmetric overlap metric:
IoUp= max
q∈N|OBB p∩OBB q|
|OBB p|(10)
This formulation penalizes historical Gaussians whose
OBBs unnecessarily cover fine-scale details in the current
view. In other words, it encourages the model to mask out
large, outdated Gaussians when they are locally replaceable
by smaller, view-specific ones.
Enabled by the Gaussian-Image Representation (GIR),
these 3D geometric comparisons are efficiently reformu-
lated as local, grid-aligned operations. Each Gaussian only

---

## Page 6

needs to check collisions with neighbors in a fixed spa-
tial window, allowing the entire process to be implemented
as parallel, pixel-wise computations on the GPU—without
global scene traversal. This localized design ensures both
computational efficiency and precise redundancy estimation
during training.
To supervise Mt, we treat Gaussians with high overlap
as redundant and assign ground truth Mgt
t= 1, and others
withMgt
t= 0. We then apply a weighted binary cross-
entropy loss:
Lmask=1
|Ω|X
(u,v)∈Ωλ(u, v)·BCE 
Mt(u, v),Mgt
t(u, v)
(11)
where Ωis the set of all pixel locations, and the weight func-
tion is defined as:
λ(Mgt
t) =(
λpos,ifMgt
t(u, v) = 1
λneg,ifMgt
t(u, v) = 0(12)
We set λpos> λ negto counteract the photometric loss,
which implicitly encourages maintaining opacity. Our mask
loss explicitly penalizes geometric redundancy, ensuring
better spatial compactness and consistency across frames.
Total Objective. The final loss combines all components
into a multi-task objective:
Ltotal=Lrgb+Lgeo+Lmask. (13)
4. Experiments
Implementation Details We adopt DepthSplat[46] as
our baseline framework while keeping all its parameters
fixed during training. The feature representations and Gaus-
sian splatting outputs from DepthSplat are directly utilized
as our model inputs. During rendering, we consistently use
10 target views for loss computation, ensuring multi-view
consistency in the optimization process. For optimization,
we employ the AdamW optimizer with a base learning rate
of1×10−4. During training, the number of input views is
randomly sampled between 2 and 8. The model is trained
with a resolution downsampling factor of 1/8 (i.e., 256 ×448
resolution). For hardware configuration, we employ 8 ×
RTX 4090 GPUs to perform 10K base training iterations
for the uncompressed model, followed by an additional 10K
iterations on 4 ×H100 GPUs for the compressed version.
Training Datasets We conduct training on the DL3DV-
10K[22] dataset, which consists of 9,896 training scenes
and 140 official test scenes. Additionally, we provide op-
tional auxiliary data reconstructed through DepthSplat (24-
view inputs) with LightGaussian compression, which con-
tributes to conditional auxiliary losses when available dur-
ing training. We selected 6,845 scenes meeting our quality
criteria: compression rate >30% and PSNR >28.0. These
selected scenes achieve an average PSNR of 31.50 with
69.60% average compression rate.4.1. DL3DV Benchmark Evaluation
Table 1. Multi-view comparison with state-of-the-art methods
Views Method PSNR ↑SSIM ↑LPIPS ↓c-ratio ↑
12MVSplat-360 17.05 0.4954 0.3575 0.00%
DepthSplat 22.02 0.7609 0.2060 0.00%
Ours 22.68 0.7824 0.1923 0.00%
Ours-c 21.69 0.7482 0.2213 25.52%
50MVSplat-360 OOM / / /
DepthSplat 21.39 0.7341 0.2212 0.00%
Ours 23.71 0.8159 0.1683 0.00%
Ours-c 23.54 0.8056 0.1742 43.77%
120DepthSplat 17.77 0.5899 0.3622 0.00%
Ours 21.02 0.7176 0.2608 0.00%
Ours-c 21.34 0.7345 0.2449 44.37%
Abbreviations : ”c-ratio” shows the percentage of compressed Gaussians. ”Ours-c”
is our compressed strategy applied.
Quantitative Results. As shown in Table 1, our method
consistently outperforms DepthSplat across all view counts,
achieving state-of-the-art results in both quality and com-
pactness. We evaluate under 12, 50, and 120 context-view
settings, corresponding to 100, 50, and 120 fixed target
views, respectively. For large-view settings (50 and 120
views), we adopt DepthSplat’s MVS-based feature extrac-
tion with temporal sampling at fixed 10-frame intervals to
ensure stable inputs. Our method then performs sequential
3D Gaussian reconstruction guided by these features.
For evaluation, we test under 12, 50, and 120-view con-
figurations with fixed target sets to ensure fairness and scal-
ability. Under sparse 12-view input, our full model achieves
22.68 PSNR, outperforming DepthSplat (22.02) with im-
proved SSIM and LPIPS. Our compressed variant (Ours-
c) retains competitive quality (21.69 PSNR) while reduc-
ing Gaussian count by 25.52%, demonstrating effective
compactness. As view count increases, DepthSplat’s per-
formance degrades due to uncontrolled Gaussian growth
(17.77 PSNR at 120 views), while our approach remains
stable. At 50 views, our full model reaches 23.71 PSNR
(+2.32 dB over DepthSplat), and Ours-c achieves 23.54
while removing 43.77% of Gaussians. At 120 views, Ours-
c still maintains high fidelity (21.34 PSNR, 0.2449 LPIPS)
with 44.37% compression. These results highlight our
method’s scalability and efficient long-sequence reconstruc-
tion.
Qualitative Results. Figure 3 shows qualitative results
on several DL3DV scenes with 12 input views. We com-
pare our method (Ours), its compressed variant (Ours-c),
MVsplat360[9], and our baselines, DepthSplat[46]. Depth-
Splat tends to produce floaters and blurred surfaces in com-
plex regions, while MVsplat360 occasionally exhibits struc-
tural inconsistencies when applied to long sequences. Our

---

## Page 7

Ref. rgb GT Ours -c Depthsplat Ours MVsplat360
… … …… … … … … …
… … …
 … … …
Figure 3. Novel view synthesis on 12 context views. Ref, GT are the input image and the ground truth. Ours-c delete the mask regions.
Ours, MVsplat360, and Depthsplat are full-scene results. Our method better removes floaters and preserves fine details, producing more
accurate and consistent renderings. )
method generally produces sharper and more consistent re-
constructions, with clearer geometry and fewer visual ar-
tifacts. Notably, the compressed variant (Ours-c) main-
tains comparable quality, and in some cases—such as the
first row—shows improved clarity in fine details like book
spines. This suggests that removing low-confidence Gaus-
sians may help reduce visual clutter and enhance overall
fidelity.
4.2. Ablation and Analysis
Ablation Study on Component-wise Contributions.
Table 2 reports the incremental impact of each core module.
Adding Unet refinement ( U) to the baseline improves PSNR
from 21.39dB to 21.71dB. History fusion ( F) brings a larger
boost to 22.78dB PSNR and significantly lowers LPIPS,
highlighting the value of temporal context. Introducing
3D supervision ( D) further enhances alignment, reaching
23.12 PSNR. We then examine compression and mask-
ing. Applying a fixed compression mask on the 3D dataset
(+D&M) reduces Gaussians by 43.77% but lowers PSNR
to 22.47dB, indicating that static pruning can discard rele-
vant points. Introducing the learned compression module
(C) without masking restores high fidelity (23.71dB, SSIM
0.8159, LPIPS 0.1683) while keeping c-ratio at 0%. Finally,
combining compression with adaptive masking (+C&M)Table 2. Ablation study on core modules
ConfigComponents Metrics
U F D C M PSNR ↑ SSIM↑ LPIPS↓ c-ratio↑
Baseline – – – – – 21.39 0.7341 0.2212 0.00
+ Unet ✓ – – – – 21.71 0.7463 0.2172 0.00
+ Fusion ✓ ✓ – – – 22.78 0.7934 0.1777 0.00
+ 3D Dataset ✓ ✓ ✓ – – 23.12 0.8087 0.1679 0.00
+ 3D Dataset(M) ✓ ✓ ✓ –✓22.47 0.7793 0.1851 43.77
+ Compression ✓ ✓ ✓ ✓ – 23.71 0.8159 0.1683 0.00
+ Compression(M) ✓ ✓ ✓ ✓ ✓ 23.54 0.8056 0.1742 43.77
Abbreviations: U: Unet, F: History fusion, D: 3D dataset, C: Compress module, M:
Compress mask. Note: Context-view and target-view sequences are both 50 frames.
retains most of this gain (PSNR 23.54dB, SSIM 0.8056,
LPIPS 0.1742) at a 43.77% compression ratio. These re-
sults validate that each component contributes complemen-
tary gains, with joint compression and masking offering the
best balance of quality and compactness.
Evaluation of Masking Strategies Under Confidence
Thresholds. We evaluate the effect of varying the mask-
ing threshold τon reconstruction quality and compression
(Table 3). As τincreases, DepthSplat exhibits gradual im-
provement in PSNR and LPIPS. However, this gain does not
stem from better modeling but rather from aggressive prun-
ing of spatially inconsistent or floating points, which often
produce visible artifacts. These improvements are thus par-
tially due to the suppression of such artifacts rather than

---

## Page 8

… … … …
 … … … …
Ref. rgb Ref.mask GT Ours -c Depthsplat Depthsplat -c OursFigure 4. Novel view synthesis on 50 context views. Ref. rgb and Ref. mask are the input image and compression mask. GT is the ground
truth. Ours-c and Depthsplat-c delete the mask regions. Ours and Depthsplat are full-scene results. Our method better removes floaters and
preserves fine details, producing more accurate and consistent renderings.)
Table 3. Performance under varying masking thresholds τ.
τ Method PSNR SSIM LPIPS ↓c-ratio. ↑
No MaskDepthSplat 21.39 0.7341 0.2212 0.0%
Ours 23.71 0.8159 0.1683 0.0%
0.1DepthSplat 21.82 0.7497 0.2081 23.32%
Ours 23.66 0.8141 0.1680 23.32%
0.3DepthSplat 22.17 0.7621 0.1996 38.11%
Ours 23.61 0.8092 0.1716 38.11%
0.5DepthSplat 22.32 0.7668 0.1964 43.77%
Ours 23.54 0.8056 0.1742 43.77%
true reconstruction fidelity. Our method leverages a learned
confidence-based mask to explicitly identify and remove re-
dundant or noisy Gaussians. As a result, even at high prun-
ing ratios (e.g., τ= 0.5, 43.77% compression), our method
maintains high PSNR (23.54 dB) and low LPIPS (0.1742),
nearly matching the unpruned case (23.71 dB, 0.1683).
This demonstrates that our compression is both precise and
structure-aware, effectively reducing redundancy while pre-
serving essential geometry and appearance. These trends
are also visually reflected in Fig. 4, where our method re-
moves floaters and preserves detail, while DepthSplat tends
to oversimplify or introduce inconsistencies in compressedregions.
5. Conclusion
We present LongSplat, a real-time 3D Gaussian Splat-
ting framework tailored for long-sequence reconstruction.
To address scalability and redundancy issues in existing
feed-forward pipelines, LongSplat introduces an incremen-
tal update mechanism that compresses redundant Gaus-
sians and incrementally integrates current-view observa-
tions into a consistent global scene. Central to our design is
the Gaussian-Image Representation (GIR), which projects
3D Gaussians into structured 2D maps for efficient fusion,
identity-aware compression, and 2D-based supervision. By
enabling lightweight per-frame updates and effective his-
torical modeling, LongSplat mitigates memory overhead
and quality degradation in dense-view settings. Extensive
experiments show that it achieves real-time rendering, im-
proves visual quality, and reduces Gaussian redundancy by
over 44%, offering a scalable solution for high-quality on-
line 3D reconstruction.
In future work, we aim to incorporate stronger, pose-
free extractors such as Cust3r[39], VGGT[37], and
DUST3R[40], removing the need for pre-computed cam-
era poses. We also plan to extend LongSplat with semantic
reasoning for 3D understanding in embodied scenarios.

---

## Page 9

References
[1] Our Proposed AdaptiveGaussian. Adaptivegaussian: Gener-
alizable 3d gaus-sian reconstruction from arbitrary views.
[2] Yanqi Bao, Jing Liao, Jing Huo, and Yang Gao. Distractor-
free generalizable 3d gaussian splatting. arXiv preprint
arXiv:2411.17605 , 2024.
[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In Proceedings of the IEEE/CVF inter-
national conference on computer vision , pages 5855–5864,
2021.
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased
grid-based neural radiance fields. In Proceedings of the
IEEE/CVF International Conference on Computer Vision ,
pages 19697–19705, 2023.
[5] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 130–141, 2023.
[6] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition , pages 19457–19467, 2024.
[7] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang,
Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast general-
izable radiance field reconstruction from multi-view stereo.
InProceedings of the IEEE/CVF international conference on
computer vision , pages 14124–14133, 2021.
[8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision , pages 370–386. Springer, 2024.
[9] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang,
Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. Mvsplat360:
Feed-forward 360 scene synthesis from sparse views. arXiv
preprint arXiv:2411.04924 , 2024.
[10] Zheng Chen, Chenming Wu, Zhelun Shen, Chen Zhao, We-
icai Ye, Haocheng Feng, Errui Ding, and Song-Hai Zhang.
Splatter-360: Generalizable 360 gaussian splatting for wide-
baseline panoramic images. In Proceedings of the Computer
Vision and Pattern Recognition Conference , pages 21590–
21599, 2025.
[11] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded
3d gaussian compression with 15x reduction and 200+
fps. Advances in neural information processing systems ,
37:140138–140158, 2024.
[12] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 12479–12488, 2023.[13] Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo
Hu, Chaopeng Zhang, Yao Yao, Ying Shan, and Long Quan.
Mani-gs: Gaussian splatting manipulation with triangular
mesh. In Proceedings of the Computer Vision and Pattern
Recognition Conference , pages 21392–21402, 2025.
[14] Xiangjun Gao, Jiaolong Yang, Jongyoo Kim, Sida Peng,
Zicheng Liu, and Xin Tong. Mps-nerf: Generalizable 3d hu-
man rendering from multiview images. IEEE Transactions
on Pattern Analysis and Machine Intelligence , 2022.
[15] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp
slam. In European Conference on Computer Vision , pages
180–197. Springer, 2024.
[16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers , pages 1–11, 2024.
[17] Yiming Ji, Yang Liu, Guanghu Xie, Boyu Ma, Zongwu Xie,
and Hong Liu. Neds-slam: A neural explicit dense semantic
slam framework using 3d gaussian splatting. IEEE Robotics
and Automation Letters , 2024.
[18] Gyeongjin Kang, Jisang Yoo, Jihyeon Park, Seungtae Nam,
Hyeonsoo Im, Sangheon Shin, Sangpil Kim, and Eunbyung
Park. Selfsplat: Pose-free and 3d prior-free generalizable 3d
gaussian splatting. In Proceedings of the Computer Vision
and Pattern Recognition Conference , pages 22012–22022,
2025.
[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph. , 42(4):139–1,
2023.
[20] Yang Li, Jinglu Wang, Lei Chu, Xiao Li, Shiu-hong Kao,
Ying-Cong Chen, and Yan Lu. Streamgs: Online general-
izable gaussian splatting reconstruction for unposed image
streams. arXiv preprint arXiv:2503.06235 , 2025.
[21] Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai,
Hujun Bao, and Xiaowei Zhou. Efficient neural radiance
fields for interactive free-viewpoint video. In SIGGRAPH
Asia 2022 Conference Papers , pages 1–9, 2022.
[22] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al. Dl3dv-10k: A large-scale scene dataset for deep
learning-based 3d vision. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 22160–22169, 2024.
[23] Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng
Wang, Christian Theobalt, Xiaowei Zhou, and Wenping
Wang. Neural rays for occlusion-aware image-based render-
ing. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition , pages 7824–7833,
2022.
[24] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 20654–20664, 2024.
[25] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings

---

## Page 10

of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 18039–18048, 2024.
[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM , 65(1):99–106, 2021.
[27] Seungtae Nam, Xiangyu Sun, Gyeongjin Kang, Younggeun
Lee, Seungjun Oh, and Eunbyung Park. Generative densifi-
cation: Learning to densify gaussians for high-fidelity gener-
alizable 3d reconstruction. In Proceedings of the Computer
Vision and Pattern Recognition Conference , pages 26683–
26693, 2025.
[28] Gyuhyeon Pak and Euntai Kim. Vigs slam: Imu-based
large-scale 3d gaussian splatting slam. arXiv preprint
arXiv:2501.13402 , 2025.
[29] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc
Pollefeys, and Andreas Geiger. Convolutional occupancy
networks. In Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceed-
ings, Part III 16 , pages 523–540. Springer, 2020.
[30] Wonseok Roh, Hwanhee Jung, Jong Wook Kim, Seungg-
wan Lee, Innfarn Yoo, Andreas Lugmayr, Seunggeun Chi,
Karthik Ramani, and Sangpil Kim. Catsplat: Context-aware
transformer with spatial guidance for generalizable 3d gaus-
sian splatting from a single-view image. arXiv preprint
arXiv:2412.12906 , 2024.
[31] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerf-
slam: Real-time dense monocular slam with neural radiance
fields. In 2023 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS) , pages 3437–3444. IEEE,
2023.
[32] Mohamed Sayed, John Gibson, Jamie Watson, Victor
Prisacariu, Michael Firman, and Cl ´ement Godard. Simplere-
con: 3d reconstruction without 3d convolutions. In European
Conference on Computer Vision , pages 1–19. Springer, 2022.
[33] Yu Sheng, Jiajun Deng, Xinran Zhang, Yu Zhang, Bei Hua,
Yanyong Zhang, and Jianmin Ji. Spatialsplat: Efficient
semantic 3d from sparse unposed images. arXiv preprint
arXiv:2505.23044 , 2025.
[34] Jiaming Sun, Yiming Xie, Linghao Chen, Xiaowei Zhou,
and Hujun Bao. Neuralrecon: Real-time coherent 3d re-
construction from monocular video. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 15598–15607, 2021.
[35] Lisong C Sun, Neel P Bhatt, Jonathan C Liu, Zhiwen Fan,
Zhangyang Wang, Todd E Humphreys, and Ufuk Topcu.
Mm3dgs slam: Multi-modal 3d gaussian splatting for slam
using vision, depth, and inertial measurements. In 2024
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS) , pages 10159–10166. IEEE, 2024.
[36] Shengji Tang, Weicai Ye, Peng Ye, Weihao Lin, Yang Zhou,
Tao Chen, and Wanli Ouyang. Hisplat: Hierarchical 3d gaus-
sian splatting for generalizable sparse-view reconstruction.
arXiv preprint arXiv:2410.06245 , 2024.
[37] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of theComputer Vision and Pattern Recognition Conference , pages
5294–5306, 2025.
[38] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P
Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo
Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibr-
net: Learning multi-view image-based rendering. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition , pages 4690–4699, 2021.
[39] Qianqian Wang, Yifei Zhang, Aleksander Holynski,
Alexei A Efros, and Angjoo Kanazawa. Continuous 3d
perception model with persistent state. arXiv preprint
arXiv:2501.12387 , 2025.
[40] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 20697–
20709, 2024.
[41] Weijie Wang, Donny Y Chen, Zeyu Zhang, Duochao Shi,
Akide Liu, and Bohan Zhuang. Zpressor: Bottleneck-aware
compression for scalable feed-forward 3dgs. arXiv preprint
arXiv:2505.23734 , 2025.
[42] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee
Lee. Freesplat: Generalizable 3d gaussian splatting towards
free view synthesis of indoor scenes. Advances in Neural
Information Processing Systems , 37:107326–107349, 2024.
[43] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee
Lee. Freesplat++: Generalizable 3d gaussian splatting
for efficient indoor scene reconstruction. arXiv preprint
arXiv:2503.22986 , 2025.
[44] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele,
and Jan Eric Lenssen. latentsplat: Autoencoding varia-
tional gaussians for fast generalizable 3d reconstruction. In
European Conference on Computer Vision , pages 456–473.
Springer, 2024.
[45] Zhe Xin, Chenyang Wu, Penghui Huang, Yanyong Zhang,
Yinian Mao, and Guoquan Huang. Large-scale gaussian
splatting slam. arXiv preprint arXiv:2505.09915 , 2025.
[46] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth. arXiv
preprint arXiv:2410.13862 , 2024.
[47] Jinbo Yan, Rui Peng, Zhiyan Wang, Luyang Tang, Jiayu
Yang, Jie Liang, Jiahao Wu, and Ronggang Wang. Instant
gaussian stream: Fast and generalizable streaming of dy-
namic scene reconstruction via gaussian splatting. In Pro-
ceedings of the Computer Vision and Pattern Recognition
Conference , pages 16520–16531, 2025.
[48] Kaiyun Yang, Yunqi Cheng, Zonghai Chen, and Jikai Wang.
Slam meets nerf: A survey of implicit slam methods. World
Electric Vehicle Journal , 15(3):85, 2024.
[49] Jianglong Ye, Naiyan Wang, and Xiaolong Wang. Featuren-
erf: Learning generalizable nerfs by distilling foundation
models. In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision , pages 8962–8973, 2023.
[50] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelnerf: Neural radiance fields from one or few images. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition , pages 4578–4587, 2021.

---

## Page 11

[51] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition , pages 19447–19456,
2024.
[52] Chuanrui Zhang, Yingshuang Zou, Zhuoling Li, Minmin Yi,
and Haoqian Wang. Transplat: Generalizable 3d gaussian
splatting from sparse multi-view images with transformers.
InProceedings of the AAAI Conference on Artificial Intelli-
gence , volume 39, pages 9869–9877, 2025.
[53] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu,
Shengping Zhang, Liqiang Nie, and Yebin Liu. Gps-
gaussian: Generalizable pixel-wise 3d gaussian splatting for
real-time human novel view synthesis. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 19680–19690, 2024.
[54] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R Oswald, Andreas Geiger, and Marc Pollefeys.
Nicer-slam: Neural implicit scene encoding for rgb slam. In
2024 International Conference on 3D Vision (3DV) , pages
42–52. IEEE, 2024.
[55] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
InProceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition , pages 12786–12796, 2022.
[56] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yi-
cong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-
sequence large reconstruction model for wide-coverage
gaussian splats. arXiv preprint arXiv:2410.12781 , 2024.