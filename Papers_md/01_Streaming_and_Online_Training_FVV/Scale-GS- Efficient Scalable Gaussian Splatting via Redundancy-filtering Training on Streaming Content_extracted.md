

---

## Page 1

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 1
Scale-GS: Efficient Scalable Gaussian Splatting via
Redundancy-filtering Training on Streaming Content
Jiayu Yang, Weijian Su, Songqian Zhang, Yuqi Han, Jinli Suo, Qiang Zhang, Senior Member, IEEE
Abstract —3D Gaussian Splatting (3DGS) enables high-fidelity
real-time rendering, a key requirement for immersive applica-
tions. However, the extension of 3DGS to dynamic scenes remains
limitations on the substantial data volume of dense Gaussians and
the prolonged training time required for each frame. This paper
presents Scale-GS, a scalable Gaussian Splatting framework
designed for efficient training in streaming tasks. Specifically,
Gaussian spheres are hierarchically organized by scale within an
anchor-based structure. Coarser-level Gaussians represent the
low-resolution structure of the scene, while finer-level Gaussians,
responsible for detailed high-fidelity rendering, are selectively
activated by the coarser-level Gaussians. To further reduce
computational overhead, we introduce a hybrid deformation and
spawning strategy that models motion of inter-frame through
Gaussian deformation and triggers Gaussian spawning to charac-
terize wide-range motion. Additionally, a bidirectional adaptive
masking mechanism enhances training efficiency by removing
static regions and prioritizing informative viewpoints. Extensive
experiments demonstrate that Scale-GS achieves superior visual
quality while significantly reducing training time compared to
state-of-the-art methods.
Keywords: Streaming Gaussian Splatting, multi-scale repre-
sentation, dynamic scene rendering, novel view synthesis.
I. I NTRODUCTION
The rapid advancement of 3D Gaussian Splatting
(3DGS) [1] has significantly reshaped the domain of real-
time 3D rendering. In particular, the introduction of Gaussian
training methods designed for dynamic scenes [2]–[9] has
greatly enhanced the feasibility of 3D streaming applications,
including virtual reality (VR), augmented reality (AR), and
immersive telepresence systems. By explicitly representing
scenes with differentiable Gaussians, these methods facil-
itate real-time rendering—capabilities that are critical for
interactive applications demanding low-latency visual feed-
back. However, as computational demands scale sharply with
scene and temporal complexity, the training time for dynamic
scenes—ranging from tens of minutes to hours—conflicts with
the low-latency requirements of real-time streaming.
The primary reason for the slow training time in Gaus-
sian Splatting stems from redundant computations involving
Gaussian spheres. The standard 3D Gaussian splatting methods
process each frame independently, thereby incurring repetitive
calculations on predominantly static Gaussians due to the
limited extent of dynamic regions. Although recently some re-
search partitions the scene into static and dynamic components
Jiayu Yang, Weijian Su, Songqian Zhang, Yuqi Han, and Qiang Zhang are
with the School of Computer Science and Technology, Dalian University of
Technology, Dalian 116024, China, and are also with Key Laboratory of Social
Computing and Cognitive Intelligence (Dalian University of Technology),
Ministry of Education, Dalian, 116024 China. Jinli Suo is with the Department
of Automation, Tsinghua University, Beijing 100084, China.
(Corresponding authors: Yuqi Han; yqhanscst@dlut.edu.cn. )to focus computational resources on dynamic Gaussians, the
overall volume remains substantial. Consequently, significant
overlapping computations occur across spatially and tempo-
rally adjacent regions, leading to inefficient resource usage
and bottlenecks that hinder real-time performance.
We observe that the size and number of Gaussian spheres
to represent the 3D scene vary significantly depending on
scene complexity. For example, textureless planar regions can
be effectively modeled by a small number of large Gaussian
spheres, whereas highly textured areas necessitate dozens or
even hundreds of smaller Gaussians. The larger Gaussian
spheres often contribute more significantly to scene represen-
tation and therefore warrant higher training priority. Inspired
from scalable video coding, this work proposes Scale-GS,
a scalable Gaussian splatting framework aimed at mitigating
redundancy in 3D spatial representation, thereby enabling ac-
celerated training. Specifically, Gaussian spheres are organized
by scale, with each level independently trained under the
viewpoints at corresponding resolutions. For each frame, large
scale Gaussians are first optimized using low-resolution views
to approximate the scene. Upon convergence, a triggering
criterion evaluates whether small scale Gaussians should be
activated for refinement, thereby reducing the redundant Gaus-
sian spheres optimization. As shown in Fig. 1, the proposed
method achieves high rendering quality with short training
time compared to the SOTA algorithm. The framework con-
ducts a coarse-to-fine principle where the training at each level
of scale is triggered by the preceding level of scale, thereby
filtering redundant training of irrelevant Gaussians.
To enable efficient training for streaming content, we intro-
duce a hybrid strategy that combines deformation and spawn-
ing to infer dynamic changes in the current frame based on the
preceding frame. Generally, the Gaussian deformation [6], [9]
models the motion of Gaussian sphere, but insufficient for cap-
turing newly appearing objects or wide-range motion. In con-
trast, Gaussian spawning [10], [11] introduces new Gaussians
to fine-tune dynamic regions but requires considerably longer
training time. To balance the trade-off, the hybrid strategy
first applies deformation to model inter-frame motion and then
determines, based on the training outcome, whether spawning
should be triggered for finer-grained refinement. Specifically,
under the Scale-GS framework, if the deformation of a deter-
mined scale fails to adequately represent the dynamic, either
the emergence of new content or the motion of smaller-scale
Gaussians happens. Thus, the deformation result triggers new
Gaussians spawning at the current scale and deformation at
the next scale. This sequential activation ensures Gaussians
are progressively introduced at locations with actual dynamics
along with increasingly finer scales, enabling efficient andarXiv:2508.21444v1  [cs.CV]  29 Aug 2025

---

## Page 2

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 2
PSNR:33.83dB(IGS:33.15dB)
Training:3.14s(IGS:3.46s)
FPS:274(IGS:204)
Res:1600 ×1200PSNR:31.14dB(IGS:30.61dB)
Training:3.01s(IGS:3.22s)
FPS:278(IGS:251)
Res:1280 ×720
2 3 7 9 20 40 5031.032.032.132.233.033.634.3
Per-frame training time(s)PSNR(dB)IGS
(CVPR2025)3DGStream
(CVPR2024)
4DGS
(CVPR2024)HiCom
(NIPS2024)
Deformable
(CVPR2024)Scale -GS
(Ours )
Kplanes
(CVPR2023)
StreamRF
(NIPS2022)
{Trimming } {Coffee martini }
Fig. 1. The proposed Scale-GS under dynamic scene achieves best rendering quality with the shortest training time. The left figures show results of
our Scale-GS on N3DV Coffee martini and MeetRoom Trimming datasets, where “Res” indicates video resolution. The right figure is tested on the N3DV
dataset, where the radius of the circle corresponds to the average storage per frame and the method in the top left corner demonstrates the best performance.
high-fidelity temporal GS representation.
To enhance training efficiency, we propose a bidirectional
adaptive masking mechanism that simultaneously suppresses
static regions and selects informative training viewpoints.
The forward masking component detects dynamic and static
anchors via inter-frame change analysis, where pixel-wise
differences between consecutive frames are back-projected
to estimate motion patterns. For backward camera viewpoint
selection, we define a relevance score between projected
dynamic anchors and camera fields of view, further weighted
by directional factors that prioritize orthogonal or novel view-
points. The top-ranked views, as determined by the relevance
score, are selected to form the active viewpoint set. This
bidirectional masking mechanism reduces computational re-
dundancy caused by uninformative viewpoints and facilitates
accurate reconstruction of dynamic scenes.
Comprehensive experiments conducted on three challenging
real-world datasets—NV3D, MeetRoom, and Google Immer-
sive—demonstrate the superior performance of the proposed
framework across multiple evaluation metrics. Qualitative
comparisons show that our method reconstructs significantly
sharper fine-grained details, particularly in complex scenar-
ios involving human interactions, dynamic phenomena such
as flames, and intricate textures. Furthermore, experimental
results demonstrate that Scale-GS not only improves visual
quality but also outperforms current state-of-the-art methods
in both training and rendering time. These findings validate the
effectiveness of Scale-GS, which prioritizes more important
Gaussian spheres and improves the average training efficiency.
The main contributions of this work are as follows:
1) We propose Scale-GS, a scalable GS framework per-
forming redundancy-filtering training on streaming con-
tent to improve the efficiency. The Scale-GS achieves
the most efficient training compared to the existing
Gaussian training methods on streaming content.
2) The Scale-GS integrates a hybrid deformation-spawning
Gaussian training strategy that prioritizes large-scale
Gaussians and selectively activating finer ones to reduce
redundancy in 3D scene representation while preserving
high-fidelity dynamic representations.
3) Extensive evaluations show that Scale-GS achieves su-
perior efficiency–quality trade-offs for streaming novel
view synthesis, improving visual quality, reducing train-ing time, and supporting real-time rendering.
In the following, we first introduce the research related to
the Scale-GS, including the novel view synthesis for static
scene and videography at Sec. II. Later we thoroughly present
the detail of the method at Sec. III. The qualitative and quanti-
tative experimental results are exhibits at Sec. IV. Finally, we
draw the conclusion and propose the future work at Sec. V.
II. R ELATED WORK
In this section, we separately review research on implicit
and explicit novel view synthesis methods for both static and
dynamic scenes. These studies primarily focus on improving
visual quality and enhancing training efficiency.
A. Novel View Synthesis for Static Scenes
Early novel view synthesis methods predominantly rely
on geometric interpolation, with approaches such as the Lu-
migraph [12] and Light Field rendering [13], [14], laying
the groundwork through advanced interpolation techniques
applied to densely sampled input images.
Neural Radiance Fields (NeRF) [15] introduces a break-
through in photorealistic view synthesis by modeling scene
radiance through implicit neural representations using multi-
layer perceptrons. This innovation has spurred extensive re-
search aiming at overcoming NeRF’s inherent limitations
across various dimensions. Key efforts include accelerating
training procedures [16]–[19], achieving real-time rendering
performance [20]–[22], improving synthesis fidelity in com-
plex scenes [21], [23], [24], and enhancing robustness un-
der sparse input conditions [25]–[27]. However, the com-
putational overhead inherent in NeRF’s volume rendering
paradigm—which requires numerous neural network compu-
tation per frame—presents significant challenges in balancing
training efficiency, rendering speed, and visual fidelity.
To address these limitations, Kerbl et al. [1] proposes 3DGS,
which leverages explicit 3D Gaussian primitives combined
with differentiable rasterization-based rendering to enable
real-time, high-quality view synthesis. The 3DGS inspires
a broad range of research efforts exploring various aspects
of Gaussian-based scene representations. Some studies fo-
cus on enhancing rendering fidelity [28]–[30], while others
aim to improve geometric precision and accuracy [31], [32].

---

## Page 3

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 3
Furthermore, considerable efforts are devoted to developing
compression techniques to mitigate storage overhead [28],
[33]–[36]. Recent studies investigate the joint optimization of
camera parameters alongside Gaussian field estimation [37],
as well as the extension of Gaussian splatting to broader 3D
content generation tasks [38]–[40]. Despite the 3DGS in static
scenes achieves high-quality rendering, the development of an
on-demand training framework for 3DGS remains an open
challenge.
Generalization is introduced as an effective strategy to
enhance inference speed. Recent developments in NeRF
methodologies [27], [41], [42] and 3DGS [43]–[45] focus on
generalizable reconstruction networks trained on large-scale
datasets. Specifically, PixelSplat [46] uses Transformers to
encode features and decode Gaussian primitive parameters.
DepthSplat [47] leverages monocular depth to recover 3D de-
tails from sparse inputs. Other frameworks [48]–[50] combine
Transformer or Multi-View Stereo (MVS) [51] methods to
build geometric cost volumes, enabling real-time generaliza-
tion. However, due to the insufficient diversity of available
3D datasets, the generalization performance of these methods
remains to be further improved.
B. Novel View Synthesis for Dynamic Scenes
Novel view synthesis for dynamic scenes naturally extends
static models, with early approaches building on NeRF [11],
[52]–[57] and 3DGS [1], leveraging their efficient rendering
capabilities for dynamic scene reconstruction. While Gaussian-
based methods [2]–[9] learn temporal attributes to model
dynamic scenes as unified representations and improve recon-
struction quality, their requirement to load all data simultane-
ously results in high memory usage, limiting their feasibility
for long-sequence streaming.
To address these challenges, streaming-based methods, such
as ReRF [58], NeRFPlayer [59], and StreamRF [60] refor-
mulate dynamic scene reconstruction as an online problem.
Moreover, 3DGStream [10] utilizes Gaussian-based represen-
tations combined with Neural Transformation Caches to model
inter-frame motion, though it still requires over 10 seconds
per frame. HiCoM [61] introduces an online reconstruction
pipeline for multi-view video streams employing perturbation-
based smoothing for robust initialization and hierarchical
motion coherence mechanisms. Instant Gaussian Stream (IGS)
[62] proposes enables single-pass motion computation guided
by keyframes, reducing error accumulation and achieving
reconstruction times around 4 seconds per frame. Alternative
frame-tracking methods [63], [64] track Gaussian evolution
across frames, supporting streaming protocols but incurring
substantial per-frame data overhead.
In contrast to existing methods that require full sequence
processing or incur significant per-frame optimization over-
head, we propose Scale-GS, a scalable Gaussian splatting
framework for efficient streaming rendering. By constructing
multi-scale Gaussian representations combined with selective
training, Scale-GS enables on-the-fly novel view synthesis
while preserving high rendering quality.III. M ETHOD
In this section, we first introduce the preliminaries in
Sec. III-A. Later we present the key pipeline of Scale-GS.
The framework of Scale-GS is presented as in Fig. 2.
After decompositing the dynamic part, Scale-GS follows an
anchor-based multi-scale Gaussian representation (Sec. III-B)
as its core framework. We apply hybrid deformation-spawning
Gaussian optimization (Sec. III-C) to model inter-frame mo-
tion. When deformation is insufficient to capture dynamics, we
activate the next scale and selectively spawn new Gaussians.
Once all scales have converged, redundant Gaussians are
pruned to optimize the representation (Sec. III-D). In addi-
tion, Scale-GS employs bidirectional bidirectional adaptive
masking (Sec. III-E) to identify dynamic anchors and select
informative viewpoints.
A. Preliminaries
3DGS uses a dense set of Gaussian spheres to represent
the whole space, and renders viewpoints via differentiable
splatting combined with tile-based rasterization of these Gaus-
sian components. For each Gaussian sphere i, the expectation
position µiand variance Σidetermine the formation of the
Gaussian Gi(x). The point xon the Gaussian sphere Gi(x)
is noted as
Gi(x) = exp
−1
2(x−µi)⊤Σ−1
i(x−µi)
, (1)
where Σiis composed of a rotation matrix Ri∈R3×3
and a diagonal scale matrix Si∈R3×3, denoted as Σi=
RiSiST
iRT
i. The color and opacity of Gaussian sphere iare
denoted as ciandαi.
We use x′to define the 2D projection pixel position, and
the pixel value C∈R3is rendered via α-composite blending.
Specifically, we assume the light ray projected onto a pixel x′
intersects with NGaussian surfaces along its path. The color
C(x′)is defined as
C(x′) =X
i∈Nciαii−1Y
j=1(1−αj), (2)
where NGaussians are sorted from near to far and αi
signifies the opacity of Gaussian Gi(x). With the differentiable
rasterizer, all attributes of the 3D Gaussians become learnable
and can be directly optimized in an end-to-end manner through
training view reconstruction.
To enable structural rendering in the Gaussian splatting
model, we introduce an anchor-based mechanism [28], [33].
The 3D space is uniformly partitioned into multiple voxels,
with each voxel assigned a dedicated anchor responsible for
managing all Gaussian primitives within its region.
The anchor-based framework is initialized by voxelizing
the sparse point cloud generated from Structure-from-Motion
(SfM) pipelines. Let vdenote the index of a specific anchor,
andVrepresent the complete set of anchors across the entire
space. Each v∈Vcorresponds to a local context feature ˆfv,
a 3D scaling factor cv, and klearnable offsets Ov∈Rk×3
to identify the attribute of the anchor vandkcorresponding
Gaussians (indexed from 0tok−1). Specifically, given the

---

## Page 4

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 4
Deformed Gaussian(c) Gaussian deformation  (GD) (d) Octree Gaussian spawning (OGS )
Original 
Gaussian
Colored for 
new GaussianMLP
MLPℎ(𝜇𝜇)
𝑐𝑐
𝛼𝛼∆𝜇𝜇
∆𝑞𝑞
∆𝑐𝑐
∆𝛼𝛼Dynamic anchorStatic anchorAnchor -based Gaussian
Spawned 
Gaussian
Same as previous frameRedundant removing(b) Anchor -based multi- scale Gaussian representationGDLevel 1OGSLevel   2 Level  L……OGS GD
GD OGS
Fusion
𝜵𝜵𝒈𝒈(𝒍𝒍)>𝝉𝝉𝒂𝒂𝒂𝒂𝒂𝒂(𝒍𝒍)
Dynamic decomposition
t t-1 t+1
Level 1
Level 2
Level 3
…
(a) Multi- scale decomposition
Fig. 2. The framework of Scale-GS. (a) The multi-scale decomposition across different resolution levels, where finer scales capture increasingly detailed
scene dynamics. (b) Anchor-based multi-scale Gaussian representation. After completing training at each level, all scales are combined, followed by redundant
Gaussians removing. (c-d) The hybrid deformation and spawning Gaussian optimization. (c) Gaussian deformation module that models temporal changes
through anchor-guided MLPs, (b) Octree Gaussian Spawning that adaptively adds new Gaussians based on Octree subdivision.
location xvof anchor v, the positions of Gaussians winthin
the anchor vare derived as
{µ0, . . . , µ k−1}=xv+{O0, . . . , O k−1} ·cv. (3)
Since the Gaussians associated with the same anchor share
similar attributes, the other factors of Gaussians could be pre-
dicted by lightweight MLPs Fα, Fc, Fq, Fstaking the anchor
attributes as input. Specifically, we denote the relative viewing
distance from anchor vto the camera as δvand the viewing
direction from anchor vto the camera as ⃗dv. Taking opacity
prediction as an example, the opacity values of all kGaussians
within anchor vare derived as
{α0, . . . , α k−1}=Fα(ˆfv, δv,⃗dv). (4)
The color, rotation, and scale predictions follow similar
formulations using their respective MLPs Fc,Fq, and Fs.
By partitioning the whole space into voxels and assigning
an anchor to each voxel, the anchor-based approach facili-
tates localized organization and efficient indexing of Gaussian
distributions, significantly reducing the overhead of traversing
and computing irrelevant Gaussians. Moreover, the neural
Gaussians, which infer all Gaussian factors from the anchor
attribute improves the efficiency of training.
B. Anchor-based Multi-Scale Gaussian Representation
Variations in texture detail of the 3D scene lead to Gaussian
representations of differing granularity. To improve computa-
tional efficiency, we introduce a multi-scale Gaussian opti-
mization framework, drawing inspiration from scalable video
encoding [65], [66]. In this framework, coarse-scale Gaussians
are first optimized using low-resolution inputs, followed by the
refinement of fine-scale Gaussians guided by high-resolution
images. We separate static and dynamic regions and observethe dynamic parts. When inter-frame variations arise, opti-
mization starts at the coarse scale and progressively refines
finer scales, ensuring global structures and local details, as
shown in Fig. 2(a).
We define a multi-scale structure with Llevels of scale and
lcorresponds to a specific level. The Mtraining viewpoint
at time twith original resolution as I0,t, ..., I M−1,t. The
viewpoint resolution at the level lis denoted as Il
0,t, ..., Il
M−1,t.
At each level l, the corresponding set of Gaussians G(l)is
supervised by viewpoints Il
0,t, ..., Il
M−1,t. Considering that
variations in Gaussian representations between temporally
adjacent frames of the same scene are limited, the distribution
learned from the initial frame can reasonably approximate that
of all frames. Thus, we initialize the 3DGS on the first frame to
estimate the scale of each Gaussian. We define the maximum
scale s(0)
max, minimum s(0)
min, and mean s(0)
mean at each level l.
Given that Gaussian spheres with larger scales encode less
fine-grained detail, we employ a binary partitioning strat-
egy to divide the scale space. Specifically, once the scale
range for a given level is established, the subsequent level
is recursively defined within the finer half of the current
level’s range. Specifically, we assume the size of level lis
defined as [s(l)
min, s(l)
max]. If the level l+ 1 is required, the
scale of level lis revised to [s(l)
mean, s(l)
max]and the the scale
of level l+ 1is revised to [s(l)
min, s(l)
mean], i.e, [s(l+1)
min, s(l+1)
max]←
[s(l)
min, s(l)
mean],[s(l)
min, s(l)
max]←[s(l)
mean, s(l)
max]. This ensures that
higher-indexed levels (higher resolution) consistently receive
smaller Gaussian scale ranges.
We introduce a clamp function to ensure that Gaussians at
each level are constrained within their designated scale ranges,
which is represented as clamp (,), to enforce both upper and
lower bounds on the scale of the Gaussians, i.e.,
si=clamp (s(l)
min, s(l)
max, si),ifsi∈l. (5)

---

## Page 5

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 5
MLPTime = t
Level l
MLPTime = t
Level l+1
Freeze next levelMulti scale fusion
Operation 
flow
Gradient 
flow
𝜵𝜵𝒈𝒈𝒗𝒗(𝒍𝒍)>𝝉𝝉𝐚𝐚𝐚𝐚𝐚𝐚(𝒍𝒍):
𝜵𝜵𝒈𝒈𝒗𝒗(𝒍𝒍)>𝝉𝝉𝐚𝐚𝐚𝐚𝐚𝐚(𝒍𝒍)
𝜵𝜵𝒈𝒈𝒗𝒗(𝒍𝒍)<𝝉𝝉𝐚𝐚𝐚𝐚𝐚𝐚(𝒍𝒍):Freeze next level
Fig. 3. The detail of hybrid deformation-spawning strategy across multi-
scale levels. At level l, Gaussians undergo temporal deformation via MLPs.
When the average gradient exceeds the threshold , the next level l+ 1 is
activated for finer-grained optimization. Meanwhile, within each subspace at
level l, when the mean gradient exceeds the threshold, new Gaussians are
spawned within that subspace. Conversely, when the average gradient is less
than the threshold, the hierarchical progression stops and multi-scale fusion
is performed to integrate representations across all active levels.
According to Eq. (5), scale parameters at each level are con-
strained within their designated ranges, ensuring that coarser
and finer Gaussians are optimized independently.
C. Hybrid Deformation-spawning Gaussian Optimization
As the scene changes over time, existing methods rely on
either deformation, which lacks expressiveness for complex
dynamics, or spawning, which is computationally inefficient
due to the need for many new Gaussians. To overcome the
limitations, we propose a hybrid approach that combines de-
formation and spawning policy. After performing deformation
at scale level l, unresolved dynamics—either from higher-
resolution variations or newly emerged components—are ad-
dressed by guided spawning at the current scale and activating
deformation at the l+ 1, as shown in Fig. 2(b).
We use the anchor vas an example to present the following
description. For scale level l, the deformation process is
processed through two lightweight Multi-Layer Perceptrons
(MLPs) to model the dynamics of each Gaussian from t−1
tot. We adopt a decoupled design to separately model geom-
etry and appearance. Geometric deformation leverages spatial
context via hash encoding to capture 3D structure, while pho-
tometric changes—reflecting intrinsic material properties—are
processed directly without spatial encoding. The framework of
Gaussian deformation is presented as Fig. 2(c). At each level
l, an geometric deformation MLP ( MLP g) takes the previous
frame’s center position µ(l)
i,t−1as input, which is encoded via
multi-scale hash encoding h(µ(l)
i,t−1)to capture both local and
global spatial context. The MLP gprocesses the hash-encoded
position to predict geometric changes
∆µ(l)
i,∆q(l)
i= MLP g(h(µ(l)
i,t−1)). (6)
The output is a 7-dimensional vector, where the first 3 di-
mensions represent the position offset ∆µ(l)
iand the last 4
dimensions denote the quaternion increment ∆q(l)
i.
The appearance deformation MLP ( MLP a) receives the
original color c(l)
i,t−1and opacity α(l)
i,t−1as inputs, and predicts
the updated values c(l)
i,tandα(l)
i,t, i.e.,
c(l)
i,t, α(l)
i,t= MLP a(c(l)
i,t−1, α(l)
i,t−1). (7)Overall, we model temporal change of all Gaussian at-
tributes at level lvia residual updates from the previous frame
θ(l,t)
i=θ(l,t−1)
i + ∆θ(l)
i, θ∈ {µ,Σ, α,c},
q(l)
i,t= norm( q(l)
i,t−1)·norm(∆ q(l)
i),(8)
where norm( ·)indicates quaternion normalization to ensure
unit quaternion constraints for valid rotations.
The rendering of scale level lfollows the volume render-
ing approach where each level processes its own Gaussians
independently
C(l)(x′) =X
i∈Nc(l)
i,tα(l)
i,ti−1Y
j=1(1−α(l)
j,t). (9)
The reconstruction loss at scale level lis computed with
respect to the corresponding resolution images, including an
ℓ1loss and a structural similarity loss to enforce perceptual
fidelity.
L(l)=L(l)
1(Il
n,t,ˆCl
n,t) +λSSIML(l)
SSIM(Il
n,t,ˆCl
n,t), (10)
where ˆCn,tindicates the volume rendering result of viewpoint
nat time twith Gaussian scale level l.
For each anchor vat level l, we compute the average
gradients of constituent Gaussians over dtraining iterations,
denoted as ∇g(l)
v, denoted as
∇g(l)
v=1
|Glv|X
i∈Glv∂L
∂θ(l)
i, (11)
where Gl
vrepresents the set of Gaussians belonging to anchor
vat scale level l.
We define a level-specific gradient thresholds [28], denoted
as
τ(l)
add=V ol
4l−1, (12)
where V ol indicates the volume size of each anchor and the
threshold decreases exponentially with scale level l, guid-
ing progressively finer control at higher resolution levels.
If∇g(l)
v> τ(l)
addafter deformation at l, suggesting that the
resolution of lis not sufficient to capture the underlying
dynamic changes, the Scale-GS framework triggers octree
Gaussian Spawning at land deformation at l+ 1. The detail
of hybrid deformation-spawning strategy across multi-scale is
represented as Fig. 3.
Octree Gaussian Spawning of Scale l.As shown in Fig. 2(d),
for the Spawning of level l, Gaussians are spawned at pre-
defined anchor locations, followed by an optimization phase
to adapt the newly added Gaussians to the dynamic scene.
After training, these Gaussians are associated with their cor-
responding anchors and carried forward for overfitting in the
subsequent frame.
We construct an octree-based Gaussian representation to
model dynamics with minimal Gaussian usage. The octree
structure enables adaptive allocation of Gaussian guided by
gradient information. We refer to each region partitioned by
the octree within an anchor space as a subspace. At each
level l, we compute the mean gradient ∇g(l)
vof all Gaussians
within each subspace. If ∇g(l)
v> τ(l)
add, a fixed number of

---

## Page 6

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 6
Gaussians are randomly assigned within the subspace, which
is then recursively subdivided. This process continues until
all subspaces fall below the threshold or the spatial resolution
reaches one-thousandth of the original domain.
Gaussian Deformation of Scale l+ 1.To enable finer-
grained deformation modeling, the optimization process ac-
tivates Gaussians at the next resolution level l+1. These
Gaussians are inherited from frame t−1and share the same
anchor structure as level l, but operate at a higher spatial
resolution through hierarchical refinement.
D. Redundant Gaussian Removing
As the hybrid deformation-spawning strategy introduces
additional Gaussians in each frame, a redundant Gaussian
removing step is applied after all scales converging to prevent
uncontrolled growth in the number of Gaussian over time.
For each Gaussian imanaged by anchor v, we define a 1-D
optimizable mask Mv,i. The Mv,iis passed through a sigmoid
function σ(·)to ensure differentiability. When Mv,iis large,
σ(Mv,i)→1; conversely, when Mv,iis small, σ(Mv,i)→0,
effectively removing the corresponding Gaussian.
For rendering pixels covered by anchor v, the projection
color after masking is defined as:
C(x′) =X
i∈Nσ(Mv,i)ci,tαi,ti−1Y
j=1(1−σ(Mv,i)αj,t).(13)
When the sigmoid value σ(Mv,i)approaches zero, the
corresponding Gaussian iunder anchor vis excluded from the
volume rendering process, regardless of the scale level from
which it originated. Thus, the loss function is then employed
to optimize the Gaussians, filtering out redundant ones. The
total loss combines the reconstruction error from the multi-
scale fused rendering with a sparsity regularization term, i.e.,
L=LX
l=1L(l)+λrX
v∈VNX
i=1σ(Mv,i), (14)
where L(l)represents the rendering loss on each scale, and
λrcontrols the redundancy removing regularization. The re-
dundancy removing term reduces redundant Gaussians by pe-
nalizing non-zero mask values, thereby promoting a compact
representation across all anchors and scale levels.
E. Bidirectional Adaptive Masking
We propose a bidirectional adaptive masking mechanism
that selects dynamic spatial anchors and informative camera
views for each training frame, serving as a preprocessing step
to reduce redundant computation in static regions and mitigate
supervision from views with limited marginal utility.
The forward masking process selects dynamic anchors
by distinguishing motion across adjacent multi-view frames.
Specifically, temporal variations between two consecutive
frames enable a coarse estimation of spatial motion for in-
dividual image pixels. The motion is then back-projected into
3D space to identify spatial position with significant displace-
ment. If an anchor’s coverage consistently exhibits prominent
motion patterns over multiple frames, it is designated as aAlgorithm 1 The optimization of Scale-GS
Require: Prior Gaussians Gt−1, frame It, scale levels L,
thresholds τ(l)
add.
Ensure: Updated Gt;
1:Initialize: Gt← G t−1;
2:Apply bidirectional adaptive masking (III-E) to identify
dynamic anchors Vdynusing Eq.(16);
3:foreach level l= 1 toLdo
4: Apply scale constraints using Eq.(5);
5: foreach dynamic anchor v∈Vdyndo
6: Apply deformation using Eq.(6-7): geometric and
appearance MLPs;
7: Compute reconstruction loss using Eq.(10);
8: Compute gradient ∇g(l)
v;
9: if∇g(l)
v> τ(l)
add(Eq.(12)) then
10: Spawn Gaussians at scale l;
11: Activate deformation at level l+ 1;
12: else
13: break;
14: end if
15: end for
16:end for
17:foreach anchor v∈Vdo
18: Apply redundant Gaussian removing using Eq.(13) for
masked rendering;
19:end for
20:Optimize total loss using Eq.(14);
21:return Updated Gt.
dynamic anchor, which subsequently guides the optimization
of dynamic objects.
The backward masking is designed to select a subset of cam-
era views aligned with these dynamic regions. We introduce
a relevance score for each camera view ck. Let IoU(ck, v)
denote the normalized intersection-over-union between the
image-space projection of anchor vand the field-of-view
of camera ck. Given a threshold τview, we define the view
relevance score as
S(ck) =X
v∈V1[IoU( ck, v)> τ view]·ω(ck, v), (15)
where 1()is the indicator function and ω(ck, v)is a direction
weight, defined as
ω(ck, v) =|n⊤
vdck|, (16)
where nvis the average normal vector of Gaussians in anchor
v,dvdenotes the normal direction of the viewpoint dck.
Given the ranking of S(ck), the top-ranked views are
selected for training, as they are considered the most rele-
vant to the dynamics. This bidirectional masking mechanism
reduces computational redundancy caused by uninformative
viewpoints of dynamic scenes.
F . Algorithmic Summarization
We summarize Scale-GS dynamic scene optimization ap-
proach in Algorithm 1, which performs multi-scale guided

---

## Page 7

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 7
dynamic updates through hybrid deformation-spawning opti-
mization (III-C) combined with bidirectional adaptive masking
(III-E).
The algorithm operates on the pre-initialized multi-
resolution Gaussian hierarchy, applying the hybrid
deformation-spawning optimization strategy across multiple
resolution levels while using bidirectional adaptive masking to
focus computation on dynamic regions and informative views.
This approach ensures efficient and targeted optimization for
dynamic scene reconstruction while maintaining scale-aware
structure throughout the process.
Implementation Details. All experiments are conducted in
a virtual environment using Python 3.9 and PyTorch 2.1.0
as the primary deep learning framework. Additional libraries
include plyfile 0.8.1 for data handling, torchaudio 0.12.1
for audio processing, and torchvision 0.13.1 for computer
vision tasks. Training and inference were performed on a
workstation equipped with an NVIDIA RTX 4090 GPU with
CUDA 12.6 support, complemented by cudatoolkit 11.8 for
GPU acceleration. The proposed method is implemented in
PyTorch and trained under the above configuration, ensuring
seamless integration of hardware capabilities and software
functionalities for reliable experimental outcomes.
IV. E XPERIMENT RESULTS
In this section, we thoroughly analyze the experiment results
of Scale-GS. Specifically, we first introduce the dataset and
the experimental setup. Later, we present the qualitative results
and quantitative results. Finally, we conduct the ablation study
to illustrate the effectiveness of each key module.
A. Datasets
We evaluate Scale-GS method on three real-world dynamic
scene datasets: the MeetRoom dataset [60], the NV3D (Neural
3D Video) dataset [67], and the Google Immersive Light Field
Video dataset [68]. All of the datasets exhibit complex motion
and occlusion patterns, thus suitable for evaluating the free-
viewpoint rendering.
NV3D Dataset [67]. The Neural 3D Video dataset contains
six dynamic scenes captured using a synchronized 21-camera
array arranged in a semi-circular configuration. Each camera
records the frames at a resolution of 2704×2028 with 300
frames, including various human motion interactions under
indoor lighting conditions.
MeetRoom Dataset [60]. The MeetRoom dataset provides
3 indoor dynamic scenes recorded with 13 synchronized
cameras, each capturing at 1280×720 resolution with 300
frames. The subjects are performing structured activities such
as sitting, walking, or conversing in an office-like setting.
Google Immersive Dataset [68]. This dataset contains 15
complex dynamic scenes captured using a high-fidelity immer-
sive light field video rig consisting of 46 time-synchronized
cameras on a 92 cm diameter hemisphere, each capturing at
2560×1920 resolution. The system supports a large viewing
baseline (up to 80 cm) and a wide field of view ( >220°),
posing very challenging novel view synthesis scenes.B. Experimental Setup
Hyper-parameter settings. In the qualitative experiment and
quantitative experiment, we adopt a three-level hierarchical
structure ( L= 3 ), corresponding to an image resolution
pyramid obtained by downsampling the original input to
{1/4,1/2,1}for levels l= 1,2,3, respectively. We set
λSSIM = 0.2,λr= 0.001 and the level-specific gradi-
ent thresholds τ(l)
add= 0.01/4l−1, which yields {τ(l)
add}=
{0.01,0.0025,0.000625 }for levels l= 1,2,3respectively.
Metrics . We assess the rendering fidelity of Scale-GS method
using three widely adopted perceptual and photometric met-
rics: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural
Similarity Index), and LPIPS (Learned Perceptual Image
Patch Similarity). These metrics respectively reflect pixel-
level accuracy, structural coherence, and perceptual similarity.
Furthermore, we compare the training time and rendering time
to specify the efficiency of the proposed Scale-GS.
Baselines. We evaluate Scale-GS against five state-of-the-
art dynamic scene rendering methods that represent different
approaches to temporal modeling and real-time performance.
Deformable 3D Gaussians [6] extends the static 3D Gaus-
sian splatting framework to handle dynamic scenes through a
canonical space representation. The method learns a set of 3D
Gaussians in canonical space and captures temporal variations
using a deformation field implemented as an MLP that predicts
position, rotation, and scaling offsets. To address potential
jitter from inaccurate camera poses, the authors introduce an
annealing smooth training mechanism, enabling both high-
fidelity rendering quality and real-time performance.
4DGS [8] takes a different approach by incorporating a
Gaussian deformation field network that operates on canonical
3D Gaussians. The method employs a spatial-temporal struc-
ture encoder coupled with a multi-head deformation decoder
to predict Gaussian transformations across time. By modeling
both Gaussian motion and shape changes through decomposed
neural voxel encoding, 4DGS achieves efficient real-time
rendering while maintaining temporal consistency.
3DGStream [10] focuses on streaming applications, en-
abling on-the-fly training for photo-realistic free-viewpoint
videos. The method introduces a Neural Transformation Cache
(NTC) to model 3D Gaussian transformations and employs
an adaptive Gaussian spawn strategy for handling newly ap-
pearing objects. The framework operates through a two-stage
pipeline: first training the NTC for existing Gaussians, then
spawning and optimizing additional frame-specific Gaussians
to accommodate emerging scene content.
HiCoM [61] presents a comprehensive framework specifi-
cally designed for streamable dynamic scene reconstruction.
It combines three key components: a perturbation smooth-
ing strategy for robust initial 3D Gaussian representation, a
hierarchical coherent motion mechanism that captures multi-
granular motions through parameter sharing within regional
hierarchies, and a continual refinement process that evolves
scene content while maintaining representation compactness.
IGS [62] offers a generalized streaming framework centered
around an Anchor-driven Gaussian Motion Network (AGM-
Net). This network projects multi-view 2D motion features

---

## Page 8

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 8
IGS4DGS
Scale- GS GT3DGStream HiCom
IGS4DGS
GT3DGStream HiCom
IGS4DGS
GT3DGStream HiCom
Frame Index = 2 Frame Index = 151 Frame Index = 183 Scale- GS
Scale- GS
Fig. 4. Qualitative comparison results on the NV3D datasets(scene flame-steak). The frame index is 2, 151, and 183 from up to down. For each frame index,
we present the result of 4DGS, 3DStream, HiCom, IGS, Scale-GS , and ground truth.
into 3D space using strategically placed anchor points to
drive Gaussian motion. The method further incorporates a key-
frame-guided streaming strategy that refines key frames and ef-
fectively mitigates error accumulation during long sequences.
C. Qualitative Results
We choose 2 scenes from NV3D and the Google Immersive
dataset to conduct a comprehensive qualitative evaluation,
including indoor scene and outdoor scene.
1) Comparison in Indoor Scene: Fig. 4 presents the visual
result on the NV3D dataset on 3 progressive frame index. Inthis scenario, it is critical to analyze the motion of the human
and the dog and accurately reconstruct the dynamics of the
flame. According to Fig. 4, Scale-GS reconstructs significantly
sharper and accurate fine-grained details compared to the
baselines.
Specifically, in the frame 2, the 4DGS and 3DGStream fail
to capture the trailing motion of hands, exhibiting noticeable
blur. Meanwhile, four baselines render blurred occluded back-
grounds, showing a large difference from the ground truth. The
proposed Scale-GS not only reconstructs the clear texture of
the blender, but renders the contour of the occluded object.

---

## Page 9

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 9
IGS4DGS
GT3DGStream HiCom
IGS4DGS
GT3DGStream HiCom
IGS4DGS
GT3DGStream HiCom
Frame Index = 1 Frame Index = 18 Frame Index = 33 Scale- GS
Scale- GS
Scale- GS
Fig. 5. Qualitative comparison results on the Google Immersive datasets(scene Dog). The frame index is 1, 18 and 33 from up to down. For each frame
index, we present the result of 4DGS, 3DStream, HiCom, IGS, Scale-GS , and ground truth.
In the frame 151, 4DGS and IGS fail to properly render
the dog’s eyes. The 3DGStream and HiCom display obvious
abnormal Gaussian points in the dog’s head area. Compared
to the baselines, the Scale-GS renders a richly furred dog
head without introducing floaters. The frame 183 indicates that
the Scale-GS accurately captures the natural shape variations
of the flame and intricate details. Although the baselines
achieve flame rendering from a global perspective, they do
not match the ground truth in the zoomed-in regions.
2) Comparison in Outdoor Scene: Fig. 5 presents outdoor
comparative results on the Google Immersive dataset. Com-pared to indoor scenes, outdoor scenes cover a larger area,
involve more rendering details, and are more challenging. The
red bounding boxes highlight regions showing the dog’s head.
In addition to reconstructing the changes in the dog’s facial
features and head, it is necessary to render the complex fur.
According to Fig. 5, the proposed Scale-GS reconstruct
clear and consistent face of the dog, while the baselines fail
to accurately represent the dog’s facial contours. Specifically,
4DGS, 3DGStream, and HiCom exhibit strong blurring around
the dog’s nose area in the frame 18. Moreover, the 3DGStream
and HiCom methods display abnormal Gaussian representa-

---

## Page 10

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 10
TABLE I
The quantitative results on different datasets. NV , MR, and GI are separately denoted NV3D dataset, Meeting room dataset, and Google Immersive dataset.
The best and second-best results are red and purple , respectively.
SSIM ↑ PSNR ↑ LPIPS ↓ Training time(s) ↓ FPS↑
Method NV MR GI NV MR GI NV MR GI NV MR GI NV MR GI
Deform 0.956 0.857 0.843 32.10 27.81 26.46 0.127 0.206 0.216 38 29 81 40 45 35
4DGS 0.959 0.870 0.853 32.23 28.69 27.88 0.109 0.184 0.214 8.2 7.5 73.5 30 38 26
3DG-S 0.958 0.906 0.868 32.93 29.09 28.60 0.101 0.145 0.209 9.6 7.7 76.1 210 252 190
HiCoM 0.967 0.909 0.852 33.28 29.15 28.68 0.111 0.152 0.208 7.1 3.9 60.8 256 284 212
IGS 0.965 0.909 0.909 33.62 30.13 29.72 0.109 0.143 0.168 3.6 3.2 43.2 204 251 186
Scale-GS 0.966 0.908 0.912 34.47 31.58 31.18 0.106 0.142 0.169 3.2 3.0 37.3 274 276 199
tions at the dog’s eyebrows. In contrast, the subtle details
around the eyes and nose of the Scale-GS are preserved with
high fidelity, especially maintaining consistent quality across
different viewing angles.
Overall, Fig. 4 and Fig. 5 demonstrate that Scale-
GS achieves superior visual quality and faithfulness compared
to the baselines, particularly in challenging scenarios involving
complex human interactions, dynamic elements like flame, and
intricate textures such as animal fur.
D. Quantitative Results
Tab. I presents the quantitative comparison to baselines on
the 3 different datasets. We evaluate the novel view quality
and rendering efficiency. To ensure a fair comparison, all
competing methods are evaluated using the same set of Gaus-
sians initialized from the 0-th frame, and the same variant of
Gaussian splatting rasterization is applied consistently across
all approaches.
According to Table I, Scale-GS outperforms the base-
lines and achieves the best reconstruction quality with the
fastest training speed. In the rendering quality aspect, Scale-
GS achieves a PSNR improvement from 33.62dB (second-best
IGS) to 34.47dB on NV3D dataset, from 30.13dB to 31.58dB
on Meeting Room dataset, and from 29.72dB to 31.18dB on
Google Immersive dataset, demonstrating consistent quality
enhancement across all evaluation scenarios. For SSIM met-
rics, Scale-GS achieves competitive performance with 0.912
on Google Immersive dataset (best), 0.966 on NV3D dataset
(0.001 below the best), and 0.908 on Meeting Room dataset
(0.001 below the best), indicating excellent structural preserva-
tion. The LPIPS scores show perceptual quality improvements,
with Scale-GS achieving the best result of 0.142 on Meeting
Room dataset and maintaining competitive performance on
other datasets.
The sub-optimal performance of competing methods can
be attributed to several limitations: (1) Global deformation
inefficiency: Methods like Deform and 4DGS apply defor-
mation to all Gaussians uniformly, leading to unnecessary
computations on static regions and reduced optimization fo-
cus on truly dynamic areas. (2) Single-scale representation
constraints: Traditional approaches like 3DG-S and HiCoM
operate at fixed resolutions, failing to leverage hierarchical
optimization strategies. In contrast, our multi-scale approach
enables rapid convergence by first focusing on coarse-scaleGaussians that capture major scene changes, then progres-
sively refining finer-scale Gaussians with increasing precision.
This hierarchical refinement allows the optimization to quickly
identify and target only the Gaussians that require updates at
each scale level, resulting in accelerated training convergence
and progressively improved rendering quality as finer details
are incorporated. (3) Lack of adaptive supervision: Existing
methods fail to adaptively select the most informative views
and spatial regions during training, resulting in suboptimal
resource allocation and slower convergence. In contrast, our
multi-scale framework with hybrid deformation-spawning pol-
icy and bidirectional adaptive masking addresses these limi-
tations systematically, enabling both superior reconstruction
quality and computational efficiency.
E. Ablation Study
1) The Evaluation of Scale Number: We conduct a group
of ablation study to investigate the influence of the scale
number of Scale-GS. We choose the Face Print scene from
the Google Immersive Dataset and set the scale to 2, 3, 4, and
5, respectively. The result of average SSIM, PSNR, LPIPS,
and training time is demonstrated in Table II.
The experimental results indicate that the configuration with
3 scales achieves the best performance across all metrics,
yielding an SSIM of 0.916, a PSNR of 31.233 dB, and an
LPIPS score of 0.154. It is noted that with the increase of
the scale number, the rendering quality may decline under the
same training iterations. This is because the redundant scale
requires training computation, thus the convergence of each
scale becomes insufficient. Therefore, 3 level of scales strike
the tradeoff between visual quality and training efficiency.
To further validate the visual performance gap across
different scale settings, in Fig. 6, we present a qualitative
comparison of the zoom-in region in the same scene. It can
be observed that the 3-scale configuration produces the most
faithful reconstruction, particularly in facial regions such as
the eyes, eyelashes, and hair strands, closely resembling the
ground truth. The hand region near the drawing board is
reconstructed the clearest details with the scale number as
3. In comparison, other scale configurations exhibit varying
degrees of Gaussian transparency or blurring in the hand area,
which negatively affects the rendering quality.
2) The Effectiveness of Hybrid Deformation-spawning: To
evaluate the effectiveness of the hybrid deformation-spawning

---

## Page 11

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 11
TABLE II
Ablation study of scale number on Google Immersive Dataset, where time
indicates training time.
Scale SSIM ↑ PSNR ↑ LPIPS ↓ Time(s) ↓
2 0.893 30.602 0.168 36
3 0.916 31.233 0.154 33
4 0.881 29.917 0.170 35
5 0.857 28.242 0.191 39
*Our setting is highlighted with a shaded background .
2、3、4、5尺度
w/o spawn w/o deformation
 ours
2 3
4 5
2 3
4 5
Fig. 6. The visual quality comparison with different numbers of scales. From
top-left to bottom-right: results with 2, 3, 4, and 5 scales, respectively.
strategy, we conduct an ablation study on the MeetRoom
dataset by comparing three variants: (1) the full hybrid strat-
egy, (2) w/o deformation, and (3) w/o spawn. The results,
summarized in Table III, show that the hybrid approach
achieves the best performance across all evaluated metrics,
in which SSIM is 0.913, PSNR is 31.602 dB, and LPIPS is
0.138, and the least training time per frame (2.9 seconds).
In contrast, removing either component leads to a noticeable
degradation in both reconstruction quality and efficiency.
Furthermore, we demonstrate the visual quality(top) and
the statistical PSNR and training time(bottom) comparison
among the full hybrid strategy, without deformation, and
without spawning in Fig. 7. According to the visual result,
the full hybrid strategy exhibits clear edge and accurate color
representation. Moreover, the details of the plant leaves and
the fabric textures are also well preserved. These visual results
further validate that the hybrid mechanism achieves high-
quality rendering and efficient training of dynamic scenes.
As shown in Fig. 7, our hybrid approach consistently
achieves the shortest training time per frame (averaging around
3.2 seconds), while both ablated variants require significantly
longer training periods. Specifically, the “w/o spawn” variant,
though taking more time than our method, has relatively more
stable training duration compared to “w/o deformation”; the
“w/o deformation” variant shows considerable training time
fluctuations throughout the sequence and exhibits the highest
training costs. Fig. 7(b) demonstrates the PSNR evolution
across frames, where our method maintains relatively stable
and high PSNR values (around 34.5 dB). The “w/o spawn”
variant can keep a decent level of PSNR for a while but shows
more pronounced fluctuations than our method as frames
progress, while the “w/o deformation” variant experiences aTABLE III
Ablation study of the hybrid deformation-spawning policy, where time
indicates training time.
SSIM ↑ PSNR ↑ LPIPS ↓ Time(s) ↓
deform + spawn 0.913 31.602 0.138 2.9
w/o deformation 0.854 26.719 0.245 4.6
w/o spawning 0.887 29.304 0.127 3.8
2、3、4、5尺度
w/o spawning w/o deformation Ours
2 3
4 5
2 3
4 5
(a) Ablation study on training time of hybrid 
deformation and spawn(b) Ablation study on PSNR of hybrid 
deformation and spawnFrame indexTraining time
PSNR
Frame index•Statistical result •Visual quality
Fig. 7. The visual quality(from left to right: full hybrid, w/o deformation,
and w/o spawning) and statistical results of the ablation study on hybrid
deformation-spawning in the Meeting Room dataset.
more gradual yet evident decline in PSNR over time and over-
all lower reconstruction quality. The results demonstrate that
the Scale-GS method not only trains faster but also maintains
more stable rendering quality throughout the sequence.
Even the dynamic nature of the scene leads to unstable
PSNR variations, the hybrid deformation and spawning strat-
egy performs the best in both visual quality and training effi-
ciency in most cases. This suggests that the joint optimization
of deformation and spawning enables more efficient and stable
learning of dynamic scene representations.
3) Viewpoint Selection: To validate the effectiveness of
Bidirectional Adaptive Masking (BAM) mechanism, we con-
duct ablation studies on five challenging scenes from the
Google Immersive Dataset: Car, Goats, Dogs, Face Paint, and
Welder. These scenes feature diverse dynamic content ranging
from fast-moving objects to complex deformations, providing
a comprehensive testbed for evaluating view selection strategy.
The quantitative results are presented in Table IV. Scale-
GS BAM-based view selection consistently outperforms the
baseline across all tested scenes, achieving improvements of
0.7-1.7 dB in PSNR and 0.007-0.035 in SSIM. The most
significant improvement is observed in the Welder scene,
where BAM selection achieves 30.281 dB PSNR compared
to 28.552 dB without selection, representing a substantial
1.729 dB gain. This scene benefits particularly due to its
complex welding sparks and rapid lighting changes, where
targeted view selection effectively focuses learning on the most
informative perspectives.
The improvement across diverse scene types demonstrates
that BAM successfully identifies and prioritizes views that pro-
vide meaningful supervision for dynamic regions. By filtering
out redundant or less informative viewpoints, the BAM not

---

## Page 12

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 12
TABLE IV
The ablation study of viewpoint selection results, where time indicates
training time.
BAM selection w/o BAM selection
Scene PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑
Car 31.320 +1.012 0.917 +0.011 30.308 0.906
Goats 31.778 +1.150 0.922 +0.007 30.628 0.915
Dogs 31.022 +0.737 0.914 +0.021 30.285 0.893
Paint 31.751 +0.906 0.916 +0.006 30.845 0.910
Welder 30.281 +1.729 0.901 +0.035 28.552 0.866
only improves reconstruction quality but also enhances train-
ing efficiency. The bidirectional masking—selecting both spa-
tial anchors and camera views—proves essential for handling
the complexity of multi-view dynamic scene reconstruction.
These results validate that bidirectional view selection
mechanism effectively ensure that Gaussians receive super-
vision from the most relevant camera perspectives, leading to
more accurate and stable dynamic scene modeling.
V. C ONCLUSION AND FUTURE WORK.
Conclusion. In this work, we present Scale-GS, a scalable
Gaussian Splatting framework for efficient and redundancy-
aware dynamic scene training. The proposed Scale-GS pro-
poses the anchor-based multi-scale Gaussian representation in-
tegrating a hybrid deformation–spawning optimization, redun-
dant Gaussian removing, and a bidirectional adaptive masking
module. Our method effectively reduces the computational
overhead of existing approaches by filtering static or irrelevant
Gaussian spheres. The hybrid deformation–spawning strategy
preserves the structured inference from deformation while
enhancing the model’s capacity to represent large-scale dy-
namic motions. Extensive experiments demonstrate that Scale-
GS achieves superior visual fidelity and significantly acceler-
ates training. The proposed framework opens up promising
opportunities for remote immersive experiences such as VR,
AR, and immersive video conferencing.
Future work. To further reduce the redundant computation
and storage, we aim to incorporate semantic understanding into
the multi-scale representation to improve the representation ef-
ficiency of Gaussian spheres. The semantic information guides
the allocation and prioritization of Gaussian spheres by adapt-
ing their scale, density, and training frequency according to
the semantic importance and structural complexity of different
areas. Furthermore, integrating Scale-GS with neural compres-
sion could further enhance scalability and enable deployment
in bandwidth and resource-constrained environments.
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk ¨uhler, and G. Drettakis, “3D Gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph. ,
vol. 42, no. 4, pp. 139–1, 2023.
[2] Y .-H. Huang, Y .-T. Sun, Z. Yang, X. Lyu, Y .-P. Cao, and X. Qi, “Sc-
gs: Sparse-controlled gaussian splatting for editable dynamic scenes,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2024, pp. 4220–
4230.
[3] X. Tong, T. Shao, Y . Weng, Y . Yang, and K. Zhou, “As-rigid-as-possible
deformation of gaussian radiance fields,” IEEE Trans. Vis. Comput.
Graph. , 2025.[4] Z. Fan, S.-S. Huang, Y . Zhang, D. Shang, J. Zhang, Y . Guo, and
H. Huang, “RGAvatar: Relightable 4D Gaussian avatar from monocular
videos,” IEEE Trans. Vis. Comput. Graph , 2025.
[5] R. Fan, J. Wu, X. Shi, L. Zhao, Q. Ma, and L. Wang, “Fov-GS: Foveated
3D Gaussian splatting for dynamic scenes,” IEEE Trans. Vis. Comput.
Graph , 2025.
[6] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y . Zhang, and X. Jin, “Deformable
3D Gaussians for high-fidelity monocular dynamic scene reconstruc-
tion,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2024,
pp. 20 331–20 341.
[7] Z. Li, Z. Chen, Z. Li, and Y . Xu, “Spacetime Gaussian feature splat-
ting for real-time dynamic view synthesis,” in Proc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit. , 2024, pp. 8508–8520.
[8] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and
X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,”
inProc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2024, pp.
20 310–20 320.
[9] J. Yan, R. Peng, L. Tang, and R. Wang, “4D Gaussian splatting
with scale-aware residual field and adaptive optimization for real-time
rendering of temporally complex dynamic scenes,” in Proc. 32nd ACM
Int. Conf. Multimedia , 2024, pp. 7871–7880.
[10] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3DGStream:
On-the-fly training of 3D Gaussians for efficient streaming of photo-
realistic free-viewpoint videos,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. , 2024, pp. 20 675–20 685.
[11] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y . Liu, “Tensor4d:
Efficient neural 4D decomposition for high-fidelity dynamic reconstruc-
tion and rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. , 2023, pp. 16 632–16 642.
[12] S. J. Gortler, R. Grzeszczuk, R. Szeliski, and M. F. Cohen, “The
lumigraph,” in Proc. 23rd Annu. Conf. Comput. Graph. Interactive
Techn. , 2023, pp. 453–464.
[13] X. Meng, R. Du, J. F. JaJa, and A. Varshney, “3D-kernel foveated
rendering for light fields,” IEEE Trans. Vis. Comput. Graph , vol. 27,
no. 8, pp. 3350–3360, 2020.
[14] M. Levoy and P. Hanrahan, “Light field rendering,” in Proc. 23rd Annu.
Conf. Comput. Graph. Interactive Techn. , 2023, pp. 441–452.
[15] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “NeRF: Representing scenes as neural radiance fields for
view synthesis,” Commun. ACM , vol. 65, no. 1, pp. 99–106, 2021.
[16] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance
fields,” in Proc. Eur. Conf. Comput. Vis. Springer, 2022, pp. 333–350.
[17] X.-S. Hu, X.-Y . Lin, Y .-J. Liu, M.-H. Xiang, Y .-Q. Guo, Y . Xing, and Q.-
H. Wang, “Culling-based real-time rendering with accurate ray sampling
for high-resolution light field 3D display,” IEEE Trans. Vis. Comput.
Graph , 2024.
[18] T. M ¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Trans. Graph. ,
vol. 41, no. 4, pp. 1–15, 2022.
[19] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and
A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2022, pp. 5501–
5510.
[20] Z. Chen, T. Funkhouser, P. Hedman, and A. Tagliasacchi, “MobileNeRF:
Exploiting the polygon rasterization pipeline for efficient neural field
rendering on mobile architectures,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. , 2023, pp. 16 569–16 578.
[21] K. Ye, H. Wu, X. Tong, and K. Zhou, “A real-time method for inserting
virtual objects into neural radiance fields,” IEEE Trans. Vis. Comput.
Graph , 2024.
[22] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, and A. Kanazawa, “Plenoctrees
for real-time rendering of neural radiance fields,” in Proc. IEEE/CVF
Int. Conf. Comput. Vis. , 2021, pp. 5752–5761.
[23] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-NeRF: A multiscale representation for anti-
aliasing neural radiance fields,” in Proc. IEEE/CVF Int. Conf. Comput.
Vis., 2021, pp. 5855–5864.
[24] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2022, pp. 5470–
5479.
[25] J. Lu, T. Shao, H. Wang, Y .-L. Yang, Y . Yang, and K. Zhou, “Relightable
detailed human reconstruction from sparse flashlight images,” IEEE
Trans. Vis. Comput. Graph , 2024.
[26] F. Wimbauer, N. Yang, C. Rupprecht, and D. Cremers, “Behind
the scenes: Density fields for single view reconstruction,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2023, pp. 9076–9086.

---

## Page 13

IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS, VOL. XX, NO. XX, AUGUST 2025 13
[27] A. Yu, V . Ye, M. Tancik, and A. Kanazawa, “pixelNeRF: Neural radiance
fields from one or few images,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. , 2021, pp. 4578–4587.
[28] T. Lu, M. Yu, L. Xu, Y . Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-
GS: Structured 3D Gaussians for view-adaptive rendering,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2024, pp. 20 654–
20 664.
[29] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, “Octree-
GS: Towards consistent real-time rendering with lod-structured 3D
Gaussians,” arXiv:2403.17898 , 2024.
[30] D. Chen, H. Li, W. Ye, Y . Wang, W. Xie, S. Zhai, N. Wang, H. Liu,
H. Bao, and G. Zhang, “PGSR: Planar-based Gaussian splatting for
efficient and high-fidelity surface reconstruction,” IEEE Trans. Vis.
Comput. Graph , 2024.
[31] Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient
adaptive surface reconstruction in unbounded scenes,” ACM Trans.
Graph. , vol. 43, no. 6, pp. 1–13, 2024.
[32] J. Lin, J. Gu, L. Fan, B. Wu, Y . Lou, R. Chen, L. Liu, and J. Ye,
“HybridGS: Decoupling transients and statics with 2D and 3D Gaussian
splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. ,
2025, pp. 788–797.
[33] Y . Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac: Hash-grid
assisted context for 3D Gaussian splatting compression,” in Proc. Eur.
Conf. Comput. Vis. Springer, 2024, pp. 422–438.
[34] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3D Gaussian
representation for radiance field,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. , 2024, pp. 21 719–21 728.
[35] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed
3D Gaussian splatting for accelerated novel view synthesis,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2024, pp. 10 349–
10 358.
[36] D. Li, S.-S. Huang, and H. Huang, “MPGS: Multi-plane Gaussian
splatting for compact scenes rendering,” IEEE Trans. Vis. Comput.
Graph , 2025.
[37] Y . Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, “Colmap-
free 3D Gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. , 2024, pp. 20 796–20 805.
[38] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, “LGM: Large
multi-view Gaussian model for high-resolution 3D content creation,” in
Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 1–18.
[39] K. Tang, S. Yao, and C. Wang, “iVR-GS: Inverse volume rendering
for explorable visualization via editable 3D Gaussian splatting,” IEEE
Trans. Vis. Comput. Graph , 2025.
[40] Z.-X. Zou, Z. Yu, Y .-C. Guo, Y . Li, D. Liang, Y .-P. Cao, and S.-H. Zhang,
“Triplane meets Gaussian splatting: Fast and generalizable single-view
3D reconstruction with transformers,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. , 2024, pp. 10 324–10 335.
[41] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su,
“MVSNeRF: Fast generalizable radiance field reconstruction from multi-
view stereo,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. , 2021, pp.
14 124–14 133.
[42] M. M. Johari, Y . Lepoittevin, and F. Fleuret, “GeoNeRF: Generalizing
NeRF with geometry priors,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. , 2022, pp. 18 365–18 375.
[43] S. Szymanowicz, C. Rupprecht, and A. Vedaldi, “Splatter image: Ultra-
fast single-view 3D reconstruction,” in Proc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. , 2024, pp. 10 208–10 217.
[44] C. Zhang, Y . Zou, Z. Li, M. Yi, and H. Wang, “Transplat: Generalizable
3D Gaussian splatting from sparse multi-view images with transform-
ers,” in Proc. AAAI Conf. Artif. Intell. , vol. 39, no. 9, 2025, pp. 9869–
9877.
[45] S. Zheng, B. Zhou, R. Shao, B. Liu, S. Zhang, L. Nie, and Y . Liu, “GPS-
Gaussian: Generalizable pixel-wise 3D Gaussian splatting for real-time
human novel view synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. , 2024, pp. 19 680–19 690.
[46] D. Charatan, S. L. Li, A. Tagliasacchi, and V . Sitzmann, “pixelSplat:
3D Gaussian splats from image pairs for scalable generalizable 3D re-
construction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. ,
2024, pp. 19 457–19 467.
[47] H. Xu, S. Peng, F. Wang, H. Blum, D. Barath, A. Geiger, and
M. Pollefeys, “DepthSplat: Connecting gaussian splatting and depth,”
inProc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2025, pp.
16 453–16 463.
[48] Y . Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-
J. Cham, and J. Cai, “MVSplat: Efficient 3D Gaussian splatting from
sparse multi-view images,” in Proc. Eur. Conf. Comput. Vis. Springer,
2024, pp. 370–386.[49] T. Liu, G. Wang, S. Hu, L. Shen, X. Ye, Y . Zang, Z. Cao, W. Li, and
Z. Liu, “MVSGaussian: Fast generalizable gaussian splatting reconstruc-
tion from multi-view stereo,” in Proc. Eur. Conf. Comput. Vis. Springer,
2024, pp. 37–53.
[50] K. Zhang, S. Bi, H. Tan, Y . Xiangli, N. Zhao, K. Sunkavalli, and Z. Xu,
“GS-LRM: Large reconstruction model for 3D Gaussian splatting,” in
Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 1–19.
[51] Y . Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “MVSNet: Depth inference
for unstructured multi-view stereo,” in Proc. Eur. Conf. Comput. Vis. ,
2018, pp. 767–783.
[52] A. Cao and J. Johnson, “HexPlane: A fast representation for dynamic
scenes,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2023,
pp. 130–141.
[53] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa,
“K-planes: Explicit radiance fields in space, time, and appearance,”
inProc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2023, pp.
12 479–12 488.
[54] X. Guo, J. Sun, Y . Dai, G. Chen, X. Ye, X. Tan, E. Ding, Y . Zhang, and
J. Wang, “Forward flow for novel view synthesis of dynamic scenes,”
inProc. IEEE/CVF Int. Conf. Comput. Vis. , 2023, pp. 16 022–16 033.
[55] H. Lin, S. Peng, Z. Xu, T. Xie, X. He, H. Bao, and X. Zhou, “High-
fidelity and real-time novel view synthesis for dynamic scenes,” in
SIGGRAPH Asia 2023 Conference Papers , 2023, pp. 1–9.
[56] J.-W. Liu, Y .-P. Cao, W. Mao, W. Zhang, D. J. Zhang, J. Keppo, Y . Shan,
X. Qie, and M. Z. Shou, “DeVRF: Fast deformable voxel radiance fields
for dynamic scenes,” Adv. Neural Inf. Process. Syst. , vol. 35, pp. 36 762–
36 775, 2022.
[57] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-
NeRF: Neural radiance fields for dynamic scenes,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. , 2021, pp. 10 318–10 327.
[58] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and
M. Wu, “Neural residual radiance fields for streamably free-viewpoint
videos,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. , 2023,
pp. 76–87.
[59] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y . Xu, and
A. Geiger, “NeRFPlayer: A streamable dynamic scene representation
with decomposed neural radiance fields,” IEEE Trans. Vis. Comput.
Graph , vol. 29, no. 5, pp. 2732–2742, 2023.
[60] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, “Streaming radiance
fields for 3D video synthesis,” Adv. Neural Inf. Process. Syst. , vol. 35,
pp. 13 485–13 498, 2022.
[61] Q. Gao, J. Meng, C. Wen, J. Chen, and J. Zhang, “HiCoM: Hierarchical
coherent motion for dynamic streamable scenes with 3D Gaussian
splatting,” Adv. Neural Inf. Process. Syst. , vol. 37, pp. 80 609–80 633,
2024.
[62] J. Yan, R. Peng, Z. Wang, L. Tang, J. Yang, J. Liang, J. Wu, and R. Wang,
“Instant Gaussian stream: Fast and generalizable streaming of dynamic
scene reconstruction via gaussian splatting,” in Proc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit. , 2025, pp. 16 520–16 531.
[63] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, “Motion-aware 3D
Gaussian splatting for efficient dynamic scene reconstruction,” IEEE
Trans. Circuits Syst. Video Technol. , 2024.
[64] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3D
Gaussians: Tracking by persistent dynamic view synthesis,” in Proc.
IEEE Int. Conf. 3D Vis. IEEE, 2024, pp. 800–809.
[65] T. Mizuho, T. Narumi, and H. Kuzuoka, “Reduction of forgetting
by contextual variation during encoding using 360-degree video-based
immersive virtual environments,” IEEE Trans. Vis. Comput. Graph ,
2024.
[66] C. Groth, S. Fricke, S. Castillo, and M. Magnor, “Wavelet-based fast
decoding of 360 videos,” IEEE Trans. Vis. Comput. Graph , vol. 29,
no. 5, pp. 2508–2516, 2023.
[67] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim,
T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe et al. , “Neural
3D video synthesis from multi-view video,” in Proc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit. , 2022, pp. 5521–5531.
[68] M. Broxton, J. Flynn, R. Overbeck, D. Erickson, P. Hedman, M. Duvall,
J. Dourgarian, J. Busch, M. Whalen, and P. Debevec, “Immersive light
field video with a layered mesh representation,” ACM Trans. Graph. ,
vol. 39, no. 4, pp. 86–1, 2020.