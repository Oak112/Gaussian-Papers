

---

## Page 1

arXiv:2505.13215v1  [cs.CV]  19 May 2025Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation
Seungjun Oh1Younggeun Lee1Hyejin Jeon1Eunbyung Park2
1Department of Artificial Intelligence, Sungkyunkwan University
2Department of Artificial Intelligence, Yonsei University
https://ohsngjun.github.io/3D-4DGS/
Figure 1. Left: Rendering results on the coffee martini scene. Right: PSNR vs. training time. The proposed method converges in 12
minutes while maintaining competitive rendering quality. All methods were evaluated under the same machine equipped with the NVIDIA
RTX4090 GPU, except for 4D-Rotor GS [11]—whose results were estimated from iteration counts since the code is not publicly available.
Abstract
Recent advancements in dynamic 3D scene reconstruc-
tion have shown promising results, enabling high-fidelity
3D novel view synthesis with improved temporal consis-
tency. Among these, 4D Gaussian Splatting (4DGS) has
emerged as an appealing approach due to its ability to
model high-fidelity spatial and temporal variations. How-
ever, existing methods suffer from substantial computational
and memory overhead due to the redundant allocation of
4D Gaussians to static regions, which can also degrade im-
age quality. In this work, we introduce hybrid 3D–4D Gaus-
sian Splatting (3D-4DGS), a novel framework that adap-
tively represents static regions with 3D Gaussians while re-
serving 4D Gaussians for dynamic elements. Our method
begins with a fully 4D Gaussian representation and itera-
tively converts temporally invariant Gaussians into 3D, sig-
nificantly reducing the number of parameters and improv-
ing computational efficiency. Meanwhile, dynamic Gaus-
sians retain their full 4D representation, capturing complex
motions with high fidelity. Our approach achieves signifi-
cantly faster training times compared to baseline 4D Gaus-
sian Splatting methods while maintaining or improving the
visual quality.1. Introduction
Accurately representing and rendering complex dynamic
3D scenes is fundamental to a wide range of applications,
including immersive media for virtual and augmented re-
ality. For example, in commercial and industrial domains
such as sports broadcasting, film production, and live per-
formances, the demand for high-quality dynamic scene re-
construction continues to grow, driven by the need for en-
hanced viewer engagement. While significant progress has
been made, achieving high-fidelity, computationally effi-
cient, and temporally coherent modeling of dynamic scenes
remains a challenging problem.
Recent advances in neural rendering, particularly Neural
Radiance Fields (NeRF) [4, 5, 14, 36, 38, 50], have emerged
as a powerful representation for novel view synthesis and
3D scene reconstruction, leveraging neural networks, grid-
based data structures, and volumetric rendering [6]. Exten-
sions of NeRF to dynamic 3D scene modeling [7, 15, 20,
29, 34, 35, 43–45, 49, 54] have shown promising results,
enabling the reconstruction of time-varying environments
with improved fidelity. However, real-time and high-fidelity
rendering of complex dynamic scenes continues to be an
open problem due to the computational cost of volume ren-
dering and the complexity of spatio-temporal modeling.
1

---

## Page 2

More recently, 3D Gaussian Splatting (3DGS) [18] has
become a promising alternative to NeRF-based approaches
for 3D scene reconstruction and novel view synthesis , of-
fering improved quality and real-time rendering capabili-
ties. Unlike NeRF, which relies on implicit representa-
tion and computationally expensive volumetric rendering,
3DGS represents scenes as a collection of Gaussian primi-
tives and leverages a fast rasterization. Several extensions
have been proposed to adapt 3DGS for dynamic 3D scene
reconstruction, incorporating motion modeling and tempo-
ral consistency to handle time-varying environments.
Two primary paradigms have been developed for ap-
plying 3DGS to dynamic 3D capture. The first approach
extends 3D Gaussians to dynamic 3D scenes by tracking
Gaussians over time [17, 21, 22, 26, 56, 61], using tech-
niques such as multi-layer perceptrons [26], temporal resid-
uals [56], or interpolation functions [22]. These methods
leverage temporal redundancy across frames to improve
the representation efficiency and accelerate training, but
they often struggle with fast-moving objects. The second
paradigm, directly optimizing 4D Gaussians , represents the
entire spatio-temporal volume as a set of splatted 4D Gaus-
sians [11, 30, 60]. While this approach enables high-quality
reconstructions, it incurs significant memory and computa-
tional overhead. Furthermore, allocating 4D Gaussians to
inherently static regions is inefficient, as these areas do not
benefit from time-varying parameters [10].
In this work, we propose a hybrid 3D-4D Gaussian
Splatting (3D-4DGS) framework that addresses the inef-
ficiencies of conventional 4DGS pipelines. A key limi-
tation of 4DGS [60] is their treatment of static regions,
which often requires multiple 4D Gaussians across differ-
ent timesteps. While an optimal solution would involve
assigning large scales along the temporal axis to represent
static regions more effectively, this rarely occurs in prac-
tice. As illustrated in Fig. 2, most Gaussians exhibit small
temporal scales, leading to redundant memory usage and
increased computational overhead. Building on this obser-
vation, we propose a hybrid approach that models static
regions with 3D Gaussians while reserving 4D Gaussians
for dynamic elements. The proposed approach significantly
reduces the number of Gaussians, leading to lower mem-
ory consumption and faster training speed. As shown in
Fig. 1, we achieved near state-of-the-art reconstruction fi-
delity while substantially reducing training time compared
to prior 4DGS baselines.
Our approach begins by modeling all Gaussians as 4D
and then adaptively identifying those with minimal tempo-
ral variation across the sequence. These Gaussians are clas-
sified as static and converted into a purely 3D representation
by discarding the time dimension, effectively freezing their
position, rotation, and color parameters. Meanwhile, fully
dynamic Gaussians retain their 4D nature to capture com-plex motion. Importantly, this classification is not a one-
time process but is performed iteratively at each densifica-
tion stage, progressively refining the regions that truly re-
quire 4D modeling. The final rendering pipeline seamlessly
integrates both 3D and 4D Gaussians, projecting them into
screen space for alpha compositing. This design ensures
that temporal modeling is applied where necessary, cap-
turing motion effectively while eliminating redundant over-
head in static regions.
We demonstrate the effectiveness of the proposed 3D-
4DGS on two standard challenging datasets: Neural 3D
Video (N3V) [25], which primarily comprises 10-second
multi-view videos (plus one 40-second long sequence), and
Technicolor [47], featuring 16-camera light field captures of
short but complex scenes. Our method consistently achieves
competitive or superior PSNR and SSIM scores while sig-
nificantly reducing training times. Additionally, we conduct
ablation studies to reveal how key design choices—such as
the scale threshold and opacity reset strategies—impact fi-
nal quality and efficiency. We summarize our main contri-
butions as follows:
•Hybrid 3D–4D representation. We introduce a novel
approach, 3D-4DGS , that dynamically classifies Gaus-
sians as either static (3D) or dynamic (4D), enabling an
adaptive strategy that optimizes storage and computa-
tion.
•Significantly reduced training time. By removing re-
dundant temporal parameters for static Gaussians, our
approach converges about 3–5 ×faster than baseline
4DGS methods while preserving fidelity.
•Memory efficiency. Converting large static regions to
3D Gaussians lowers memory requirements, allowing
longer sequences or more detailed scenes given the same
hardware specification.
•High-fidelity dynamic modeling. Focusing time-
variant parameters on genuinely dynamic content
achieves comparable or superior visual quality to 4DGS
only representations across various challenging scenes.
2. Related Work
2.1. Novel View Synthesis
The field of novel view synthesis has transitioned from
fully implicit neural fields to more explicit representations
that enable faster training and rendering. Neural Radi-
ance Fields (NeRF) [36] introduced the foundational ap-
proach by modeling scenes as continuous volumetric func-
tions from multi-view images. However, its reliance on
deep MLP weights results in slow training and render-
ing times, motivating extensive research into more effi-
cient alternatives. A key development in this direction in-
volves replacing fully implicit representations with voxel
grids, hash-encodings, or compact tensor-based structures
2

---

## Page 3

[3, 8, 14, 15, 38, 39, 50, 51]. These approaches signif-
icantly reduce computational overhead by using spatially
structured representations, enabling near-real-time render-
ing while maintaining high reconstruction fidelity.
More recently, point-based approaches have emerged as
a promising alternative, culminating in 3D Gaussian Splat-
ting (3DGS) [18], which represents a scene as a collec-
tion of anisotropic Gaussian primitives. By leveraging its
explicit nature and eliminating the need for costly empty-
space sampling, 3DGS enables real-time, high-fidelity ren-
dering while efficiently utilizing modern GPU architec-
tures. Despite these advantages, optimizing 3DGS for
broader scalability presents challenges in memory effi-
ciency and training speed. In terms of compact representa-
tions, several methods have explored utilizing vector quan-
tization [23, 40–42, 55], entropy coding [9], and image
or video codes [24, 37]. Regarding fast training, Mini-
Splatting2 [12] and Turbo-GS [31] demonstrate that near-
minute training times are feasible via aggressive densifica-
tion and careful tuning, suggesting that 3DGS can be opti-
mized far more quickly with the right strategies, while other
works [33, 53] improve the convergence speed by introduc-
ing flexible optimization techniques and density control.
2.2. Dynamic Scene Representation
Dynamic scene reconstruction extends static modeling tech-
niques to time-varying objects and environments. Early
works, such as D-NeRF [45] and Neural V olumes [29], used
time-conditioned radiance fields to track temporal changes,
enabling the representation of dynamic objects and their in-
teractions over time. More recent methods based on explicit
representations [7, 13, 15, 48, 49] decompose 4D scenes
into lower-dimensional spaces, providing efficient ways to
capture spatial and temporal dynamics both while improv-
ing scalability and rendering performance.
Building upon 3DGS, extended methods [11, 22, 26, 60]
represent scenes with 4D Gaussian primitives, incorporat-
ing space-time geometry and corresponding features for
real-time dynamic content rendering. Other approaches [32,
59] model motion through 6-DoF trajectories or defor-
mation fields, learning to transform Gaussians between
frames. However, treating every scene component as dy-
namic can be inefficient, especially when the background
remains static while only certain components move. Re-
cent work has also explored online or streaming reconstruc-
tion [16, 28, 52], where new frames are processed incre-
mentally, and Gaussian parameters are adaptively updated
based on motion characteristics. While these methods han-
dle continuous capture effectively, they also require com-
plex Gaussian management in dynamics.
Our approach leverages the insight that modeling the en-
tire scene with dynamic components is inefficient. We dis-
tinguish between static and dynamic content by introduc-ing a novel scale-based classification method to automati-
cally identify static regions, improving training and render-
ing speed, memory efficiency, and achieving performance
on par with existing state-of-the-art methods for dynamic
novel view synthesis.
3. Preliminary
In this section, we provide an overview of 3D Gaussian
Splatting (3DGS) and its extension to dynamic scenes, 4D
Gaussian Splatting (4DGS), which serve as the foundation
for our approach.
3.1. 3D Gaussian Splatting
3D Gaussian Splatting (3DGS) represents a scene by opti-
mizing a collection of anisotropic 3D Gaussian ellipsoids,
each defined by its center position µ, and covariance matrix
Σ, which encodes spatial extent and orientation:
G(x) = exp
−1
2(x−µ)⊤Σ−1(x−µ)
, (1)
where xdenotes a point in 3D space. To impose a structured
representation, the covariance matrix Σis reparameterized
using a rotation matrix Rand a scaling matrix S:
Σ =R S S⊤R⊤, (2)
where, Scontrols the scaling along the principal axes, and
Rdefines the orientation. Rendering is performed via alpha
compositing, aggregating Gaussian contributions per pixel:
C=X
i∈Nciαii−1Y
j=1(1−αj), (3)
where ciandαidenote the color and opacity of the i-th
Gaussian, and Ndenotes a set of Gaussians affecting a pixel
to be rendered. This approach ensures a smooth and realis-
tic blending of overlapping Gaussian contributions.
3.2. 4D Gaussian Splatting
Dynamic scene modeling requires extending the 3D formu-
lation to model the temporal variations. 4D Gaussian Splat-
ting (4DGS) [60] achieves this by incorporating an addi-
tional temporal dimension into the 3D Gaussian representa-
tion, enabling the capture of motion and scene changes over
time.
In the 4DGS framework, the spatial and temporal com-
ponents are jointly modeled, resulting in four-dimensional
rotation matrix, formulated as follows,
R=RlRr=
a−b−c−d
b a −d c
c d a −b
d−c b a

p−q−r−s
q p s −r
r−s p q
s r −q p
(4)
3

---

## Page 4

Figure 2. Distribution of the t-axis scale for Gaussians in the
coffee martini scene. Most Gaussians cluster at smaller
scales, indicating dynamic content, while a minority have larger
scales that suggest static regions.
where RlandRrare left and right rotation matrix,
each constructed by a quaternion vector, (a, b, c, d )and
(p, q, r, s ).
The temporally conditioned mean and covariance for a
given time tis computed as,
µxyz|t=µ1:3+ Σ 1:3,4Σ−1
4,4(t−µt), (5)
Σxyz|t= Σ 1:3,1:3−Σ1:3,4Σ−1
4,4Σ4,1:3. (6)
For further details, please refer to the original 4DGS pa-
per [60].
4. Hybrid 3D-4D Gaussian Splatting
In this section, we present the proposed hybrid 3D-
4D Gaussian splatting ( 3D-4DGS ). First, we describe a
method that adaptively identifies static and dynamic regions
throughout the training process (Sec. 4.1). Second, we in-
troduce how we can convert 4D Gaussians to 3D Gaussians
(Sec. 4.2). Then, we will discuss hybrid rendering and opti-
mization to train the parameters of the proposed 3D-4DGS
framework (Sec. 4.3).
4.1. Static and Dynamic Region Identification
The prior works [22, 28] often identify static and dynamic
content by analyzing the flow of Gaussians. Since our ap-
proach does not explicitly model the flows of 3D Gaussians,
we leverage a 4D coordinate system, where each Gaus-
sian has a scale parameter along the time axis. Concretely,
each Gaussian is initially modeled as a 4D Gaussian, and
fori-th Gaussian, its effective time-axis scale is given by
exp(st,i), where exp(·)is an exponential activation func-
tion and st,i∈Rdenotes the time-axis scale parameter for
i-th Gaussian. If exp(st,i)exceeds a predefined threshold
τ, the Gaussian is classified as static Gaussian.We empirically determined the threshold τbased on the
distribution of temporal scales in fully trained 4DGS [60]
and the characteristics of the target datasets. For example,
as shown in Fig. 2, most Gaussians exhibit small tempo-
ral scales (below 0.5). We choose τto lie in the “valley”
between these smaller (dynamic) scales and larger (static)
values.
Intuitively, a larger temporal scale indicates that the
Gaussian covers a static part of the scene without high-
frequency temporal changes. Once a Gaussian’s scale sur-
passes τ, it is converted from a 4D (spatio-temporal) Gaus-
sian to a 3D (spatial only) Gaussian. Importantly, this clas-
sification is performed dynamically at each densification
stage rather than in a one-off preprocessing step. In other
words, a Gaussian can remain 4D during early iterations
and later transition to 3D once it expands to a larger tempo-
ral size. By continuously applying this process, our method
adaptively separates static background elements from dy-
namic elements throughout the optimization process.
4.2. 3D–4D Gaussian Conversion
We convert each 4D Gaussian to a 3D Gaussian by discard-
ing its temporal component and preserving its spatial com-
ponents. More specifically, a 4D Gaussian is characterized
by a mean
µ4D= (µx, µt), (7)
where µx∈R3represents the spatial center and µt∈Ren-
codes the temporal coordinate. In addition, each Gaussian
maintains a 4×4rotation matrix R4D, which determines
how the Gaussian is oriented in the joint spatio-temporal
domain. In principle, R4Dcan mix spatial and temporal
axes, allowing the Gaussian to “tilt” across time.
Forstatic Gaussians (those spanning the entire sequence
without localized time variation), R4Deffectively operates
as a block-diagonal transform: the top-left 3×3sub-block
is a pure spatial rotation, and the time dimension remains
separate. Formally,
R4D=R3D0
0⊤1
(ideal static case), (8)
where R3D∈SO(3)is an orthonormal 3×3rotation ma-
trix and 0is a three-dimensional zero vector. While this
ideal case rarely happens in practice, we observe that by re-
taining only R3Dinformation does not significantly affect
the training process.
The corresponding unit quaternion for R3Dmatrix,
q3D= (w, x, y, z ), is derived as follows:
4

---

## Page 5

Figure 3. Overview of our hybrid 3D–4D Gaussian Splatting framework. (a) 4D Gaussians are optimized over time, and those exceeding a
temporal scale threshold ( τ) are converted into 3D Gaussians. (b) Both 3D and 4D Gaussians are projected into screen space, assigned tile
and depth keys, and sorted for rasterization. The rendered image is generated by blending static (3D) and dynamic (4D) Gaussians.
w=1
2p
1 + tr( R3D),
x=R3D(3,2)−R3D(2,3)
4w,
y=R3D(1,3)−R3D(3,1)
4w,
z=R3D(2,1)−R3D(1,2)
4w,(9)
where tr(·)is a trace operator, and R3D(·,·)denotes an el-
ement of the R3Dmatrix given an index.
Next, the temporal component of the mean, µt, is dis-
carded, and the spatial mean µxis retained as the 3D posi-
tion of the Gaussian. Since the Gaussian is static, its posi-
tion no longer changes over time; it remains fixed at µxin
every time step. Also, its appearance attributes–including
opacity σand spherical harmonic (SH) color coefficients–
remain unchanged since static content does not require
time-dependent updates. Consequently, each converted 3D
Gaussian is fully specified by (µx, q3D, sx, sy, sz, σ,SH),
where q3Dprovides the orientation and sx, sy, szspecify
the ellipsoid’s principal scales. By converting all time-
invariant Gaussians in this manner, we eliminate their de-
pendence on temporal variable tand reduce the dimension-
ality of the model. Meanwhile, dynamic Gaussians retain
their full 4D parameterization (including time-based trans-
formations). At runtime, each static Gaussian remains iden-
tical across frames, whereas each dynamic Gaussian is com-
puted conditioned on the current timestamp.
4.3. Optimization and Rendering Pipeline
We perform a short initial training phase (up to 500 itera-
tions) with the full 4DGS model, allowing the 4D Gaussians
to stabilize. We then apply the proposed static/dynamic
identification scheme to split 4DGS into two groups: 3D
and 4D Gaussians. Alongside this process, we apply adap-tive densification and pruning separately to 3D and 4D
Gaussians (also every 100 iteration), ensuring continuous
refinement within their respective optimization pipelines.
This split mechanism and separate optimization substan-
tially accelerate the training. In the original 4DGS training,
only a small subset of 4D Gaussians is updated per training
iteration, as many are culled when they do not contribute
significantly to the rendering of training image timesteps.
On the other hand, our approach updates static 3D Gaus-
sians in every training iteration, leading to much faster con-
vergence. As a result, our model typically converges in ap-
proximately 6K iterations for 10-second dynamic scenes,
whereas standard 4DGS methods often require 20K to 30K
iterations to achieve comparable visual quality.
Additionally, we eliminate opacity resets during train-
ing, a technique commonly used in 3D Gaussian splatting
piplines to remove floaters in static scenes. While effective
for static reconstructions, we found that periodic opacity
reinitialization disrupts joint spatial-temporal optimization
in dynamic scenes, particularly when training time is lim-
ited. Instead, we opt for a straightforward continuous opti-
mization in which both static and dynamic Gaussians retain
their opacities throughout the training procedure, achiev-
ing more stable convergence. Furthermore, since our hy-
brid model inherently reduces the number of Gaussians, it
mitigates opacity saturation issues without requiring resets,
unlike standard static scene reconstruction methods.
Finally, we integrate both 3D and 4D Gaussians into a
unified CUDA rasterization pipeline. Our method builds
upon the original 3DGS implementation [18], extending it
to support 4D Gaussians at arbitrary timestamps alongside
static ones. As illustrated in Fig. 3, each 4D Gaussian is
sliced at time tto generate a transient 3D Gaussian with
mean µxyz|tand covariance Σxyz|t. We then aggregate all
Gaussians (both 3D and 4D) into a single list, project them
into screen space, assign tile and depth keys, and sort them
5

---

## Page 6

Table 1. Quantitative comparison on the N3V dataset [25], with PSNR as the primary evaluation metric. The best and second-best results
are highlighted in bold and underlined, respectively. For training time, (*): measured on our machine equipped with an RTX 4090 GPU, †:
from Lee et al. [22], and other numbers are adopted from the original papers.
Methodcoffee
martinicook
spinachcutroasted
beefflame
salmonflame
steaksear
steakAverage Training Time FPS Storage
HyperReel [1] 28.37 32.3 32.92 28.26 32.2 32.57 31.1 9 h†2 360 MB
NeRFPlayer [49] 31.53 30.56 29.35 31.65 31.93 29.13 30.69 6 h 0.05 5.1 GB
K-Planes [15] 29.99 32.6 31.82 30.44 32.38 32.52 31.63 1.8 h 0.3 311 MB
MixV oxel-L [54] 29.63 32.25 32.4 29.81 31.83 32.1 31.34 1.3 h 38 500 MB
4DGS [60] 28.33 32.93 33.85 29.38 34.03 33.51 32.01 (5.5 h)∗114 2.1 GB
4DGaussian [56] 27.93 32.87 30.96 29.33 32.84 32.44 31.06 (30 m )∗137 34 MB
STG [26] 28.61 33.18 33.52 29.48 33.64 33.89 32.05 1.3 h†140 200 MB
4D-RotorGS [11] 28.6 32.9 31.39 28.82 32.9 32.65 31.21 1 h 277 144 MB
Ex4DGS [22] 28.79 33.23 33.73 29.29 33.91 33.69 32.11 36 m (1 h 8 m)∗121 115 MB
Ours 28.86 33.3 33.73 29.38 33.79 34.45 32.25 (11 m 53 s )∗208 273 MB
for back-to-front alpha compositing. By rendering both
types of Gaussians in a single pass, our approach maintains
the efficiency of 3D splatting while preserving the flexibil-
ity of 4D temporal modeling.
5. Experiments
5.1. Datasets
Neural 3D Video (N3V). We evaluate our method on the
N3V dataset [25], which comprises six multi-view video se-
quences captured using 18-21 cameras at a native resolution
of2704×2028 . Five sequences last 10 seconds each, while
one sequence spans 40 seconds. In most experiments, we
follow standard practice by using 10-second segments for
fair comparisons, specifically extracting a 10-second clip
from the 40-second video ( flame salmon ). In line with
prior work, we hold out cam00 as the test camera for each
scene and use the remaining cameras for training. Addi-
tionally, we experiment with the full 40-second sequence to
demonstrate the scalability and robustness of our method on
longer dynamic content. For all experiments, we downsam-
ple the videos by a factor of two (both training and evalua-
tion), following the protocol used in previous works.
Technicolor. We also evaluate our method on a subset of
the Technicolor dataset [47], which comprises video record-
ings from a 4×4camera array (16 cameras) at a resolution
of 2048×1088. Following the common practice, we select
five scenes ( Birthday ,Fabien ,Painter ,Theater ,
Trains ), each limited to 50 frames. We keep the original
resolution and designate cam10 as the held-out test view,
using the remaining cameras for training.
5.2. Implementation Details
Following Yang et al. [60], we initialize our 4D Gaussian
representation using dense COLMAP reconstructions forTable 2. Quantitative comparison on the 40-second sequence. The
best and second-best results are highlighted in bold and under-
lined, respectively. All metric scores are taken from Xu et al. [58].
‡: Initializes point clouds using sparse COLMAP from each frame,
**: split all 300 frames for training.
Method PSNR ↑SSIM↑LPIPS ↓Training Time VRAM FPS Storage
ENeRF [27] 23.48 0.8944 0.2599 4.6 h 23 GB 5 0.83 GB
4K4D∗∗[57] 21.29 0.8266 0.3715 26.6 h 84 GB 290 2.46 GB
Dy3DGS [32] 25.91 0.8809 0.2555 37.1 h 5 GB 610 19.5 GB
4DGS∗∗[60] 28.89 0.9521 0.1968 10.4 h 84 GB 90 2.68 GB
Xuet al.‡[58] 29.44 0.945 0.2144 2.1 h 6.1 GB 550 0.09 GB
Ours 29.2 0.9175 0.1173 52 m 12 GB 111 0.96 GB
Table 3. Quantitative results on the Technicolor dataset. Training
times (including COLMAP) are measured on the Painter scene
with an RTX 3090 GPU. For training time, ( *): measured on our
machine, †: from Bae et al. [2],‡: uses sparse COLMAP initial-
ization.
Method PSNR ↑SSIM↑LPIPS ↓Training Time Storage
HyperReel [1] 33.32 0.899 0.118 2 h 45 m†289 MB
4DGaussian [56] 29.62 0.844 0.176 32 m†51 MB
E-D3DGS [2] 33.24 0.907 0.100 3 h 02 m†77 MB
4DGS [60] 33.35 0.910 0.095 (4 h 20 m)∗1.07 GB
Ex4DGS‡[22] 33.62 0.9156 0.088 (1 h 5 m)∗88 MB
Ours‡33.22 0.911 0.149 ( 29 m )∗218 MB
the N3V dataset (about 300k points), providing robust ge-
ometric priors. For Technicolor, which has only 50 frames
per scene, we start from a sparse COLMAP reconstruction
instead. We adopt the densification pipeline from 3D Gaus-
sian Splatting [18], progressively increasing the number of
Gaussians by cloning and splitting operations. Unlike prior
works, however, we do not perform periodic opacity resets
during training. For automatic classification of Gaussians,
we set the temporal scale threshold τto 3 for the 10-second
6

---

## Page 7

Figure 4. Qualitative comparison on the N3V dataset. While most methods yield comparable results, our approach can preserve subtle
motion cues and slightly more consistent colors in some challenging regions. Zoom in for best viewing.
N3V sequences and 6 for the 40-second sequence, while
using a threshold of 1 for Technicolor. We train the 10-
second N3V clips for 6,000 iterations (batch size 4) and the
40-second clip for 20,000 iterations, applying the adaptive
densification up to 15,000 iterations. For Technicolor, each
scene is trained for 10,000 iterations with a batch size of 2.
Our implementation is built on the codebase of Yang et al.
[60] and further leverages the efficient backward pass from
Taming-3DGS [33] to accelerate optimization.
5.3. Results
5.3.1. Quantitative Results
N3V Dataset. We first evaluate our approach on the N3V
dataset, with results summarized in Tab. 1. Our method
achieves competitive performance across all scenes, with an
average PSNR of 32.25 dB, outperforming recent methods
in both fidelity and rendering speed. Notably, we require
only 12 minutes of training time for the 10-second clips,
which is significantly faster than 4DGS [60] (5.5 hours),
while providing comparable or superior visual quality. The
combination of fast optimization, high FPS (208), and mod-
erate storage (273 MB) underscores the effectiveness of our
hybrid 3D–4D Gaussian representation.
Long Sequence (40 seconds). Tab. 2 presents the results
on the challenging 40-second clip from the N3V dataset.
Our method achieves the second-best PSNR (29.2 dB) and
the lowest LPIPS (0.1173), demonstrating strong percep-
tual quality. Remarkably, we complete training in onlyTable 4. Ablation study on the N3V dataset, comparing the 4DGS
baseline, our approach (Ours), the effect of opacity resets (w/ opa
reset), and different temporal scale thresholds ( τ). #4D and #3D
denote the number of 4D and 3D Gaussians, respectively.
Method PSNR SSIM LPIPS #4D #3D
4DGS [60] 32.01 0.9453 0.0974 3,315,333 –
Ours 32.25 0.9459 0.0970 843,175 229,707
w/ opa reset 31.52 0.9418 0.1016 683,437 243,051
τ= 2.5 31.37 0.9440 0.0979 670,807 276,265
τ= 3.5 31.98 0.9450 0.0986 913,927 184,548
52 minutes, an order of magnitude faster than other meth-
ods. Although Xu et al. [58] reports a slightly higher PSNR
(29.44 dB) by initializing point clouds from every frame
(sparse COLMAP for each frame takes approximately 1
second, additional 20 minutes for 1,200 frames to their re-
ported training time 2.1 hours), our approach relies solely
on the single-frame initialization used for 10-second exper-
iments. Despite this simpler setup, our method provides
a more balanced trade-off in terms of training speed, stor-
age, and inference performance, highlighting its scalability
to longer sequences.
Technicolor Dataset. We further validate our method on
the Technicolor dataset (Tab. 3). Despite using a sparse
COLMAP initialization for the 50-frame sequences, our
model achieves 33.22 dB PSNR and 0.911 SSIM, with only
7

---

## Page 8

Figure 5. Visual comparison of different scale thresholds τ.
29 minutes of training time on an RTX 3090 (measured on
thePainter scene). In contrast, 4DGS requires over four
hours to reach a comparable PSNR, and Ex4DGS—while
slightly more accurate—needs more than twice of our train-
ing time. Our final storage is 218 MB, which is lower than
4DGS (1.07 GB) but slightly higher than some other meth-
ods. Overall, these results confirm that our framework ef-
fectively handles diverse camera setups and short videos,
balancing speed, memory efficiency, and rendering fidelity.
5.3.2. Qualitative Results
Fig. 4 compares our method with several baselines on the
N3V dataset. Overall, the visual quality among these meth-
ods is largely similar, reflecting the challenging nature of
dynamic scenes. However, our hybrid representation shows
sharper details in some dynamic regions and more consis-
tent color transitions in backgrounds, reducing minor flick-
ers across frames. These observations align with our quan-
titative findings, suggesting that our approach remains com-
petitive for complex, real-world scenarios.
5.4. Ablation Studies and Analysis
Scale Threshold τ.We investigate how varying the tem-
poral scale threshold τaffects both reconstruction quality
and storage (see Tab. 4). As shown in Fig. 5, a lower thresh-
old (e.g., τ= 2.5) aggressively converts 4D Gaussians
into 3D, which can inadvertently merge dynamic content
into the static representation, reducing motion detail despite
simplifying the final geometry. Conversely, a higher thresh-
old (τ= 3.5) is more lenient about switching Gaussians to
3D, preserving subtle dynamics at the cost of slower con-
Figure 6. Influence of opacity resets on a dynamic scene.
Figure 7. Visualization of spatially distributed Gaussians.
vergence and higher memory usage. The mid-range setting
(τ= 3.0) strikes a balanced trade-off, maintaining near-
optimal quality while avoiding excessive storage overhead.
Opacity Reset. Many 3D/4D Gaussian methods period-
ically reinitialize opacities to a small constant to remove
floaters or spurious elements [18, 60]. However, such resets
are heuristic and can inadvertently disrupt optimization in
dynamic regions. As shown in Tab. 4 and Fig. 6, forcibly
lowering the opacities of both 3D and 4D Gaussians can
erase previously learned motion cues, leading to flicker or
lower final PSNR. By avoiding opacity resets, our pipeline
continuously refines all Gaussians in a single pass, preserv-
ing subtle temporal details and stabilizing motion bound-
aries. This simpler, reset-free approach also reduces hyper-
parameter tuning overhead and prevents abrupt representa-
tion changes that might otherwise degrade performance.
Visualization of spatially distributed Gaussians Fig. 7
visualizes the spatially distributed Gaussians, comparing
our model to 4DGS [60]. To visualize, we first project
8

---

## Page 9

all 3D and 4D Gaussians (for 4DGS, only 4D Gaussians)
on the image plane given a specific view point. Then, we
color-coded based on the number of projected Gaussians in
each spatial location (the darker color, the more Gaussians).
This shows how each approach allocates Gaussians differ-
ently to different spatial regions, and the original 4DGS in-
troduces many Gaussians in static areas (highlighted as red
boxes), implying that numerous 4D Gaussians with small
time scales are used to represent static parts of the scene. On
the other hand, our approach uses 3D Gaussians for static
areas, resulting in evenly distributed Gaussians across the
scene. This result supports our experimental results that our
method significantly reduces redundancy, lowers memory
usage, and accelerates optimization. By contrast, the base-
line model places dense clusters of Gaussians in static re-
gions, leading to unnecessary computations, inflating mem-
ory costs, and often degrading the rendering quality.
6. Conclusion
We have presented a novel hybrid 3D-4D Gaussian Splat-
ting framework for dynamic scene reconstruction. By dis-
tinguishing static regions and selectively assigning 4D pa-
rameters only to dynamic elements, our method substan-
tially reduces redundancy while preserving high-fidelity
motion cues. Extensive experiments on the N3V and Tech-
nicolor datasets demonstrate that our approach consistently
achieves competitive or superior quality and faster training
compared to state-of-the-art baselines.
Limitations First, our heuristic scale thresholding could
be refined, potentially using learning-based or data-driven
methods. Second, a specialized 4D densification strat-
egy could further reduce redundancy and optimize mem-
ory usage, building on recent successes in 3DGS densifica-
tion [12, 19, 46]. Such an approach may lead to even higher
reconstruction quality and more efficient training.
References
[1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael
Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil
Kim. Hyperreel: High-fidelity 6-dof video with ray-
conditioned sampling. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 16610–16620, 2023.
[2] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee,
Gun Bang, and Youngjung Uh. Per-gaussian embedding-
based deformation for deformable 3d gaussian splatting. In
European Conference on Computer Vision , pages 321–335.
Springer, 2024.
[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. In Proceedings of the IEEE/CVF inter-national conference on computer vision , pages 5855–5864,
2021.
[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 5470–5479, 2022.
[5] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased
grid-based neural radiance fields. In Proceedings of the
IEEE/CVF International Conference on Computer Vision ,
pages 19697–19705, 2023.
[6] Robert A Brebin, Loren Carpenter, and Pat Hanrahan. V ol-
ume rendering. In Seminal graphics: pioneering efforts that
shaped the field , pages 363–372. ACM, 1998.
[7] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 130–141, 2023.
[8] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance fields. In European con-
ference on computer vision , pages 333–350. Springer, 2022.
[9] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai. Hac: Hash-grid assisted context for 3d
gaussian splatting compression. In European Conference on
Computer Vision , pages 422–438. Springer, 2024.
[10] Woong Oh Cho, In Cho, Seoha Kim, Jeongmin Bae,
Youngjung Uh, and Seon Joo Kim. 4d scaffold gaussian
splatting for memory efficient dynamic scene reconstruction.
arXiv preprint arXiv:2411.17044 , 2024.
[11] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:
towards efficient novel view synthesis for dynamic scenes.
InACM SIGGRAPH 2024 Conference Papers , pages 1–11,
2024.
[12] Guangchi Fang and Bing Wang. Mini-splatting2: Building
360 scenes within minutes via aggressive gaussian densifica-
tion. arXiv preprint arXiv:2411.12788 , 2024.
[13] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance fields with time-aware neural vox-
els. In SIGGRAPH Asia 2022 Conference Papers , pages 1–9,
2022.
[14] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 5501–5510, 2022.
[15] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 12479–12488, 2023.
[16] Qiankun Gao, Jiarui Meng, Chengxiang Wen, Jie Chen,
and Jian Zhang. Hicom: Hierarchical coherent motion
for dynamic streamable scenes with 3d gaussian splatting.
Advances in Neural Information Processing Systems , 37:
80609–80633, 2025.
9

---

## Page 10

[17] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu,
Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled
gaussian splatting for editable dynamic scenes. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition , pages 4220–4230, 2024.
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph. , 42(4):139–1,
2023.
[19] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar,
Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splat-
ting as markov chain monte carlo. Advances in Neural Infor-
mation Processing Systems , 37:80965–80986, 2024.
[20] Seoha Kim, Jeongmin Bae, Youngsik Yun, Hahyun Lee,
Gun Bang, and Youngjung Uh. Sync-nerf: Generalizing dy-
namic nerfs to unsynchronized videos. In Proceedings of
the AAAI Conference on Artificial Intelligence , pages 2777–
2785, 2024.
[21] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis.
Dynmf: Neural motion factorization for real-time dynamic
view synthesis with 3d gaussian splatting. In European Con-
ference on Computer Vision , pages 252–269. Springer, 2024.
[22] Junoh Lee, Changyeon Won, Hyunjun Jung, Inhwan Bae,
and Hae-Gon Jeon. Fully explicit dynamic gaussian splat-
ting. Advances in Neural Information Processing Systems ,
37:5384–5409, 2025.
[23] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park. Compact 3d gaussian representation for
radiance field. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 21719–
21728, 2024.
[24] Soonbin Lee, Fangwen Shu, Yago Sanchez, Thomas Schierl,
and Cornelius Hellge. Compression of 3d gaussian splatting
with optimized feature planes and standard video codecs.
arXiv preprint arXiv:2501.03399 , 2025.
[25] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition , pages 5521–5531, 2022.
[26] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 8508–8520, 2024.
[27] Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai,
Hujun Bao, and Xiaowei Zhou. Efficient neural radiance
fields for interactive free-viewpoint video. In SIGGRAPH
Asia Conference Proceedings , 2022.
[28] Zhening Liu, Yingdong Hu, Xinjie Zhang, Jiawei Shao, Ze-
hong Lin, and Jun Zhang. Dynamics-aware gaussian splat-
ting streaming towards fast on-the-fly training for 4d recon-
struction. arXiv preprint arXiv:2411.14847 , 2024.
[29] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel
Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural vol-
umes: Learning dynamic renderable volumes from images.
arXiv preprint arXiv:1906.07751 , 2019.[30] Jiahao Lu, Jiacheng Deng, Ruijie Zhu, Yanzhe Liang, Wenfei
Yang, Tianzhu Zhang, and Xu Zhou. Dn-4dgs: Denoised de-
formable network with temporal-spatial aggregation for dy-
namic scene rendering. arXiv preprint arXiv:2410.13607 ,
2024.
[31] Tao Lu, Ankit Dhiman, R Srinath, Emre Arslan, Angela
Xing, Yuanbo Xiangli, R Venkatesh Babu, and Srinath Srid-
har. Turbo-gs: Accelerating 3d gaussian fitting for high-
quality radiance fields. arXiv preprint arXiv:2412.13547 ,
2024.
[32] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 3DV, 2024.
[33] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre. Taming 3dgs: High-quality radiance
fields with limited resources. In SIGGRAPH Asia 2024 Con-
ference Papers , pages 1–11, 2024.
[34] Marko Mihajlovic, Sergey Prokudin, Marc Pollefeys, and
Siyu Tang. Resfields: Residual neural fields for spatiotem-
poral signals. arXiv preprint arXiv:2309.03160 , 2023.
[35] Marko Mihajlovic, Sergey Prokudin, Siyu Tang, Robert
Maier, Federica Bogo, Tony Tung, and Edmond Boyer.
Splatfields: Neural gaussian splats for sparse 3d and 4d re-
construction. In European Conference on Computer Vision ,
pages 313–332. Springer, 2024.
[36] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM , 65(1):99–106, 2021.
[37] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and
Peter Eisert. Compact 3d scene representation via self-
organizing gaussian grids. In European Conference on Com-
puter Vision , pages 18–34. Springer, 2024.
[38] Thomas M ¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG) , 41(4):1–15, 2022.
[39] Seungtae Nam, Daniel Rho, Jong Hwan Ko, and Eunbyung
Park. Mip-grid: Anti-aliased grid representations for neural
radiance fields. Advances in Neural Information Processing
Systems , 36:2837–2849, 2023.
[40] K Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi
Koohpayegani, and Hamed Pirsiavash. Compact3d: Com-
pressing gaussian splat radiance field models with vector
quantization. arXiv preprint arXiv:2311.18159 , 4, 2023.
[41] Simon Niedermayr, Josef Stumpfegger, and R ¨udiger West-
ermann. Compressed 3d gaussian splatting for accelerated
novel view synthesis. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
10349–10358, 2024.
[42] Panagiotis Papantonakis, Georgios Kopanas, Bernhard
Kerbl, Alexandre Lanvin, and George Drettakis. Reducing
the memory footprint of 3d gaussian splatting. Proceedings
of the ACM on Computer Graphics and Interactive Tech-
niques , 7(1):1–17, 2024.
[43] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
10

---

## Page 11

Martin-Brualla. Nerfies: Deformable neural radiance fields.
InProceedings of the IEEE/CVF international conference on
computer vision , pages 5865–5874, 2021.
[44] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz. Hypernerf: A higher-
dimensional representation for topologically varying neural
radiance fields. arXiv preprint arXiv:2106.13228 , 2021.
[45] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer. D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition , pages
10318–10327, 2021.
[46] Samuel Rota Bul `o, Lorenzo Porzi, and Peter Kontschieder.
Revising densification in gaussian splatting. In European
Conference on Computer Vision , pages 347–362. Springer,
2024.
[47] Neus Sabater, Guillaume Boisson, Benoit Vandame, Paul
Kerbiriou, Frederic Babon, Matthieu Hog, Remy Gendrot,
Tristan Langlois, Olivier Bureller, Arno Schubert, et al.
Dataset and pipeline for multi-view light-field video. In Pro-
ceedings of the IEEE conference on computer vision and pat-
tern recognition Workshops , pages 30–40, 2017.
[48] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu,
Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural
4d decomposition for high-fidelity dynamic reconstruction
and rendering. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 16632–
16642, 2023.
[49] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele
Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerf-
player: A streamable dynamic scene representation with de-
composed neural radiance fields. IEEE Transactions on Visu-
alization and Computer Graphics , 29(5):2732–2742, 2023.
[50] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel
grid optimization: Super-fast convergence for radiance fields
reconstruction. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition , pages 5459–
5469, 2022.
[51] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Improved
direct voxel grid optimization for radiance fields reconstruc-
tion. arXiv preprint arXiv:2206.05085 , 2022.
[52] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei
Zhao, and Wei Xing. 3dgstream: On-the-fly training of
3d gaussians for efficient streaming of photo-realistic free-
viewpoint videos. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
20675–20685, 2024.
[53] Chengbo Wang, Guozheng Ma, Yifei Xue, and Yizhen Lao.
Faster and better 3d splatting via group training. arXiv
preprint arXiv:2412.07608 , 2024.
[54] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei
Song, and Huaping Liu. Mixed neural voxels for fast multi-
view video synthesis. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision , pages 19706–
19716, 2023.
[55] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jia-
jun Deng, Jiang Bian, and Zhibo Chen. End-to-end rate-distortion optimized 3d gaussian representation. In European
Conference on Computer Vision , pages 76–92. Springer,
2024.
[56] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
InProceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition , pages 20310–20320, 2024.
[57] Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming
Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou. 4k4d:
Real-time 4d view synthesis at 4k resolution. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition , pages 20029–20040, 2024.
[58] Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun,
Hujun Bao, and Xiaowei Zhou. Representing long volumet-
ric video with temporal gaussian hierarchy. ACM Transac-
tions on Graphics (TOG) , 43(6):1–18, 2024.
[59] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for
high-fidelity monocular dynamic scene reconstruction. arXiv
preprint arXiv:2309.13101 , 2023.
[60] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang.
Real-time photorealistic dynamic scene representation and
rendering with 4d gaussian splatting. arXiv preprint
arXiv:2310.10642 , 2023.
[61] Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng,
Jiahao Lu, Wenfei Yang, Tianzhu Zhang, and Yongdong
Zhang. Motiongs: Exploring explicit motion guidance
for deformable 3d gaussian splatting. arXiv preprint
arXiv:2410.07707 , 2024.
11

---

## Page 12

Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation
Supplementary Material
Algorithm 1 GPU Rasterization of 3D&4D Gaussians
Require: w, h: image dimensions
Require: M4D, S4D: 4D Gaussian means and covariances
Require: M3D, S3D: 3D Gaussian means and covariances
Require: A: 3D/4D Gaussian attributes
Require: V: camera/view configuration
Require: s: time
1:function RASTERIZE (w, h, M 4D, S4D, M3D, S3D, A, V, s )
2: CullGaussian( p, V)
3: (M′, S′
4D)←ProjGaussian4D (M4d, S4d, V, s)
4: iflen(M3D)>0then
5: (M′, S′
3d)←ProjGaussian3D (M′, M3d, S3d, V)
6: end if
7: T←CreateTiles (w, h)
8: (L, K)←DuplicateWithKeys (M′, T)
9: SortByKeys( K, L )
10: R←IdentifyTileRanges (T, K)
11: I←0
12: for all Tilest∈Ido
13: for all pixels i∈tdo
14: r←GetTileRange(R,t)
15: I[i]←BlendInOrder (i, L, r, K, M′, S′
4D, S′
3D, A)
16: end for
17: end for
18: return I
19:end function
A. CUDA Rasterization Pipeline
Compared to the original pipeline in the 3DGS [18], lines
4–6 are newly introduced to seamlessly integrate static (3D)
Gaussians with dynamic (4D) Gaussians. In particular, the
size of M′is allocated to accommodate both 3D and 4D
points. The conditional check at line 4 verifies whether any
3D Gaussians exist; if so, it projects them into screen space
viaProjGaussian3D , and stores tile, depth, and screen-
space position data jointly with the 4D Gaussians.
B. Additional Results
In this section, we provide further quantitative and qualita-
tive evaluations to supplement our main paper.
B.1. SSIM and LPIPS Comparisons
We present additional metrics on SSIM and LPIPS for the
N3V dataset. As summarized in Table 5, our method consis-
tently maintains strong perceptual quality across these met-
rics, corroborating the PSNR improvements reported in the
main text. In particular, our SSIM and LPIPS scores remain
on par with, or exceed, those of baseline methods, indicat-
ing sharper details and fewer artifacts in dynamic regions.Table 5. Additional SSIM and LPIPS results on the N3V dataset.
Higher SSIM and lower LPIPS indicate better perceptual quality.
Method SSIM ↑LPIPS ↓
HyperReel [1] 0.927 0.096
NeRFPlayer [49] 0.931 0.111
K-Planes [15] 0.947 0.090
MixV oxel-L [54] 0.933 0.095
4DGS [60] 0.9453 0.0974
STG [26] 0.948 0.046
4DGaussian [56] 0.935 0.074
4D-RotorGS [11] 0.939 0.106
Ex4DGS [22] 0.940 0.048
Ours 0.9459 0.097
B.2. Per-Scene Graphs on N3V
Figure 8 shows the per-scene PSNR curves over train-
ing iterations for three different scale thresholds. While
τ= 2.5can converge quickly in the early iterations,
it sometimes saturates at a slightly lower peak PSNR
(e.g., cook spinach ) or collapse after few iteration(e.g.
flame steak ), possibly merging subtle dynamics into
static representation. In contrast, τ= 3.5tends to re-
tain more 4D Gaussians longer, occasionally surpassing
τ= 2.5in later stages (e.g., sear steak ), but it also
requires more training to reach its final quality. The mid-
range threshold ( τ= 3.0) typically offers a balanced trade-
off between these extremes, achieving stable and competi-
tive performance across scenes with moderate or complex
motion.
B.3. Additional Qualitative Results
Finally, we present further visual comparisons, highlight-
ing subtle differences in dynamic objects, complex light-
ing, and motion boundaries. Our hybrid 3D–4D representa-
tion consistently captures both static and moving elements
with minimal artifacts, reinforcing the quantitative gains re-
ported in the main paper.
Long-Sequence Comparison. In Fig. 9, we compare our
reconstructions to ground-truth frames from the 40-second
N3V sequence. Despite the longer duration and more com-
plex motion, our method maintains coherent geometry and
color transitions, demonstrating robust performance for ex-
tended temporal dynamics without significant artifacts.
12

---

## Page 13

Multi-Dataset Visuals. Fig. 10 showcases additional re-
sults on both N3V and Technicolor scenes. We observe that
our method preserves fine-grained details under challeng-
ing lighting conditions, while effectively modeling diverse
motion patterns. These qualitative improvements align with
our quantitative gains in PSNR and SSIM.
Dynamic and Static Visuals. In Fig. 11, we visualize dy-
namic and static Gaussians side by side, with dynamic re-
gions rendered on a white background to highlight the sep-
aration from static areas. Our method adaptively assigns
4D Gaussians to genuinely moving objects while converting
large, motionless regions to 3D Gaussians. This selective
allocation preserves subtle motion cues, reduces memory
overhead, and accelerates the optimization process. The fi-
nal rendered results confirm that our representation remains
faithful to the original scenes, even under challenging light-
ing and motion conditions.
13

---

## Page 14

Figure 8. Per-scene PSNR curves on the N3V dataset for different temporal scale thresholds ( τ= 2.5,3.0,3.5). Each plot corresponds
to one scene, showing how PSNR evolves over 6000 iterations of training. The mid-range setting ( τ= 3.0) often strikes a balance,
maintaining competitive final quality across a range of motion complexities.
Figure 9. Comparison with Ground Truth on the 40-second sequence. We sample frames at different timestamps (top: GT, bottom:
ours) to illustrate that our approach preserves both global structure and subtle motion details over extended temporal ranges.
14

---

## Page 15

Figure 10. Additional results on N3V and Technicolor scenes. Despite challenging lighting conditions and fast motion, our hybrid
3D-4D approach maintains crisp object boundaries and more consistent textures across frames.
15

---

## Page 16

Figure 11. Dynamic vs. Static Visualization. Each row shows (left) the dynamic portion on a white background, (middle) the static
region, and (right) the fully rendered result. By converting most static elements into 3D Gaussians, our approach effectively handles
dynamic scenes while reducing redundant computations and preserving high-fidelity details.
16