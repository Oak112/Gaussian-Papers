

---

## Page 1

HexPlane: A Fast Representation for Dynamic Scenes
Ang Cao Justin Johnson
University of Michigan, Ann Arbor
fancao, justincj g@umich.edu
Abstract
Modeling and re-rendering dynamic 3D scenes is a chal-
lenging task in 3D vision. Prior approaches build on NeRF
and rely on implicit representations. This is slow since it re-
quires many MLP evaluations, constraining real-world ap-
plications. We show that dynamic 3D scenes can be ex-
plicitly represented by six planes of learned features, lead-
ing to an elegant solution we call HexPlane. A HexPlane
computes features for points in spacetime by fusing vec-
tors extracted from each plane, which is highly efﬁcient.
Pairing a HexPlane with a tiny MLP to regress output col-
ors and training via volume rendering gives impressive re-
sults for novel view synthesis on dynamic scenes, match-
ing the image quality of prior work but reducing training
time by more than 100. Extensive ablations conﬁrm our
HexPlane design and show that it is robust to different fea-
ture fusion mechanisms, coordinate systems, and decoding
mechanisms. HexPlane is a simple and effective solution
for representing 4D volumes, and we hope they can broadly
contribute to modeling spacetime for dynamic 3D scenes.1
1. Introduction
Reconstructing and re-rendering 3D scenes from a set
of 2D images is a core vision problem which can enable
many AR/VR applications. The last few years have seen
tremendous progress in reconstructing static scenes, but this
assumption is restrictive: the real world is dynamic , and in
complex scenes motion is the norm, not the exception.
Many current approaches for representing dynamic
3D scenes rely on implicit representations, building on
NeRF [46]. They train a large multi-layer perceptron (MLP)
that inputs the position of a point in space and time, and out-
puts either the color of the point [31, 32] or a deformation
to a canonical static scene [17, 53, 54, 58]. In either case,
rendering images from novel views is expensive since each
generated pixel requires many MLP evaluations. Training is
1Project page: https://caoang327.github.io/HexPlane .
Figure 1. HexPlane for Dynamic 3D Scenes. Instead of regress-
ing colors and opacities from a deep MLP, we explicitly compute
features for points in spacetime via HexPlane. Pairing with a tiny
MLP, it allows above 100speedups with matching quality.
similarly slow, requiring up to days of GPU time to model
a single dynamic scene; this computational bottleneck pre-
vents these methods from being widely applied.
Several recent methods for modeling static scenes have
demonstrated tremendous speedups over NeRF through the
use of explicit andhybrid methods [7, 47, 71, 86]. These
methods use an explicit spatial data structure that stores ex-
plicit scene data [15, 86] or features that are decoded by a
tiny MLP [7, 47, 71]. This decouples a model’s capacity
from its speed , and allows high-quality images to be ren-
dered in realtime [47]. While effective, these methods have
thus far been applied only to static scenes.
In this paper, we aim to design an explicit representa-
tion of dynamic 3D scenes, building on similar advances
for static scenes. To this end, we design a spatial-temporal
data structure that stores scene data. It must overcome two
key technical challenges. First is memory usage . We must
model all points in both space and time; na ¨ıvely storing data
in a dense 4D grid would scale with the fourth power of
grid resolution which is infeasible for large scenes or long
durations. Second is sparse observations . Moving a single
camera through a static scene can give views that densely
cover the scene; in contrast, moving a camera through a
dynamic scene gives just one view per timestep. Treating
timesteps independently may give insufﬁcient scene cov-
erage for high-quality reconstruction, so we must instead
share information across timesteps.arXiv:2301.09632v2  [cs.CV]  27 Mar 2023

---

## Page 2

We overcome these challenges with our novel HexPlane
architecture. Inspired by factored representations for static
scenes [5, 7, 55], a HexPlane decomposes a 4D spacetime
grid into six feature planes spanning each pair of coordinate
axes ( e.g.XY,ZT). A HexPlane computes a feature vector
for a 4D point in spacetime by projecting the point onto
each feature plane, then aggregating the six resulting feature
vectors. The fused feature vector is then passed to a tiny
MLP which predicts the color of the point; novel views can
then be rendered via volume rendering [46].
Despite its simplicity, a HexPlane provides an elegant
solution to the challenges identiﬁed above. Due to its fac-
tored representation, a HexPlane’s memory footprint only
scales quadratically with scene resolution. Furthermore,
each plane’s resolution can be tuned independently to ac-
count for scenes requiring variable capacity in space and
time. Since some planes rely only on spatial coordinates
(e.g.XY), by construction a HexPlane encourages sharing
information across disjoint timesteps.
Our experiments demonstrate that HexPlane is an effec-
tive and highly efﬁcient method for novel view synthesis
in dynamic scenes. On the challenging Plenoptic Video
dataset [31] we match the image quality of prior work but
improve training time by >100; we also outperform prior
approaches on a monocular video dataset [58]. Extensive
ablations validate our HexPlane design and demonstrate
that it is robust to different feature fusion mechanisms, co-
ordinate systems (rectangular vs. spherical), and decoding
mechanisms (spherical harmonics vs. MLP).
HexPlane is a simple, explicit, and general representa-
tion for dynamic scenes. It makes minimal assumptions
about the underlying scene, and does not rely on deforma-
tion ﬁelds or category-speciﬁc priors. Besides improving
and accelerating view synthesis, we hope HexPlane will be
useful for a broad range of research in dynamic scenes [66].
2. Related Work
Neural Scene Representations. Using neural networks to
implicitly represent 3D scenes [43, 50, 67, 68, 72, 80] has
achieved exciting progress recently. NeRF [46] and its vari-
ants [2, 3, 44, 48, 74, 76, 85, 92] show impressive results on
novel view synthesis [9,80,87,99] and many other applica-
tions including 3D reconstruction [42, 72, 90, 94, 100], se-
mantic segmentation [27,59,98], generative model [5,6,10,
49,62,82], and 3D content creation [1,23,33,52,57,77,91].
Implicit neural representations exhibit remarkable ren-
dering quality, but they suffer from slow rendering speeds
due to the numerous costly MLP evaluations required for
each pixel. To address this challenge, many recent papers
propose hybrid representations that combine a fast explicit
scene representation with learnable neural network compo-
nents, providing signiﬁcant speedups over purely implicit
methods. Various explicit representations have been inves-tigated, including sparse voxels [15, 37, 63, 71], low-rank
components [5, 7, 34, 55], point clouds [4, 22, 84, 97, 101]
and others [8,39,47,73,95]. However, these approaches as-
sume static 3D scenes, leaving explicit representations for
dynamic scenes unexplored. This paper provides an explicit
model for dynamic scenes, substantially accelerating prior
methods that rely on fully implicit methods.
Neural Rendering for Dynamic Scenes. Representing dy-
namic scenes by neural radiance ﬁelds is an essential ex-
tension of NeRF, enabling numerous real-world applica-
tions [30, 51, 56, 70, 83, 89, 96]. One line of research repre-
sents dynamic scenes by extending NeRF with an additional
time dimension (T-NeRF) or additional latent code [17,
31, 32, 81]. Despite the ability to represent general typol-
ogy changes, they suffer from a severely under-constrained
problem, requiring additional supervision like depths, opti-
cal ﬂows or dense observations for decent results. Another
line of research employs individual MLPs to represent a de-
formation ﬁeld and a canonical ﬁeld [11, 53, 54, 58, 75, 88],
where the canonical ﬁeld depicts a static scene, and the
deformation ﬁeld learns coordinate maps to the canonical
space over time. We propose a simple yet elegant solution
for dynamic scene representation using six feature planes,
making minimal assumptions about the underlying scene.
Recently, MAV3D [66] adopted our design for text-to-4D
dynamic scene generation, demonstrating an exciting direc-
tion for dynamic scenes beyond reconstruction.
Accelerating NeRFs. Many works have been proposed to
accelerate NeRF at diverse stages. Some methods improve
inference speeds of trained NeRFs by optimizing the com-
putation [19,21,60,86]. Others reduce the training times by
learning a generalizable model [9,25,79,80]. Recently, ren-
dering speeds during both stages are substantially reduced
by using explicit-implicit representations [5, 7, 15, 36, 47,
71]. In line with this idea, we propose an explicit represen-
tation for dynamic ﬁelds to accelerate dynamic NeRFs.
Very recently, several concurrent works have aimed to
accelerate dynamic NeRFs. [13, 16, 20, 35, 78] use time-
aware MLPs to regress spacetime points’ colors or defor-
mations from canonical spaces. However, they remain par-
tially implicit for dynamic ﬁelds, as they rely on MLPs with
time input to obtain spacetime features. In contrast, our pa-
per proposes a more elegant and efﬁcient explicit represen-
tation for dynamic ﬁelds without using time-aware MLPs.
Like [29], NeRFPlayer [69] uses a highly compact 3D grid
at each time step for 4D ﬁeld representation , which results
in substantial memory costs for lengthy videos.
Tensor4D [64] shares a similar idea as ours, which repre-
sents dynamic scenes with 9 planes and multiple MLPs. D-
TensoRF [24] regards dynamic ﬁelds as 5D tensors and ap-
plies CP/MM decomposition on them for compact represen-
tation. Our paper is most closely related to K-Planes [14],
which also employs six feature planes for representation.

---

## Page 3

Figure 2. Method Overview. HexPlane contains six feature planes spanning each pair of coordinate axes (e.g. XY , ZT ). To compute
features of points in spacetime, it multiplies feature vectors extracted from paired planes and concatenated multiplied results into a single
vector, which are then multiplied by VRFfor ﬁnal results. RGB colors are regressed from point features using a tiny MLP and images are
synthesized via volumetric rendering. HexPlane and the MLP are trained by photometric loss between rendered and target images.
3. Method
Given a set of posed and timestamped images of a dy-
namic scene, we aim to ﬁt a model to the scene that al-
lows rendering new images at novel poses and times. Like
NeRF [46], a model gives color and opacity for points in
spacetime; images are rendered via differentiable volumet-
ric rendering along rays. The model is trained using photo-
metric loss between rendered and ground-truth images.
Our main contribution is a new explicit representation
for dynamic 3D scenes, which we combine with a small
implicit MLP to achieve novel view synthesis in dynamic
scenes. An input spacetime point is used to efﬁciently query
the explicit representation for a feature vector. A tiny MLP
receives the feature along with the point coordinates and
view direction and regresses an output RGB color for the
point. Figure 2 shows an overview of the model.
Designing an explicit representation for dynamic 3D
scenes is challenging. Unlike static 3D scenes which are
often modeled by point clouds, voxels, or meshes, ex-
plicit representations for dynamic scenes have been under-
explored. We show how the key technical challenges of
memory usage andsparse observations can be overcome by
our simple HexPlane representation.
3.1. 4D Volumes for Dynamic 3D Scenes
A dynamic 3D scene could be na ¨ıvely represented as a
4D volume Dcomprising independent static 3D volumes
per time stepfV1;V2;;VTg. However this design
suffers from two key problems. First is memory consump-
tion: a na ¨ıve 4D volume is very memory intensive, requiring
O(N3TF)space where N,T, andFare the spatial resolu-
tion, temporal resolution, and feature size. Storing a volume
of RGB colors ( F=3) withN=512 ,T=32 infloat32
format takes 48GB of memory.
The second problem is sparse observations . A singlecamera moving through a static scene can capture dozens or
hundreds of images. In dynamic scenes capturing multiple
images per timestep requires multiple cameras, so we typi-
cally have only a few views per timestep; these sparse views
are insufﬁcient for independently modeling each timestep,
so we must share information between timesteps.
We reduce memory consumption using factorization [5,
7] which has been previously applied to 3D volumes. We
build on TensoRF [7] which decomposes a 3D volume V2
RXYZF 1as a sum of vector-matrix outer products:
V=R1X
r=1MXY
rvZ
rv1
r+R2X
r=1MXZ
rvY
rv2
r
+R3X
r=1MZY
rvX
rv3
r(1)
whereis outer product; MXY
rvZ
rv1
ris a low-rank
component of V;MXY
r2RXYis a matrix spanning the
XandYaxes, and vZ2RZ;vi
r2RFare vectors along
theZandFaxes.R1;R2;R3are the number of low-rank
components. With R=R1+R2+R3N, this design
reduces memory usage from O(N3TF)toO(RN2T).
3.2. Linear Basis for 4D Volume
Factorization helps reduce memory usage, but factoring
an independent 3D volume per timestep still suffers from
sparse observations and does not share information across
time. To solve this problem, we can represent the 3D vol-
umeVtat timetas the weighted sum of a set of shared 3D
basis volumesf^V1;:::; ^VRtg; then
Vt=RtX
i=1f(t)i^Vi (2)
1To simplify notation, we write RXYasRXYin this paper.

---

## Page 4

whereis a scalar-volume product, Rtis the number of
shared volumes, and f(t)2RRtgives weights for the
shared volumes as a function of t. Shared volumes allow
information to be shared across time. In practice each ^Viis
represented as a TensoRF as in Equation 1 to save memory.
Unfortunately, we found in practice (and will show with
experiments) that shared volumes are still too costly; we
can only use small values for Rtwithout exhausting GPU
memory. Since each shared volume is a TensoRF, it has its
own independent MXY
r;vZ
r,etc.; we can further improve
efﬁciency by sharing these low-rank components across all
shared volumes. The 3D volume Vtat timetis then
Vt=R1X
r=1MXY
rvZ
rv1
rf1(t)r+R2X
r=1MXZ
rvY
rv2
rf2(t)r
(3)
+R3X
r=1MZY
rvX
rv3
rf3(t)r
where each fi(t)2RRigives a vector of weights for the
low-rank components at each time t.
In this formulation, fi(t)captures the model’s depen-
dence on time. The correct mathematical form for fi(t)is
not obvious. We initially framed fi(t)as a learned combi-
nation of sinusoidal or other ﬁxed basis functions, with the
hope that this could make periodic motion easier to learn;
however we found this inﬂexible and hard to optimize. fi(t)
could be an arbitrary nonlinear mapping, represented as an
MLP; however this would be slow. As a pragmatic trade-
off between ﬂexibility and speed, we represent fi(t)as a
learned piecewise linear function, implemented by linearly
interpolating along the ﬁrst axis of a learned TRimatrix.
3.3. HexPlane Representation
Equation 3 fully decouples the spatial and temporal mod-
eling of the scene: fi(t)models time and other terms model
space. However in real scenes space and time are entangled;
for example a particle moving in a circle is difﬁcult to model
under Equation 3 since its xandypositions are best mod-
eled separately as functions of t. This motivates us to re-
placevZ
rf1(t)rin Equation 3 with a joint function of tand
z, similarly represented as a piecewise linear function; this
can be implemented by bilinear interpolation into a learned
tensor of shape ZTR1. Applying the same transform
to all similar terms then gives our HexPlane representation,
which represents a 4D feature volume V2RXYZTFas:
D=R1X
r=1MXY
rMZT
rv1
r+R2X
r=1MXZ
rMYT
rv2
r
(4)
+R3X
r=1MYZ
rMXT
rv3
r
where each MAB
r2RABis a learned plane of features.
This formulation displays a beautiful symmetry, and strikesa balance between representational power and speed.
We can alternatively express a HexPlane as a function D
which maps a point (x;y;z;t )to anF-dimensional feature:
D(x; y; z; t ) = (PXYR 1xyPZTR 1
zt)VR1F
+(PXZR 2xzPYTR 2
yt)VR2F(5)
+(PYZR 3yzPXTR 3
xt)VR3F
whereis an elementwise product; the superscript of each
bold tensor represents its shape, and in a subscript rep-
resents a slice so each term is a vector-matrix product.
PXYR 1stacks all MXY
rto a 3D tensor, and VR1Fstacks
allv1
rto a 2D tensor; other terms are deﬁned similarly. Co-
ordinatesx;y;z;t are real-valued, so subscripts denote bi-
linear interpolation. This design reduces memory usage to
O(N2R+NTR +RF).
We can stack all VRiFintoVRFand rewrite Eq 5 as
[PXYR 1xyPZTR 1
zt;PXZR 2xzPYTR 2
yt;PYZR 3yzPXTR 3
xt]VRF
(6)
where ;concatenates vectors. As shown in Figure 2, a Hex-
Plane comprises three pairs of feature planes; each pair has
a spatial and a spatio-temporal plane with orthogonal axes
(e.g.XY=ZT ). Querying a HexPlane is fast, requiring just
six bilinear interpolations and a vector-matrix product.
3.4. Optimization
We represent dynamic 3D scenes using the proposed
HexPlane, which is optimized by photometric loss between
rendered and target images. For point (x;y;z;t ), its opac-
ity and appearance feature are quired from HexPlane, and
the ﬁnal RGB color is regressed from a tiny MLP with ap-
pearance feature and view direction as inputs. With points’
opacities and colors, images are rendered via volumetric
rendering. The optimization objective is:
L=1
jRjX
r2RkC(r) ^C(r)k2
2+regLreg (7)
Lreg,regare regularization and its weight; Ris the set of
rays andC(r);^C(r)are rendered and GT colors of ray r.
Color Regression. To save computations, we query points’
opacities directly from one HexPlane, and query appearance
features of points with high opacities from another separate
HexPlane. Queried features and view directions are fed into
a tiny MLP for RGB colors. An MLP-free design is also
feasible with spherical harmonics coefﬁcients as features.
Regularizer. Dynamic 3D reconstruction is a severely ill-
posed problem, needing strong regularizers. We apply Total
Variational (TV) loss on planes to force the spatial-temporal
continuity, and depth smooth loss in [48] to reduce artifacts.
Coarse to Fine Training. A coarse-to-ﬁne scheme is also
employed like [7, 86], where the resolution of grids gradu-
ally grows during training. This design accelerates the train-
ing and provides an implicit regularization on nearby grids.

---

## Page 5

InputViewsDynamicNovelViewSynthesis
Figure 3. High-Quality Dynamic Novel View Synthesis on Plenoptic Video dataset [31]. The proposed HexPlane could effectively
represent dynamic 3D scenes with complicated motions and render high-quality results with faithful details at various timesteps and
unseen viewpoints. We show several samples of input video sequences and synthesis results using a cyclic camera trajectory.
Emptiness Voxel. We keep a tiny 3D voxel indicating the
emptiness of scene regions and skip points in empty regions.
Since many regions are empty, it is helpful for accelera-
tion. To get this voxel, we evaluate points’ opacities across
time steps and reduce them to a single voxel with maximum
opacities. Although keeping several voxels for various time
intervals improves speeds, we only keep one for simplicity.
4. Experiments
We evaluate HexPlane, our proposed explicit representa-
tion, on dynamic novel view synthesis tasks with challeng-
ing datasets, comparing its performance and speed to state-
of-the-art methods. Through extensive ablation studies, we
explore its advantages and demonstrate its robustness to dif-
ferent feature fusion mechanisms, coordinate systems, and
decoding mechanisms. As our objective is to demonstrate
the effectiveness of this simple design, we prioritize Hex-
Plane’s simplicity and generality without implementing in-
tricate tricks for performance enhancement.
4.1. Dynamic Novel View Synthesis Results
For a comprehensive evaluation, we use two datasets
with distinct settings: the high-resolution, multi-camera
Plenoptic Video dataset [31], with challenging dynamic
content and intricate visual details; the monocular D-NeRF
dataset [58], featuring synthetic objects. Plenoptic Video
dataset assesses HexPlane’s representational capacity for
long videos with complex motions and ﬁne details, while
D-NeRF dataset tests its ability to handle monocular videos
and extremely sparse observations (with teleporting [18]).
Plenoptic Video dataset [31] is a real-world dataset cap-
tured by a multi-view camera system using 21 GoPro at
20282704 (2.7K) resolution and 30 FPS. Each scene
comprises 19 synchronized, 10-second videos, with 18 des-
Ground Truth
LLFF
DyNeRF
Ours
Figure 4. Visual Comparison of Synthesis Results. Since DyN-
eRF [31] model is not publicly available, we compare our results
to images provided the paper. With visually similar results, our
proposed HexPlane is over 100 faster than DyNeRF.
ignated for training and one for evaluation. This dataset is
suitable to test the representation ability as it features com-
plex and challenging dynamic content such as highly specu-
lar, translucent, and transparent objects; topology changes;
moving self-casting shadows; ﬁre ﬂames and strong view-
dependent effects for moving objects; and so on.
For a fair comparison, we adhere to the same training and
evaluation pipelines as DyNeRF [31] with slight changes
due to GPU resources. [31] trains its model on 8 V100
GPUs for a week, with 24576 batch size for 650K iterations.
We train our model on a single 16GB V100 GPU, with a
4096 batch size and the same iteration numbers, which is
6fewer sampling. We follow the same importance sam-
pling design and hierarch training as [31], with 512 spatial
grid sizes and 300 time grid sizes. The scene is in NDC [46].
As shown in Figure 3, HexPlane delivers high-quality
dynamic novel view synthesis across various times and
viewpoints. It accurately models real-world scenes with
intricate motions and challenging visual features, such as
ﬂames, showcasing its robust representational capabilities.
Quantitative comparisons with SOTA methods are in Ta-
ble 1, with baseline results from [31] paper. PSNR, struc-

---

## Page 6

Table 1. Quantitative Comparisons on Plenoptic Video dataset [31]. We report synthesis quality, training times (measured in GPU
hours) with speedups relative to DyNeRF [31]. With 672speedups, HexPlane†with fewer training iterations has comparable quantitative
results to DyNeRF. And HexPlane trained with the same iterations noticeably outperforms DyNeRF. Baseline methods are evaluated on a
particular scene, and we also report average results on all public scenes ( -all).Best and Second results are in highlight.
Model Steps PSNR " D-SSIM# LPIPS# JOD" Training Time# Speeds-up"
Neural V olumes [38] - 22.800 0.062 0.295 6.50 - -
LLFF [45] - 23.239 0.076 0.235 6.48 - -
NeRF-T [31] - 28.449 0.023 0.100 7.73 - -
DyNeRF [31] 650k 29.581 0.020 0.099 8.07 1344h 1 
HexPlane 650k 29.470 0.018 0.078 8.16 12h 112
HexPlane† 100k 29.263 0.020 0.097 8.14 2h 672
HexPlane-all 650k 31.705 0.014 0.075 8.47 12h 112 
HexPlane†-all 100k 31.569 0.016 0.089 8.36 2h 672 
Table 2. Quantitative Results on D-NeRF dataset [58]. Without
deformation, HexPlane has comparable or better results compared
to other deformation-based methods, and is noticeably faster.
Model Deform. PSNR " SSIM" LPIPS# Training
Time#
T-NeRF [58] 29.51 0.95 0.08 -
D-NeRF [58] X 30.50 0.95 0.07 20 hours
TiNeuV ox-S [13] X 30.75 0.96 0.07 12m 10s
TiNeuV ox-B [13] X 32.67 0.97 0.04 49m 46s
HexPlane (ours) 31.04 0.97 0.04 11m 30s
tural dissimilarity index measure (DSSIM) [61], percep-
tual quality measure LPIPS [93] and video visual differ-
ence measure Just-Objectionable-Difference (JOD) [41] are
evaluated for comprehensive study. Besides results on the
“ﬂame salmon” scene like [31], we also report average re-
sults on all public scenes except the unsynchronized one,
referred to HexPlane-all . We also train a model with fewer
training iterations as HexPlane† .
As shown in Table 1, HexPlane† achieves compa-
rable performance to DyNeRF [31] while substantially
faster ( 672speedups), highlighting the beneﬁts of em-
ploying explicit representations. DyNeRF uses a giant MLP
and per-frame latent codes to represent dynamic scenes,
which is slow due to tremendous MLP evaluations. When
trained with the same iteration number, HexPlane outper-
forms DyNeRF in all metrics except PSNR, while being
above 100faster. Although explicit representations typi-
cally demand signiﬁcant memory for their rapid processing
speeds due to explicit feature storage, HexPlane occupies a
mere 200MB for the entire model. This relatively compact
size is suitable for most GPUs. Given its fast speed, we
believe this tradeoff presents an attractive option.
Since the model of DyNeRF is not publicly available, it
is hard to compare the visual results directly. We download
images from the original paper and ﬁnd the most matching
images in our results, which are compared in Figure 4.
D-NeRF dataset [58] is a monocular video dataset with
360observations for synthetic objects. Dynamic 3D re-
construction for monocular video is challenging since onlyone observation is available each time. Current SOTA meth-
ods for monocular video usually have a deformation ﬁeld
and a static canonical ﬁeld, where points in dynamic scenes
are mapped to positions in the canonical ﬁeld. The map-
ping (deformation) is represented by another MLP.
The underlying assumption of deformation ﬁeld design
is that there are no topology changes, which does not al-
ways hold in the real world while holding in this dataset.
Again, to keep HexPlane general enough, we do not assume
deformation, the same as T-NeRF [58]. We use this dataset
to validate the ability to work with monocular videos.
We show quantitative results in Table 2. For fairness, all
training times are re-measured on the same 2080TI GPU.
Our HexPlane distinctly outperforms other methods even
without introducing the deformation ﬁeld, demonstrating
the inherent ability to deal with sparse observations due to
the shared basis. Again, our method is hundreds of times
faster than MLP-based designs like D-NeRF and T-NeRF.
Tineuvox [13] is a recent work for accelerating D-NeRF,
replacing canonical space MLP with a highly-optimized
sparse voxel Cuda kernel and keeping an MLP to represent
deformation. Therefore, it still uses explicit representation
for static scenes while our target is dynamic scenes. With-
out any custom Cuda kernels, our method is faster and bet-
ter than its light version and achieves the same LPIPS and
SSIM as its bigger version, which takes longer time to train.
4.2. Ablations and Analysis
We run deep introspections to HexPlane by answering
questions with extensive ablations. Ablations are conducted
mainly on D-NeRF [58] dataset because of efﬁciency.
How does HexPlane compare to others? We compare
HexPlane with other designs mentioned in the Method Sec-
tionin Table 3, where each method has various basis num-
bersR: (1). Volume Basis represents 4D volumes as
weighted summation of a set of shared 3D volumes as Eq 2,
which 3D volume is represented as Eq 1; (2). VM-T (vec-
tor, matrix and time) uses Eq 3 representing 4D volumes;

---

## Page 7

Table 3. Quantitative Results for Different Factorizations. Var-
ious factorization designs are evaluated on D-NeRF dataset with
different R(basis number). HexPlane achieves the best quality
and speed among all methods.
Model R PSNR"SSIM"LPIPS#Training
Time#
V olume Basis8 30.460 0.965 0.045 18m 04s
12 30.587 0.966 0.043 24m 06s
16 30.631 0.967 0.042 29m 20s
VM-T24 30.329 0.962 0.051 14m 36s
48 30.657 0.965 0.048 15m 58s
96 30.744 0.966 0.045 17m 03s
CP Decom.48 28.370 0.942 0.083 10m 31s
96 29.371 0.951 0.070 11m 03s
192 30.086 0.957 0.063 11m 33s
384 30.302 0.959 0.059 13m 06s
HexPlane24 30.886 0.966 0.042 10m 27s
48 31.042 0.968 0.039 11m 30s
Table 4. Ablations on Feature Planes Designs. We remove and
swap HexPlane’s planes and show results on D-NeRF dataset.
Model PSNR " SSIM" LPIPS# Training
Time#
Spatial Planes 20.369 0.879 0.148 9m 02s
Spatial-Temporal Planes 21.112 0.879 0.148 9m 29s
DoublePlane (XY-ZT) 30.370 0.961 0.054 8m 04s
HexPlane-Swap 28.562 0.954 0.056 11m 44s
HexPlane 31.042 0.968 0.039 11m 30s
(3). CP Decom. (CANDECOMP Decomposition) follows
[7], which represents 4D volumes using a set of vectors for
each axis. Implementation details are shown in Supp.
HexPlane gives optimal performance among all meth-
ods, illustrating the advantages of spatial-temporal planes.
Compared to other methods, spatial-temporal planes allow
HexPlane to model motions effectively with a small basis
numberR, leading to higher efﬁciency as well. Increasing
Rused for representation leads to better results while also
resulting in more computations. We also notice that an un-
suitable large Rmay lead to the overﬁtting problem, which
instead harms synthesis quality on novel views.
Could variants of HexPlane work? HexPlane has excel-
lent symmetry as it contains all pairs of coordinate axes.
By breaking this symmetry, we evaluate other variants in
Table 4. Spatial Planes only have three spatial planes:
PXY;PXZ;PYZ, and Spatial-Temporal Planes contain
the left three spatial-temporal planes; DoublePlane con-
tains only one group of paired planes, i.e. PXY;PZT;
HexPlane-Swap groups planes with repeated axes like
PXY;PXT. We report their performance and speeds.
As shown in the table, neither Spatial Planes norSpatial-
Temporal Planes could represent dynamic scenes alone, in-
dicating both are essential for representations. HexPlane-
Swap achieves inferior results since its axes are not comple-
mentary, losing features from the particular axis. Double-
Plane is less effective than HexPlane since HexPlane con-Table 5. Ablations on Feature Fusions Designs. We show results
with various fusion designs on D-NeRF dataset. HexPlane could
work with other fusion mechanisms, showing its robustness.
Fusion-
OneFusion-
TwoPSNR" SSIM" LPIPS#
MultiplyConcat 31.042 0.968 0.039
Sum 31.023 0.967 0.039
Multiply 30.345 0.966 0.041
SumConcat 25.428 0.931 0.084
Sum 25.227 0.928 0.090
Multiply 30.585 0.965 0.044
𝑁=128!𝑁=256!𝑁=512!𝑇=0.5#𝑓𝑟𝑎𝑚𝑒𝑠𝑇=1.0#𝑓𝑟𝑎𝑚𝑒𝑠31.593/0.14632.230/0.11332.498/0.08931.505/0.14532.382/0.11032.774/0.090
Figure 5. Synthesis Results with Different Spacetime Grid Res-
olutions. We show zoomed in synthesis results on Plenoptic Video
dataset with space grid resolution ranging from 1283to5123and
time grid ranging from half to one of the video frame number.
PSNR and LPIPS of the scene are reported below each images.
tains more comprehensive spatial-temporal modes.
How does grid resolution affect results? We show quali-
tative results with various spacetime grid resolutions in Fig-
ure 5 and report its PSNR/LPIPS below zoomed-in images.
Besides the space grid ranging from 1283to5123, we com-
pare results with different time grid resolutions, ranging
from half to the same as video frames. Higher resolutions
of the space grid lead to better synthesis quality, shown by
both images and metrics. HexPlane results are not notice-
ably affected by a smaller time grid resolution.
4.3. Robustness of HexPlane Designs
In addition to its performance and efﬁciency, this sec-
tion demonstrates HexPlane’s robustness to diverse design
choices, resulting in a highly adaptable and versatile frame-
work. This ﬂexibility allows for its applications across a
wide range of tasks and research directions.
Various Feature Fusion Mechanisms. In HexPlane, fea-
ture vectors from each plane are extracted and subsequently
fused into a single vector, which are multiplied by matrix
VRFlater for ﬁnal results. During fusion, features from
paired planes are ﬁrst element-wise multiplied ( fusion one )
and then concatenated into a single one ( fusion two ). We
explore other fusion designs beyond this Multiply-Concat .
Table 5 shows that Multiply-Concat is not the sole viable
design. Sum-Multiply and swapped counterpart Multiply-

---

## Page 8

Table 6. Dynamic View Synthesis without MLPs. HexPlane-SH
is a pure explicit model without MLPs on D-NeRF dataset, which
stores spherical harmonics (SH) as appearance features and di-
rectly regress RGB from it rather than MLPs. HexPlane-SH gives
reasonable results and faster than HexPlane with MLP.
Model PSNR " SSIM" LPIPS# Training Time#
HexPlane 31.042 0.968 0.039 11m 30s
HexPlane-SH 29.284 0.952 0.056 10m 42s
NDC Spherical NDC Spherical
Figure 6. Synthesis Results at Extreme Views for NDC and
Spherical Coordinates. Scenes represented in NDC are assumed
to be bounded along x; y axes, whose boundaries are observable
at extreme views(top-left and top-right corners), leading to incor-
rect geometries and artifacts. Using spherical coordinate, our Hex-
Plane could seamlessly represent dynamic unbounded scenes.
Sum both yield good results, albeit not optimal,2highlight-
ing an intriguing symmetry between multiplication and ad-
dition. Multiply-Multiply also produces satisfactory out-
comes, while Sum-Sum orSum-Concat fail, illustrating the
capacity limitations of addition compared to multiplication.
Overall, HexPlane is remarkably robust to various fusion
designs. We show complete results and analysis in Supp.
Spherical Harmonics Color Decoding. Instead of regress-
ing colors from MLPs, we evaluate a pure explicit model in
Table 6 without MLPs. Spherical harmonics (SH) [86] coef-
ﬁcients are computed directly from HexPlanes, and decoded
to RGBs with view directions. Using SH allows faster ren-
dering speeds at a slightly reduced quality. We ﬁnd that op-
timizing SH for dynamic scenes is more challenging com-
pared to [7, 15], which is an interesting future direction.
Spherical Coordinate for Unbounded Scenes. HexPlane
is limited to bounded scenes because grid sampling fails for
out-of-boundary points, which is a common issue among
explicit representations. Even normalized device coordi-
nates (NDC) [46] still require bounded x;yvalues and face-
forwarding assumptions. This limitation constrains the us-
age for real-world videos, leading to artifacts and incorrect
geometries as shown in Figure 6.
To address it, we re-parameterize (x;y;z;t )into spheri-
cal coordinate (;;r;t )and build HexPlane with ;;r;t
axes, where r= 1=p
x2+y2+z2,; is the polar an-
gle and azimuthal angle. During rendering, points are sam-
pled withrlinearly placed between 0 and 1. Without any
special adjustments, HexPlane can represent dynamic ﬁelds
with spherical coordinates, and deliver satisfactory results,
which provides a solution for modeling unbounded scenes
and exhibits robustness to different coordinate systems.
2Further tuning of initialization/other factors may lead to better results.
Figure 7. Dynamic Novel View Synthesis on Videos Captured
by iPhone. We test HexPlane on casual videos captured by
iPhone [18] and show synthesis results across novel timesteps and
views. Row one are results with interpolated camera poses, while
Row two shows results with extrapolated viewpoints, which are
signiﬁcantly distinct from camera poses used for video captures.
4.4. View Synthesis Results on Real Captured Video
We test HexPlane with monocular videos captured by
iPhone from [18], whose camera trajectories are relatively
casual and closer to real-world use cases. We show syn-
thesis results in 7. Without any deformation or category-
speciﬁc priors, our method could give realistic synthesis re-
sults on these real-world monocular videos, faithfully mod-
eling static backgrounds, casual motions of cats, typology
changes (cat’s tongue), and ﬁne details like cat hairs.
5. Conclusion
We propose HexPlane, an explicit representation for dy-
namic 3D scenes using six feature planes, which com-
putes features of spacetime points via sampling and fusions.
Compared to implicit representations, it could achieve com-
parable or even better synthesis quality for dynamic novel
view synthesis, with over hundreds of times accelerations.
In this paper, we aim to keep HexPlane neat and gen-
eral, preventing introducing deformation, category-speciﬁc
priors, or other speciﬁc tricks. Using these ideas to make
HexPlane better and faster would be an appealing future
direction. Also, using HexPlane in other tasks except for
dynamic novel view synthesis, e.g., spatial-temporal gener-
ation, would be interesting to explore [66]. We hope Hex-
Plane could contribute to a broad range of research in 3D.
Acknowledgments Toyota Research Institute provided
funds to support this work but this article solely reﬂects the
opinions and conclusions of its authors and not TRI or any
other Toyota entity. We thank Shengyi Qian for the title sug-
gestion, David Fouhey, Mohamed El Banani, Ziyang Chen,
Linyi Jin and for helpful discussions and feedbacks.

---

## Page 9

References
[1] Chong Bao, Bangbang Yang, Zeng Junyi, Bao Hu-
jun, Zhang Yinda, Cui Zhaopeng, and Zhang Guofeng.
Neumesh: Learning disentangled neural mesh-based im-
plicit ﬁeld for geometry and texture editing. In European
Conference on Computer Vision (ECCV) , 2022. 2
[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance ﬁelds. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision , pages 5855–
5864, 2021. 2
[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance ﬁelds. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 5470–5479, 2022. 2
[4] Ang Cao, Chris Rockwell, and Justin Johnson. Fwd: Real-
time novel view synthesis with forward warping and depth.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 15713–15724, 2022.
2
[5] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki
Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo,
Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis,
et al. Efﬁcient geometry-aware 3d generative adversarial
networks. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pages 16123–
16133, 2022. 2, 3, 15
[6] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu,
and Gordon Wetzstein. pi-gan: Periodic implicit generative
adversarial networks for 3d-aware image synthesis. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition , pages 5799–5809, 2021. 2
[7] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance ﬁelds. arXiv preprint
arXiv:2203.09517 , 2022. 1, 2, 3, 4, 7, 8
[8] Anpei Chen, Zexiang Xu, Xinyue Wei, Siyu Tang, Hao Su,
and Andreas Geiger. Factor ﬁelds: A uniﬁed framework for
neural ﬁelds and beyond. arXiv preprint arXiv:2302.01226 ,
2023. 2
[9] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang,
Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast
generalizable radiance ﬁeld reconstruction from multi-view
stereo. In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision , pages 14124–14133, 2021. 2
[10] Yu Deng, Jiaolong Yang, Jianfeng Xiang, and Xin Tong.
Gram: Generative radiance manifolds for 3d-aware image
generation. 2022 IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition (CVPR) , pages 10663–10673,
2022. 2
[11] Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenen-
baum, and Jiajun Wu. Neural radiance ﬂow for 4d view
synthesis and video processing. In 2021 IEEE/CVF In-
ternational Conference on Computer Vision (ICCV) , pages
14304–14314. IEEE Computer Society, 2021. 2[12] Bernhard Egger, William AP Smith, Ayush Tewari, Ste-
fanie Wuhrer, Michael Zollhoefer, Thabo Beeler, Florian
Bernard, Timo Bolkart, Adam Kortylewski, Sami Romd-
hani, et al. 3d morphable face models—past, present, and
future. ACM Transactions on Graphics (TOG) , 39(5):1–38,
2020. 13
[13] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance ﬁelds with time-aware neural voxels.
arXiv preprint arXiv:2205.15285 , 2022. 2, 6, 13
[14] Sara Fridovich-Keil, Giacomo Meanti, Frederik Warburg,
Benjamin Recht, and Angjoo Kanazawa. K-planes: Ex-
plicit radiance ﬁelds in space, time, and appearance. arXiv
preprint arXiv:2301.10241 , 2023. 2
[15] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance ﬁelds without neural networks. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 5501–5510, 2022. 1, 2, 8
[16] Wanshui Gan, Hongbin Xu, Yi Huang, Shifeng Chen, and
Naoto Yokoya. V4d: V oxel for 4d novel view synthesis.
ArXiv , abs/2205.14332, 2022. 2
[17] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin
Huang. Dynamic view synthesis from dynamic monocu-
lar video. In Proceedings of the IEEE/CVF International
Conference on Computer Vision , pages 5712–5721, 2021.
1, 2
[18] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell,
and Angjoo Kanazawa. Monocular dynamic view synthe-
sis: A reality check. In NeurIPS , 2022. 5, 8, 13, 14
[19] Stephan J Garbin, Marek Kowalski, Matthew Johnson,
Jamie Shotton, and Julien Valentin. Fastnerf: High-
ﬁdelity neural rendering at 200fps. In Proceedings of the
IEEE/CVF International Conference on Computer Vision ,
pages 14346–14355, 2021. 2
[20] Xiang Guo, Guanying Chen, Yuchao Dai, Xiaoqing Ye, Ji-
adai Sun, Xiao Tan, and Errui Ding. Neural deformable
voxel grid for fast optimization of dynamic view synthesis.
InAsian Conference on Computer Vision , 2022. 2
[21] Peter Hedman, Pratul P Srinivasan, Ben Mildenhall,
Jonathan T Barron, and Paul Debevec. Baking neural radi-
ance ﬁelds for real-time view synthesis. In Proceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, pages 5875–5884, 2021. 2
[22] Di Huang, Sida Peng, Tong He, Xiaowei Zhou, and Wanli
Ouyang. Ponder: Point cloud pre-training via neural ren-
dering. arXiv preprint arXiv:2301.00157 , 2022. 2
[23] Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter
Abbeel, and Ben Poole. Zero-shot text-guided object gen-
eration with dream ﬁelds. 2022. 2
[24] Hankyu Jang and Daeyoung Kim. D-tensorf: Tenso-
rial radiance ﬁelds for dynamic scenes. arXiv preprint
arXiv:2212.02375 , 2022. 2
[25] Mohammad Mahdi Johari, Yann Lepoittevin, and Franc ¸ois
Fleuret. Geonerf: Generalizing nerf with geometry priors.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 18365–18375, 2022.
2

---

## Page 10

[26] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. arXiv preprint arXiv:1412.6980 ,
2014. 14
[27] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitz-
mann. Decomposing nerf for editing via feature ﬁeld distil-
lation. arXiv , 2022. 2
[28] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
Imagenet classiﬁcation with deep convolutional neural net-
works. Communications of the ACM , 60(6):84–90, 2017.
14
[29] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and
Ping Tan. Streaming radiance ﬁelds for 3d video synthe-
sis.ArXiv , abs/2210.14831, 2022. 2
[30] Ruilong Li, Julian Tanke, Minh V o, Michael Zollhofer,
Jurgen Gall, Angjoo Kanazawa, and Christoph Lassner.
Tava: Template-free animatable volumetric actors. ArXiv ,
abs/2206.08929, 2022. 2
[31] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 5521–5531, 2022. 1,
2, 5, 6, 13, 14, 15, 16, 18
[32] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver
Wang. Neural scene ﬂow ﬁelds for space-time view syn-
thesis of dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 6498–6508, 2021. 1, 2
[33] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki
Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja
Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-
resolution text-to-3d content creation. arXiv preprint
arXiv:2211.10440 , 2022. 2
[34] S. Lionar, Daniil Emtsev, Dusan Svilarkovic, and Songyou
Peng. Dynamic plane convolutional occupancy networks.
2021 IEEE Winter Conference on Applications of Computer
Vision (WACV) , pages 1828–1837, 2020. 2
[35] Jia-Wei Liu, Yan-Pei Cao, Weijia Mao, Wenqiao Zhang,
David Junhao Zhang, Jussi Keppo, Ying Shan, Xiaohu Qie,
and Mike Zheng Shou. Devrf: Fast deformable voxel ra-
diance ﬁelds for dynamic scenes. ArXiv , abs/2205.15723,
2022. 2
[36] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua,
and Christian Theobalt. Neural sparse voxel ﬁelds. ArXiv ,
abs/2007.11571, 2020. 2
[37] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and
Christian Theobalt. Neural sparse voxel ﬁelds. Advances in
Neural Information Processing Systems , 33:15651–15663,
2020. 2
[38] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel
Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural
volumes: Learning dynamic renderable volumes from im-
ages. arXiv preprint arXiv:1906.07751 , 2019. 6
[39] Stephen Lombardi, Tomas Simon, Gabriel Schwartz,
Michael Zollhoefer, Yaser Sheikh, and Jason M. Saragih.
Mixture of volumetric primitives for efﬁcient neural ren-dering. ACM Transactions on Graphics (TOG) , 40:1 – 13,
2021. 2
[40] Matthew Loper, Naureen Mahmood, Javier Romero, Ger-
ard Pons-Moll, and Michael J. Black. SMPL: A skinned
multi-person linear model. ACM Trans. Graphics (Proc.
SIGGRAPH Asia) , 34(6):248:1–248:16, Oct. 2015. 13
[41] Rafał K Mantiuk, Gyorgy Denes, Alexandre Chapiro, An-
ton Kaplanyan, Gizem Rufo, Romain Bachy, Trisha Lian,
and Anjul Patney. Fovvideovdp: A visible difference pre-
dictor for wide ﬁeld-of-view video. ACM Transactions on
Graphics (TOG) , 40(4):1–19, 2021. 6, 14
[42] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi,
Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duck-
worth. Nerf in the wild: Neural radiance ﬁelds for uncon-
strained photo collections. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 7210–7219, 2021. 2
[43] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Se-
bastian Nowozin, and Andreas Geiger. Occupancy net-
works: Learning 3d reconstruction in function space. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition , pages 4460–4470, 2019. 2
[44] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla,
Pratul P Srinivasan, and Jonathan T Barron. Nerf in the
dark: High dynamic range view synthesis from noisy raw
images. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pages 16190–
16199, 2022. 2
[45] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light ﬁeld fusion: Practical view syn-
thesis with prescriptive sampling guidelines. ACM Trans-
actions on Graphics (TOG) , 38(4):1–14, 2019. 6
[46] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance ﬁelds for view syn-
thesis. In ECCV , 2020. 1, 2, 3, 5, 8, 14
[47] Thomas M ¨uller, Alex Evans, Christoph Schied, and
Alexander Keller. Instant neural graphics primitives with
a multiresolution hash encoding. ACM Trans. Graph. ,
41(4):102:1–102:15, July 2022. 1, 2
[48] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall,
Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan.
Regnerf: Regularizing neural radiance ﬁelds for view syn-
thesis from sparse inputs. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 5480–5490, 2022. 2, 4
[49] Michael Niemeyer and Andreas Geiger. Giraffe: Repre-
senting scenes as compositional generative neural feature
ﬁelds. In Proc. IEEE Conf. on Computer Vision and Pat-
tern Recognition (CVPR) , 2021. 2
[50] Michael Niemeyer, Lars Mescheder, Michael Oechsle,
and Andreas Geiger. Differentiable volumetric rendering:
Learning implicit 3d representations without 3d supervi-
sion. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition , pages 3504–3515,
2020. 2

---

## Page 11

[51] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and
Felix Heide. Neural scene graphs for dynamic scenes. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 2856–2865, 2021. 2
[52] Hao Ouyang, Bo Zhang, Pan Zhang, Hao Yang, Jiaolong
Yang, Dong Chen, Qifeng Chen, and Fang Wen. Real-time
neural character rendering with pose-guided multiplane im-
ages. ArXiv , abs/2204.11820, 2022. 2
[53] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Soﬁen
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
Martin-Brualla. Nerﬁes: Deformable neural radiance ﬁelds.
InProceedings of the IEEE/CVF International Conference
on Computer Vision , pages 5865–5874, 2021. 1, 2, 13, 18
[54] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Soﬁen Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz. Hypernerf: A higher-
dimensional representation for topologically varying neural
radiance ﬁelds. arXiv preprint arXiv:2106.13228 , 2021. 1,
2, 13
[55] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc
Pollefeys, and Andreas Geiger. Convolutional occupancy
networks. In European Conference on Computer Vision ,
pages 523–540. Springer, 2020. 2
[56] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang,
Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural
body: Implicit neural representations with structured la-
tent codes for novel view synthesis of dynamic humans.
2021 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR) , pages 9050–9059, 2021. 2
[57] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Milden-
hall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv ,
2022. 2
[58] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer. D-nerf: Neural radiance ﬁelds
for dynamic scenes. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
10318–10327, 2021. 1, 2, 5, 6, 13, 14, 15, 17
[59] Shengyi Qian, Alexander Kirillov, Nikhila Ravi, Deven-
dra Singh Chaplot, Justin Johnson, David F Fouhey, and
Georgia Gkioxari. Recognizing scenes from novel view-
points. arXiv preprint arXiv:2112.01520 , 2021. 2
[60] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas
Geiger. Kilonerf: Speeding up neural radiance ﬁelds with
thousands of tiny mlps. 2021 IEEE/CVF International Con-
ference on Computer Vision (ICCV) , pages 14315–14325,
2021. 2
[61] Umme Sara, Morium Akter, and Mohammad Shorif Ud-
din. Image quality assessment through fsim, ssim, mse and
psnr—a comparative study. Journal of Computer and Com-
munications , 7(3):8–18, 2019. 6
[62] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas
Geiger. Graf: Generative radiance ﬁelds for 3d-aware im-
age synthesis. Advances in Neural Information Processing
Systems , 33:20154–20166, 2020. 2
[63] Katja Schwarz, Axel Sauer, Michael Niemeyer, Yiyi Liao,
and Andreas Geiger. V oxgraf: Fast 3d-aware image synthe-
sis with sparse voxel grids. In Advances in Neural Informa-
tion Processing Systems , 2022. 2[64] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu,
Hongwen Zhang, and Yebin Liu. Tensor4d : Efﬁcient neu-
ral 4d decomposition for high-ﬁdelity dynamic reconstruc-
tion and rendering. ArXiv , abs/2211.11610, 2022. 2
[65] Karen Simonyan and Andrew Zisserman. Very deep convo-
lutional networks for large-scale image recognition. arXiv
preprint arXiv:1409.1556 , 2014. 14
[66] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual,
Iurii Makarov, Filippos Kokkinos, Naman Goyal, An-
drea Vedaldi, Devi Parikh, Justin Johnson, and Yaniv
Taigman. Text-to-4d dynamic scene generation. ArXiv ,
abs/2301.11280, 2023. 2, 8
[67] Vincent Sitzmann, Justus Thies, Felix Heide, Matthias
Nießner, Gordon Wetzstein, and Michael Zollhofer. Deep-
voxels: Learning persistent 3d feature embeddings. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pages 2437–2446, 2019. 2
[68] Vincent Sitzmann, Michael Zollhoefer, and Gordon Wet-
zstein. Scene representation networks: Continuous 3d-
structure-aware neural scene representations. ArXiv ,
abs/1906.01618, 2019. 2
[69] Liangchen Song, Anpei Chen, Zhong Li, Z. Chen, Lele
Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerf-
player: A streamable dynamic scene representation with
decomposed neural radiance ﬁelds. ArXiv , abs/2210.15947,
2022. 2
[70] Shih-Yang Su, Frank Yu, Michael Zollhoefer, and Helge
Rhodin. A-nerf: Articulated neural radiance ﬁelds for
learning human shape, appearance, and pose. In NeurIPS ,
2021. 2
[71] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct
voxel grid optimization: Super-fast convergence for radi-
ance ﬁelds reconstruction. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 5459–5469, 2022. 1, 2
[72] Jiaming Sun, Xi Chen, Qianqian Wang, Zhengqi Li, Hadar
Averbuch-Elor, Xiaowei Zhou, and Noah Snavely. Neural
3d reconstruction in the wild. ACM SIGGRAPH 2022 Con-
ference Proceedings , 2022. 2
[73] Towaki Takikawa, Alex Evans, Jonathan Tremblay, Thomas
M¨uller, Morgan McGuire, Alec Jacobson, and Sanja Fidler.
Variable bitrate neural ﬁelds. ACM SIGGRAPH 2022 Con-
ference Proceedings , 2022. 2
[74] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek
Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T.
Barron, and Henrik Kretzschmar. Block-nerf: Scalable
large scene neural view synthesis. 2022 IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR) ,
pages 8238–8248, 2022. 2
[75] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik,
Michael Zollh ¨ofer, Christoph Lassner, and Christian
Theobalt. Non-rigid neural radiance ﬁelds: Reconstruc-
tion and novel view synthesis of a dynamic scene from
monocular video. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision , pages 12959–
12970, 2021. 2
[76] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zick-
ler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf:

---

## Page 12

Structured view-dependent appearance for neural radiance
ﬁelds. arXiv preprint arXiv:2112.03907 , 2021. 2
[77] Can Wang, Menglei Chai, Mingming He, Dongdong Chen,
and Jing Liao. Clip-nerf: Text-and-image driven manip-
ulation of neural radiance ﬁelds. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 3835–3844, 2022. 2
[78] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, and
Huaping Liu. Mixed neural voxels for fast multi-view video
synthesis. ArXiv , abs/2212.00190, 2022. 2
[79] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao,
Yanshun Zhang, Yingliang Zhang, Minye Wu, Jingyi Yu,
and Lan Xu. Fourier plenoctrees for dynamic radiance ﬁeld
rendering in real-time. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR) , pages 13524–13534, June 2022. 2
[80] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P
Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo
Martin-Brualla, Noah Snavely, and Thomas Funkhouser.
Ibrnet: Learning multi-view image-based rendering. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 4690–4699, 2021. 2
[81] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil
Kim. Space-time neural irradiance ﬁelds for free-viewpoint
video. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pages 9421–
9431, 2021. 2
[82] Jiaxin Xie, Hao Ouyang, Jingtan Piao, Chenyang Lei, and
Qifeng Chen. High-ﬁdelity 3d gan inversion by pseudo-
multi-view optimization. arXiv preprint arXiv:2211.15662 ,
2022. 2
[83] Hongyi Xu, Thiemo Alldieck, and Cristian Sminchisescu.
H-nerf: Neural radiance ﬁelds for rendering and temporal
reconstruction of humans in motion. In NeurIPS , 2021. 2
[84] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin
Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf:
Point-based neural radiance ﬁelds. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 5438–5448, 2022. 2
[85] Lin Yen-Chen, Pete Florence, Jonathan T. Barron, Tsung-
Yi Lin, Alberto Rodriguez, and Phillip Isola. NeRF-
Supervision: Learning dense object descriptors from neu-
ral radiance ﬁelds. In IEEE Conference on Robotics and
Automation (ICRA) , 2022. 2
[86] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and
Angjoo Kanazawa. PlenOctrees for real-time rendering of
neural radiance ﬁelds. In arXiv , 2021. 1, 2, 4, 8
[87] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo
Kanazawa. pixelnerf: Neural radiance ﬁelds from one or
few images. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 4578–
4587, 2021. 2
[88] Wentao Yuan, Zhaoyang Lv, Tanner Schmidt, and Steven
Lovegrove. Star: Self-supervised tracking and reconstruc-
tion of rigid objects in motion with neural rendering. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 13144–13152, 2021.
2[89] Jiakai Zhang, Xinhang Liu, Xinyi Ye, Fuqiang Zhao, Yan-
shun Zhang, Minye Wu, Yingliang Zhang, Lan Xu, and
Jingyi Yu. Editable free-viewpoint video using a lay-
ered neural representation. ACM Transactions on Graphics
(TOG) , 40:1 – 18, 2021. 2
[90] Jason Y . Zhang, Gengshan Yang, Shubham Tulsiani, and
Deva Ramanan. Ners: Neural reﬂectance surfaces for
sparse-view 3d reconstruction in the wild. In NeurIPS ,
2021. 2
[91] Kai Zhang, Nicholas I. Kolkin, Sai Bi, Fujun Luan, Zexi-
ang Xu, Eli Shechtman, and Noah Snavely. Arf: Artistic
radiance ﬁelds. In ECCV , 2022. 2
[92] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen
Koltun. Nerf++: Analyzing and improving neural radiance
ﬁelds. arXiv preprint arXiv:2010.07492 , 2020. 2, 13
[93] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 6, 14
[94] Xiaoshuai Zhang, Sai Bi, Kalyan Sunkavalli, Hao Su, and
Zexiang Xu. Nerfusion: Fusing radiance ﬁelds for large-
scale scene reconstruction. 2022 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) , pages
5439–5448, 2022. 2
[95] Yuqi Zhang, Guanying Chen, and Shuguang Cui. Ef-
ﬁcient large-scale scene representation with a hybrid of
high-resolution grid and plane features. arXiv preprint
arXiv:2303.03003 , 2023. 2
[96] Fuqiang Zhao, Wei Yang, Jiakai Zhang, Pei-Ying Lin,
Yingliang Zhang, Jingyi Yu, and Lan Xu. Humannerf: Gen-
eralizable neural human radiance ﬁeld from sparse inputs.
ArXiv , abs/2112.02789, 2021. 2
[97] Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J
Black, and Otmar Hilliges. Pointavatar: Deformable
point-based head avatars from videos. arXiv preprint
arXiv:2212.08377 , 2022. 2
[98] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and
Andrew Davison. In-place scene labelling and understand-
ing with implicit scene representation. In Proceedings of
the International Conference on Computer Vision (ICCV) ,
2021. 2
[99] Zhizhuo Zhou and Shubham Tulsiani. Sparsefusion: Dis-
tilling view-conditioned diffusion for 3d reconstruction.
arXiv preprint arXiv:2212.00792 , 2022. 2
[100] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu,
Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc
Pollefeys. Nice-slam: Neural implicit scalable encoding
for slam. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pages 12786–
12796, 2022. 2
[101] Yiming Zuo and Jia Deng. View synthesis with sculpted
neural points. ArXiv , abs/2205.05869, 2022. 2

---

## Page 13

6. General Discussions
6.1. Broader Impacts
Our work aims to design an explicit representation for
dynamic 3D scenes. We only reconstruct existing scenes
and render images from different viewpoints and timesteps.
Therefore, we don’t generate any new scenes or deceive
contents which don’t exist before. Our current method is
not intended and can also not be used to create fake materi-
als, which could mislead others.
We use Plenoptic Video dataset [31] in our experiments,
which contains human faces in videos. This dataset is a
public dataset with License: CC-BY-NC 4.0 with consent.
Our method is hundreds of times faster than exist-
ing methods, consuming signiﬁcantly less computation re-
sources. Considering the GPU resource usages, our method
could save considerably carbon emission.
6.2. Limitations and Future Directions
For comprehensively understanding HexPlane, we dis-
cuss its limitations and potential future improvements.
The ultimate goal of our paper is to propose and validate
an explicit representation for dynamic scenes instead of pur-
chasing SOTA numbers. To this end, we intend to make
HexPlanes simple and general, making minor assumptions
about the scenes and not introducing complicated tricks to
improve performance. This principle leads to elegant solu-
tions while potentially limiting performance as well. In the
following, we will discuss these in detail.
Many methods [13, 53, 54, 58] use deformation and
canonical ﬁelds to represent dynamic 3D scenes with
monocular videos, where spacetime points are mapped into
a static 3D scene represented by a canonical ﬁeld. Again,
we don’t employ deformation ﬁelds in our design since this
assumption is not always held in the real world, especially
for scenes with typology changes and new-emerging con-
tent. But this design is very effective for monocular videos
since it introduces a solid information-sharing mechanism
and allows learning 3D structures from very sparse views.
HexPlane uses an inherent basis-sharing mechanism to cope
with sparse observations. Although this design is shown to
be powerful in our experiments, it is still less effective than
the aforementioned deformation ﬁeld, leading to degraded
results for scenes with extremely sparse observations. In-
troducing deformation ﬁelds into HexPlane, like using Hex-
Plane to represent deformation ﬁelds, would be an appeal-
ing improvement for monocular videos.
Similarly, category-speciﬁc priors like 3DMM [12] or
SMPL [40] are even more powerful than deformation ﬁelds,
which enormously improve results but are hardly limited to
particular scenes. Combining these ideas with HexPlane for
speciﬁc scenes would be very interesting.
Existing works demonstrated that explicit representa-tions are prone to giving artifacts and require strong reg-
ularizations for good results, which also holds in HexPlane.
There are color jittering and artifacts in the synthesized re-
sults, demanding stronger regularizations and other tricks to
improve results further. Special spacetime regularizations
and other losses like optical ﬂow loss would be an interest-
ing future direction to explore. Also, instead of simply rep-
resenting everything using spherical coordinates in the pa-
per, we could have a foreground and background model like
NeRF++ [92], where the background is modeled in spher-
ical coordinates. Having separate foreground background
models could noticeably improve the results. Moreover,
rather than using the same basis for representing a long
video, using a different basis for different video clips may
give better results. We believe HexPlane could be further
improved with these adjustments.
Besides dynamic novel view synthesis, we believe Hex-
Plane could be utilized in a broader range of research, like
dynamic scene generation or edits.
6.3. License
We provide licenses of assets used in our paper.
Plenoptic Video Dataset [31]. We evaluate our method
on all all public scenes of Plenoptic Video dataset [31],
except a-synchronize scene “coffee-martini”. The dataset
is inhttps://github.com/facebookresearch/
Neural_3D_Video with License CC-BY-NC 4.0
D-NeRF Dataset [58]. We use D-NeRF dataset provided
inhttps://github.com/albertpumarola/D-
NeRF .
iPhone Dataset [18]. The iPhone dataset is provided in
https://github.com/KAIR-BAIR/dycheck/ , li-
censed under Apache-2.0 license.
Plenoptic Video Dataset Baselines [31]. For all baselines
in this dataset, we use numbers reported in the original pa-
per since these models are not publicly available.
D-NeRF Dataset Baselines [58]. D-NeRF model is
inhttps://github.com/albertpumarola/D-
NeRF and Tineuvox model [13] is in https://github.
com/hustvl/TiNeuVox . Tineuvox is licensed under
Apache License 2.0.
7. Training Details and More Results
7.1. Plenoptic Video Dataset [31].
Plenoptic Video Dataset [31] is a multi-view real-world
video dataset, where each video is 10-second long. The
training and testing views are shown in Figure 8.
We haveR1= 48;R2= 24;R3= 24 for appear-
ance HexPlance, where R1;R2;R3are basis numbers for
XY ZT;XZ YT;YZ XTplanes. For opacity Hex-
Plane, we set R1= 24;R2= 12;R3= 12 . We have differ-
entR1;R2;R3since scenes in this dataset are almost face-

---

## Page 14

TestViewFigure 8. Train and Test View of Plenoptic Video Dataset [31]. Plenoptic Video Dataset has 18 train views and 1 test view.
forwarding, demanding better representation along the XY
plane. The scene is modeled using normalized device co-
ordinate (NDC) [46] with min boundaries [ 2:5; 2:0;0:0]
and max boundaries [2:5;2:0;1:0].
Instead of giving the same grid resolutions along X;Y;Z
axes, we adjust them based on their boundary distances.
That is, we give larger grid resolution to axis ranging a
longer distance, like Xaxis from 2:5to2:5, and provide
smaller grid resolution to axis going a shorter length, like Z
axis from 0to1. The ratio of grid resolutions for different
axes is the same as their distance ratios, while the total grid
size number is manually controlled.
During training, HexPlane starts with a space grid size
of643and doubles its resolution at 70k, 140k, and 210k to
5123. The emptiness voxel is calculated at 50k and 100k
iterations. The learning rate for feature planes is 0.02, and
the learning rate for VRFand neural network is 0.001. All
learning rates are exponentially decayed. We use Adam [26]
for optimization with 1= 0:9;2= 0:99. We apply Total
Variational loss on all feature planes with = 0:0005 for
spatial axes and = 0:001for temporal axes.
We follow the hierarch training pipeline as [31]. Hex-
Plane in Table 1 uses 650k iterations, with 300k stage one
training, 250k stage two training and 100k stage three train-
ing.HexPlane† uses 100k iterations in total, with 10k stage
one training, 50k stage two training and 40k stage three
training. According to [31], stage one is a global-median-
based weighted sampling with = 0:001; stage two is also
a global-median-based weighted sampling with = 0:02;
stage three is a temporal-difference-based weighted sam-
pling with= 0:1.
In evaluation, D-SSIM is computed as1 MS-SSIM
2and
LPIPS [93] is calculated using AlexNet [28]. We use de-
fault settings for Just-Objectionable-Difference (JOD) [41].
Each scene results are in Table 7, and more visualiza-
tions are in Figure 9. We found that HexPlane gives visually
more smooth results than HexPlane† . Since we don’t have
baseline results, we don’t explore new evaluation metrics.7.2. D-NeRF Dataset [58].
We haveR1=R2=R3= 48 for appearance HexPlane
since it has 360videos. For opacity HexPlane, we set
R1=R2=R3= 24 . The bounding box has max bound-
aries [1:5;1:5;1:5]and min boundaries [ 1:5; 1:5; 1:5].
During training, HexPlane starts with space grid size of
323and upsamples its resolution at 3k, 6k, 9k to 2003. The
emptiness voxel is calculated at 4k and 10k iterations. Total
training iteration is 25k. The learning rate for feature planes
are 0.02, and learning rate for VRFand neural network is
0.001. All learning rates are exponentially decayed. We use
Adam [26] for optimization with 1= 0:9;2= 0:99. Dur-
ing evaluation, the LPIPS is computed using VGG-Net [65]
following previous works. We show per-scene quantitative
results in Table 8 and visualizations in Figure 10.
7.3. iPhone dataset [18].
7.4. Ablation Details.
For a fair comparison, we ﬁx all settings in ablations.
Volume Basis represents 4D volumes as the weighted sum-
mation of a set of shared 3D volumes as Eq 2 in main paper,
where each 3D volume is represented in Eq 1 format to save
memory. The 3D volume Vtat timetis then:
Vt=RtX
i=1f(t)i^Vi
=RtX
i=1f(t)i(R1X
r=1MXY
r;ivZ
r;iv1
r;i+R2X
r=1MXZ
r;i
vY
r;iv2
r;i+R3X
r=1MYZ
r;ivX
r;iv3
r;i)(8)
Similarly, we use a piece-wise linear function to approxi-
matef(t). In experiments, we set R1=R2=R3= 16 for
appearance HexPlane and R1=R2=R3= 8 for opacity
HexPlane. We evaluate Rt= 8;12;16in experiments.
VM-T (Vector, Matrix and Time) uses Eq 3 in main paper
to represent 4D volumes.

---

## Page 15

Table 7. Results of Plenoptic Video Dataset [31]. We report results of each scene.
Model Flame Salmon Cook Spinach Cut Roasted Beef
PSNR" D-SSIM#LPIPS# JOD" PSNR" D-SSIM#LPIPS# JOD" PSNR" D-SSIM#LPIPS# JOD"
HexPlane 29.470 0.018 0.078 8.16 32.042 0.015 0.082 8.32 32.545 0.013 0.080 8.59
HexPlane† 29.263 0.020 0.097 8.14 31.860 0.017 0.097 8.25 32.712 0.015 0.094 8.37
Model Flame Steak Sear Steak Average
PSNR" D-SSIM#LPIPS# JOD" PSNR" D-SSIM#LPIPS# JOD" PSNR" D-SSIM#LPIPS# JOD"
HexPlane 32.080 0.011 0.066 8.61 32.387 0.011 0.070 8.66 31.705 0.014 0.075 8.47
HexPlane† 31.924 0.012 0.081 8.51 32.085 0.014 0.079 8.51 31.569 0.016 0.090 8.36
Table 8. Per-Scene Results of D-NeRF Dataset [58]. We report results of each scene.
Model Hell Warrior Mutant Hook
PSNR" SSIM" LPIPS" PSNR" SSIM" LPIPS" PSNR" SSIM" LPIPS"
T-NeRF 23.19 0.93 0.08 30.56 0.96 0.04 27.21 0.94 0.06
D-NeRF 25.02 0.95 0.06 31.29 0.97 0.02 29.25 0.96 0.11
TiNeuV ox-S 27.00 0.95 0.09 31.09 0.96 0.05 29.30 0.95 0.07
TiNeuV ox-B 28.17 0.97 0.07 33.61 0.98 0.03 31.45 0.97 0.05
HexPlane 24.24 0.94 0.07 33.79 0.98 0.03 28.71 0.96 0.05
Model Bouncing Balls Lego T-Rex
PSNR" SSIM" LPIPS" PSNR" SSIM" LPIPS" PSNR" SSIM" LPIPS"
T-NeRF 37.81 0.98 0.12 23.82 0.90 0.15 30.19 0.96 0.13
D-NeRF 38.93 0.98 0.10 21.64 0.83 0.16 31.75 0.97 0.03
TiNeuV ox-S 39.05 0.99 0.06 24.35 0.88 0.13 29.95 0.96 0.06
TiNeuV ox-B 40.73 0.99 0.04 25.02 0.92 0.07 32.70 0.98 0.03
HexPlane 39.69 0.99 0.03 25.22 0.94 0.04 30.67 0.98 0.03
Model Stand Up Jumping Jacks Average
PSNR" SSIM" LPIPS" PSNR" SSIM" LPIPS" PSNR" SSIM" LPIPS"
T-NeRF 31.24 0.97 0.02 32.01 0.97 0.03 29.51 0.95 0.08
D-NeRF 32.79 0.98 0.02 32.80 0.98 0.03 30.50 0.95 0.07
TiNeuV ox-S 32.89 0.98 0.03 32.33 0.97 0.04 30.75 0.96 0.07
TiNeuV ox-B 35.43 0.99 0.02 34.23 0.98 0.03 32.64 0.97 0.04
HexPlane 34.36 0.98 0.02 31.65 0.97 0.04 31.04 0.97 0.04
Vt=R1X
r=1MXY
rvZ
rv1
rf1
r(t) +R2X
r=1MXZ
rvY
r
v2
rf2
r(t) +R3X
r=1MZY
rvX
rv3
rf3
r(t)(9)
We evaluate R1=R2=R3= 24;48;96.
CP Decom. (CANDECOMP Decomposition) represents
4D volumes using a set of vectors for each axis.
Vt=RX
r=1vX
rvY
rvZ
rvrfr(t) (10)
vX;vY;vZare feature vectors corresponding to X;Y;Z
axes. We evaluate R= 48;96;192;384in experiments.7.5. Fusion Ablations
We provide complete results of fusion ablations in Ta-
ble 9. For Fusion-One andFusion-Two , we choose one
fusion method from Concat ,Sum, and Multiply , and enu-
merate all combinations of fusion methods. Besides that,
we also explore to regress opacities from MLPs like [5]. In
this setting, we sample opacity features, 8-dim feature vec-
tors from HexPlane and regress opacity values from another
MLP.
Using MLP to regress opacities could substantially boost
the the results for all designs, at the cost of slower rendering
speeds. Interestingly, we found that
Please also note that we found different fusion designs
expect different

---

## Page 16

Figure 9. View Synthesis Results and Depths at Test View on Plenoptic Video Dataset [31].

---

## Page 17

Figure 10. View Synthesis Results on D-NeRF Dataset [58].

---

## Page 18

Table 9. Ablations on Feature Fusions Designs. We show results with various fusion designs on D-NeRF dataset. HexPlane could work
with other fusion mechanisms, showing its robustness.
Opacity without MLP Regression Opacity with MLP Regression
Fusion-One Fusion-Two PSNR " SSIM" LPIPS# PSNR" SSIM" LPIPS#
MultiplyConcat 31.042 0.968 0.039 31.477 0.969 0.037
Sum 31.023 0.967 0.039 31.318 0.969 0.038
Multiply 30.345 0.966 0.041 31.094 0.968 0.038
SumConcat 25.428 0.931 0.084 29.240 0.954 0.057
Sum 25.227 0.928 0.090 28.024 0.946 0.067
Multiply 30.585 0.965 0.044 30.934 0.966 0.041
ConcatConcat 25.057 0.928 0.073 30.173 0.961 0.049
Sum 24.915 0.925 0.077 27.971 0.946 0.066
Multiply 30.299 0.965 0.041 30.874 0.971 0.036
7.6. Visualization of Feature Planes.
We visualize each channel of XT;ZT feature plane for
opacity HexPlane in Figure 11. This HexPlane is trained in
Flame Salmon scene in Plenoptic Video Dataset [31].
8. Failure Cases
HexPlane doesn’t always give satisfactory results. It
generates degraded results when objects move too fast or
there are too few observations to synthesis details. Fig-
ure 12 shows failure cases and corresponding ground-truth
images.
9. Failed Designs for Dynamic Scenes
Although HexPlane is a simple and elegant solution, it
is not the instant solution we had for this task. In this sec-
tion, we discuss other designs we tried. These designs could
model the dynamic scenes while their qualities and speeds
are not comparable to HexPlane. We discuss these “failed”
designs, hoping they could inspire future work.
9.1. Fixed Basis for Time Axis
In Eq 2 of main paper, we use f(t)as the coefﬁcients of
basis volumes at time t. Its could be further expressed as:
Vt=RtX
i=1f(t)i^Vi=^Vf(t) (11)
where the second is matrix-vector production; ^V2
RXYZFR tis the stack off^V1;:::; ^VRtg;f(t)2RRtis
a function of t. An interesting perspective to understand ^V
is: instead of storing static features with shape RF, every
spatial point in 3D volume contains a feature matrix with
shapeRFRt. And feature vectors at speciﬁc time tcould
be computed by inner product between f(t)and feature ma-
trix. That is, f(t)is a set of basis functions w.r.t to time t
and feature matrix contains coefﬁcients of basis functionsto approximate feature value changes along with time. Fol-
lowing the traditional approach of basis functions for time
series, we use a set of sine/cosine functions as f(t).
While in practice, we found this implementation
couldn’t work since it requires enormous GPU memories.
For instance, with X=Y=Z= 128;Rt= 32;F= 27 ,
it uses 7GB to store such a representation and around 30GB
during training because of back-propagation and keeping
auxiliary terms of Adam. And it is extremely slow because
of reading/writing values in memories.
Therefore, we apply tensor decomposition to reduce
memory usages by factorizing volumes into matrixes
MXY;MXZ;MYZand vectors vX;vY;vZfollowing Eq
1. Similarly, we add additional Rtdimension to matrixes
Mand vectors v, leading to MTXY
r2RXYR T;vTX
r2
RXRT. When calculating features from the representation,
f(t)is ﬁrst multiplied with MTXY
iandvTX
ialong the last
dimension to get Mandvat this time steps. We then cal-
culate features using resulted vandMfollowing Eq 1.
f(t)is designed to be like positional encoding, f(t) =
[1;sin(t);cos(t);sin(2t);cos(2t);sin(4t);cos(4
t);]. We also try to use Legendre polynomials or Cheby-
shev polynomials to represent f(t). During training, we
use weighted L1loss to regularize MTr;vTr, and assign
higher weights for high-frequency coefﬁcients to keep re-
sults smooth. We also use the smoothly bandwidth anneal-
ing trick in [53] during training, which gradually introduc-
ing high-frequency components.
This design could model dynamic scenes while it suf-
fers from severe color jittering and distortions. Compared to
HexPlane, it has additional matrix-vector production, which
reduces overall speeds.
9.2. Frequency-Domain Methods
We also tried another method from the frequency do-
main, which is orthogonal to the HexPlane idea. The no-
tations of this section are slightly inconsistent with the no-

---

## Page 19

Feature Map Visualization on XY Plane
Feature Map Visualization on ZT Plane
Figure 11. Feature Map Visualization on Flame Salmon Scene.

---

## Page 20

Synthesis Ground-truth Synthesis Ground-truth
Figure 12. Failure Cases from HexPlane.
tations of the main paper.
According to Fourier Theory, the value at (x;y;z;t )
spacetime point could be represented in its frequency do-
main (we ignore feature dimension here for simplicity):
D(x;y;z;t ) =UX
u=1VX
v=1WX
w=1KX
k=1
eD(u;v;w;k )e j2(ux
U+vy
V+wz
W+kt
K)
(12)
eDis another 4D volume storing frequency weights, having
the same size as D. Storing eDis memory-consuming, andsimilarly, we apply tensor decomposition on this volume.
D(x; y; z; t ) =UX
u=1VX
v=1WX
w=1KX
k=1RX
r=1evU(u)revV(v)r
evW(w)revK(k)re j2(ux
U+vy
V+wz
W+kt
K)
=UX
u=1VX
v=1WX
w=1KX
k=1RX
r=1(evU(u)re j2ux
U)(evV(v)re j2vy
V)
(evW(w)re j2wz
W)(evK(k)re j2kt
K))
=RX
r=1(UX
u=1evU(u)re j2ux
U)VX
v=1(evV(v)re j2vy
Y)
(WX
w=1evW(w)re j2wz
W)(KX
k=1evK(k)re j2kt
K)
(13)
whereevU;evV;evW;evKare the decomposed vectors from
eDalongU;V;W;K axes using CANDECOMP Decompo-
sition, which axes are related to x;y;z;t in time domain.

---

## Page 21

Instead of storing the 4D frequency volume and computing
values by traversing all elements inside this volume using
Eq 5, we decompose 4D volumes into many single vectors
and calculate values by summation along each axis, signiﬁ-
cantly reducing computations.
Similarly, we apply weight L1and smoothly bandwidth
annealing trick on vector weights. We also try wavelet se-
ries instead of Fourier series, and other decompositions. We
found this method leads to less-saturated colors and de-
graded details, which is shown in videos.
Also, this method replaces grid sampling of HexPlane
by inner product, which is less efﬁcient and leads to slow
speeds.