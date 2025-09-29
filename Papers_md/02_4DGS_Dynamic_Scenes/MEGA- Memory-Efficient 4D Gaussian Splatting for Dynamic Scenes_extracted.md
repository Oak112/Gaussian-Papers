

---

## Page 1

MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes
Xinjie Zhang1* Zhening Liu1Yifan Zhang2†Xingtong Ge1Dailan He3
Tongda Xu5Yan Wang5Zehong Lin1Shuicheng Yan4,2†Jun Zhang1†
1iComAI Lab, The Hong Kong University of Science and Technology2Skywork AI
3The Chinese University of Hong Kong4National University of Singapore
5Institute for AI Industry Research (AIR), Tsinghua University
{xzhangga, zhening.liu }@connect.ust.hk, yifan.zhang7@kunlun-inc.com, xingtong.ge@gmail.com
hedailan@link.cuhk.edu.hk, x.tongda@nyu.edu, wangyan202199@163.com, eezhlin@ust.hk
yansc@comp.nus.edu.sg, eejzhang@ust.hk
Abstract
4D Gaussian Splatting (4DGS) has recently emerged as
a promising technique for capturing complex dynamic 3D
scenes with high fidelity. It utilizes a 4D Gaussian rep-
resentation and a GPU-friendly rasterizer, enabling rapid
rendering speeds. Despite its advantages, 4DGS faces sig-
nificant challenges, notably the requirement of millions of
4D Gaussians, each with extensive associated attributes,
leading to substantial memory and storage cost. This pa-
per introduces a memory-efficient framework for 4DGS.
We streamline the color attribute by decomposing it into a
per-Gaussian direct color component with only 3 param-
eters and a shared lightweight alternating current color
predictor. This approach eliminates the need for spherical
harmonics coefficients, which typically involve up to 144
parameters in classic 4DGS, thereby creating a memory-
efficient 4D Gaussian representation. Furthermore, we in-
troduce an entropy-constrained Gaussian deformation tech-
nique that uses a deformation field to expand the action
range of each Gaussian and integrates an opacity-based
entropy loss to limit the number of Gaussians, thus forc-
ing our model to use as few Gaussians as possible to fit
a dynamic scene well. With simple half-precision storage
and zip compression, our framework achieves a storage
reduction by approximately 190 ×and 125 ×on the Tech-
nicolor and Neural 3D Video datasets, respectively, com-
pared to the original 4DGS. Meanwhile, it maintains com-
parable rendering speeds and scene representation quality,
setting a new standard in the field. Code is available at
https://github.com/Xinjie-Q/MEGA .
*This work was partially performed when Xinjie Zhang was an Intern
at Skywork AI.
†Corresponding Authors.1. Introduction
Dynamic scene reconstruction from multi-view videos is
gaining widespread interest in computer vision and graph-
ics due to its broad applications in virtual reality (VR), aug-
mented reality (AR), and 3D content production. The emer-
gence of neural radiance field (NeRF) [31] enables high-
quality novel view synthesis from multi-view image inputs.
It has been further extended to represent dynamic scenes
by modeling a direct mapping from spatio-temporal coordi-
nates to color and density [3, 22, 36]. Despite the impressive
visual quality of NeRF-based methods, they require dense
sampling along rays, leading to slow rendering speeds that
hinder practical applications.
The recent introduction of 3D Gaussian Splatting
(3DGS) [17] marks a significant shift in the field of
novel view synthesis. This approach incorporates the ex-
plicit 3D Gaussian representation and differentiable tile-
based rasterization to enable real-time rendering speeds
that significantly outperform NeRF-based methods. Built
on this framework, subsequent studies have developed 4D
Gaussian Splatting (4DGS) [10, 46], which conceptualizes
scene variations across different timestamps as 4D spatio-
temporal Gaussian hyper-cylinder. As shown in Fig. 2,
when depicting a 3D scene at a given timestamp, these 4D
Gaussians will first be sliced into 3D Gaussians with time-
varying positions and opacity. Then, the 3D Gaussians with
the temporal decay opacity below a specific threshold are
filtered out. This filtering operation helps 4DGS to describe
the transient content such as emerging or vanishing objects.
Finally, following 3DGS, the remaining 3D Gaussians are
projected onto 2D screens through fast rasterization. By
directly optimizing a collection of 4D Gaussians, 4DGS ef-
fectively captures both static and dynamic scene elements,
thereby achieving photorealistic visual quality.
However, 4DGS requires millions of Gaussians to ad-arXiv:2410.13613v3  [cs.CV]  22 Jul 2025

---

## Page 2

PSNR: 31.00dB, Mem: 7.79GB PSNR: 32.02dB, Mem: 31.42MB40FPS61FPS254×Compression4DGSOurs(a) High performance at the Birthday scene.
Ours
STG
DyNeRFHyperReelE-D3DGS
4DGS
Deformable 3DGS
(b) Comparison on quality, size, and speed.
Figure 1. Our approach significantly reduces storage requirements
while maintaining comparable photorealistic quality and real-time
rendering speed with 4D Gaussian Splatting (4DGS) [46]. The
core idea is to develop a memory-efficient 4D Gaussian represen-
tation and use as few Gaussians as possible to fit dynamic scenes
well. (a) 4DGS requires up to 13 million Gaussians to render
theBirthday scene, whereas our method only needs 0.91 million
Gaussians. (b) Quantitative comparisons of rendering quality, stor-
age size, and speed against various competitive baselines on the
Technicolor dataset.
equately represent dynamic scenes with high fidelity. As
depicted in Fig. 1 (a), rendering the Birthday scene necessi-
tates up to 13 million Gaussian points, leading to a stor-
age overhead of approximately 7.79GB. This substantial
storage and transmission challenge can severely limit the
practical applications of 4DGS, particularly on resource-
constrained devices. For example, the significant memory
requirements may make it impractical to store, transmit, and
render various scenes on AR/VR headsets. Consequently, it
is of critical importance to compress 4D Gaussians to mini-
mize the memory footprint of 4DGS while preserving high-
quality scene representation and reconstruction.
To address the significant memory and storage chal-
lenges associated with 4DGS, we propose a Memory-
Efficient 4D Gaussian Splatting (MEGA) framework. In
the original 4D Gaussian representation, 144 out of the to-
FilterTemporal SlicingFigure 2. Illustration of temporal slicing in 4DGS, with the z-axis
omitted for simplicity. A 4D Gaussian can be conceptualized as
a hyper-cylinder in 4D space. Given the specific time query, a
corresponding 3D Gaussian ellipsoid is extracted from this hyper-
cylinder. The depth of color in the 3D Gaussian ellipsoid rep-
resents its temporal opacity. Those 3D Gaussian ellipsoids with
temporal opacity below a predefined threshold are excluded from
the scene rendering.
tal 161 parameters are 4D spherical harmonics (SH) coeffi-
cients, which occupy the majority of the storage space and
exhibit considerable redundancy. To develop a memory-
efficient 4D Gaussian representation, we draw inspiration
from the concepts of Direct Current (DC) and Alternating
Current (AC) in electrical engineering, which symbolize the
steady and varying components, respectively. Specifically,
we decouple the color attribute into a per-Gaussian DC
color component and a shared temporal-viewpoint aware
AC color predictor. This predictor is capable of accurately
estimating the color variations of a Gaussian at given times
and viewing angles, thereby effectively preserving visual
quality. It is noteworthy that our DC color component
requires only 3 parameters, while the predictor utilizes a
lightweight multi-layer perceptron (MLP) with three linear
layers. Consequently, this modification achieves a compres-
sion ratio of approximately 8×relative to the original 4D
Gaussians with equivalent Gaussian points, substantially re-
ducing the storage demands of the Gaussian representation.
Nevertheless, compacting the properties of the 4D Gaus-
sian alone cannot effectively alleviate the problem of exces-
sive number of Gaussians required. Existing 4DGS base-
lines [10, 46] assume that each sliced 4D Gaussian ex-
hibits only linear movement over time while maintaining
constant covariance, which means that the complex motion
in the scene has to be modeled by a combination of multi-
ple Gaussians. Moreover, as illustrated in Fig. 4 (a), only
about 6% of Gaussians actively participate in rendering at
any given time, because the temporal decay opacity forces
each Gaussian to be visible only near its mean time cen-
ter and invisible at other times. These inherent properties
significantly limit the effective utilization of each Gaussian,
thereby increasing the number of Gaussians needed for ad-
equate scene rendering. To overcome this limitation, we
introduce an efficient entropy-constrained Gaussian defor-
mation field designed to expand the operational range of 4D
Gaussians. This deformation model leverages both tempo-

---

## Page 3

ral and viewpoint information to accurately represent Gaus-
sian motion, shape, and transience changes. Meanwhile, a
spatial opacity-based entropy loss is introduced to push the
spatial opacity of each Gaussian towards binary states (ei-
ther one or zero). This adjustment aids in identifying and
eliminating non-essential Gaussians that contribute mini-
mally to the overall performance. In this way, our proposed
strategy not only effectively reduces the number of Gaus-
sians, but also improves the utilization rate of the Gaussians
involved in rendering given the time and viewing angle.
Finally, to store the parameters of our streamlined 4DGS,
we employ 16-bit floating-point (FP16) precision with zip
delta compression algorithm to achieve further reductions
in memory footprint. In summary, our main contributions
are three-fold:
• To the best of our knowledge, we are among the first
to develop a memory-efficient framework for 4D Gaus-
sian Splatting. By decomposing the color attribute into
a per-Gaussian DC color component and a lightweight,
temporal-viewpoint aware AC color predictor, we suc-
cessfully eliminate the need for redundant spherical har-
monics coefficients.
• We introduce an entropy-constrained Gaussian deforma-
tion technique to enhance the potential of each 4D Gaus-
sian for depicting complex scene motion. This approach
not only substantially reduces the number of Gaussians
but also improves their utilization rate. Moreover, we in-
tegrate straightforward post-processing techniques, such
as FP16 precision and zip delta compression, to further
decrease storage overhead.
• Extensive experimental results demonstrate that our
proposed method achieves significant storage reduc-
tions—approximately 190×and125×on the Technicolor
and Neural 3D Video datasets, respectively—while main-
taining comparable quality of scene representation and
rendering speed relative to the original 4DGS.
2. Related Works
Neural Rendering for Static Scenes. Recently, the advent
of neural rendering has attracted increasing interest in 3D
scene representation and reconstruction. NeRF, pioneered
by [31], represents the volume density and view-dependent
emitted radiance of a 3D scene as a function of 5D coor-
dinates (3D position and 2D viewing direction) using an
MLP. However, the vanilla NeRF relies solely on a large
MLP to store scene information, significantly limiting its
training and rendering efficiency. Subsequent works have
explored explicit grid-based representations [4, 12, 32, 40]
to enhance training efficiency. Nonetheless, these NeRF-
based methods still face challenges of slow rendering due
to dense sampling for each ray. In contrast, the work [17]
introduce 3D Gaussian Splatting, a novel explicit represen-
tation framework that employs a highly optimized customCUDA rasterizer to achieve unparalleled rendering speeds
with high-fidelity novel view synthesis for complex scenes.
Subsequent studies have further improved the representa-
tion efficiency [7, 15, 20, 27, 48, 50] and have been ex-
tended to various vision understanding and editing applica-
tions [5, 26, 35, 38, 43, 49].
Neural Rendering for Dynamic Scenes. Synthesizing new
views of dynamic scenes from a series of 2D images cap-
tured at different times presents a significant challenge. Re-
cent advancements have extended NeRF to handle monoc-
ular or multi-object dynamic scenes by learning a map-
ping from spatio-temporal coordinates to color and density
[1, 3, 13, 21, 22, 24, 25, 30, 36, 39, 41]. Unfortunately,
these methods suffer from low rendering efficiency. To ad-
dress this issue, some recent studies [2, 9, 14, 28, 44, 45]
have developed deformable 3D GS, which decouples dy-
namic scenes into a static canonical 3DGS and a deforma-
tion motion field to account for temporal variations in the
3D Gaussian parameters. Concurrently, a series of recent
studies [10, 16, 18, 23, 46] directly learn a set of spatio-
temporal Gaussians to model static, dynamic, and transient
content within a scene. However, these methods require
a large number of Gaussians to achieve high-quality scene
modeling, which brings expensive storage overhead. To this
end, our work focuses on developing effective compression
techniques for 4DGS [46].
3D Gaussian Splatting Compression. Since optimized
scenes in 3DGS typically comprise millions of 3D Gaus-
sians and require up to several gigabytes of storage, various
compression strategies have been proposed to reduce the
size, including redundant Gaussian pruning [11, 20], spher-
ical harmonics distillation or compactness [11, 20, 34, 42],
vector quantization [11, 20, 33, 42], and entropy models [6].
However, due to the differences between 3DGS for static
scene representation and 4DGS for dynamic scene repre-
sentation, existing methods may be inapplicable to or un-
suitable for 4DGS. In this paper, we aim to develop a more
compact color representation and reduce the number of 4D
Gaussians by considering temporal and viewpoint factors,
thereby achieving a more efficient memory footprint.
3. Method
In Section 3.1, we first review the technique of 4DGS [46],
which serves as the foundation of our method. Subse-
quently, in Section 3.2, we introduce how to develop our
memory-efficient 4D Gaussian Splatting for modeling dy-
namic scenes. Finally, we detail the training process and
describe how to store our compact 4DGS in Section 3.3.
3.1. Preliminary: 4D Gaussian Splatting
4D Gaussian Splatting [46] optimizes a set of anisotropic
4D Gaussians via differentiable rasterization to effectively
represent a dynamic scene. With a highly efficient ras-

---

## Page 4

(a) Original 4D Gaussian
(b) Our Memory-Ef ficient 4D Gaussian(c) Per -Gaussian Transformation
Deformation
PredictorAC Color
Predictor
Time query
View query
Transformed
4D Gaussian(d) Rendering Process
Differentiable RasterizationProjectionTemporal Slicing
Per-Gaussian TransformationA Set of Memory-
Efficient 4D Gaussians 4D GaussianFigure 3. Overview of our proposed memory-efficient Gaussian Splatting framework. (a) The original 4D Gaussian uses 4D spherical
harmonics hto represent color, which is highly redundant and consumes substantial memory. (b) Our memory-efficient 4D Gaussian
replaces hwith a compact, view-independent, and time-independent color component cdc, achieving an about 8 ×reduction in storage
overhead. (c) In the per-Gaussian transformation, a lightweight AC color predictor compensates for the absent viewpoint and temporal
information in cdc, and a deformation predictor expands the action range of each Gaussian. (d) Our rendering process consists of four
steps: per-Gaussian transformation, temporal slicing, projection, and differentiable rasterization.
terizer, the optimized model facilitates real-time render-
ing of high-fidelity novel views. Each 4D Gaussian is
characterized by the following attributes: (i) 4D center
µ4D= (µx, µy, µz, µt)T∈R4; (ii) 4D rotation R4D
represented by a pair of left quaternion ql∈R4and
right quaternion qr∈R4; (iii) 4D scaling factor s4D=
(sx, sy, sz, st)T∈R4; (iv) time- and view-dependent RGB
color represented by 4D spherical harmonics coefficients
h∈R3(kv+1)2(kt+1)with the view degrees of freedom
kvand time degress of freedom kt; (v) spatial opacity
o∈[0,1].
Given 4D scaling matrix S4D=diag(s4D)and 4D ro-
tation matrix R4D, we parameterize 4D Gaussian’s covari-
ance matrix as:
Σ4D=R4DS4DST
4DRT
4D=U V
VTW
, (1)
where U∈R3×3. When rendering the scene at time t, each
4D Gaussian is sliced into 3D space. The density of the
sliced 3D Gaussian at the spatial point xis expressed as:
G3D(x, t) =σ(t)e−1
2[x−µ3D(t)]TΣ−1
3D[x−µ3D(t)],(2)
where Σ3D=U−VVT
Wrepresents the time-invariant 3D
covariance matrix. The temporal decay opacity, σ(t) =
e−(t−µt)2
2W, utilizes a 1D Gaussian function to modulate
the contribution of each Gaussian to the t-th scene. The
time-variant 3D center, µ3D(t) =µ3D+ (t−µt)V
W, in-
troduces a linear motion term to the 3D center positionµ3D= (µx, µy, µz)T, assuming that all motions can be ap-
proximated as linear motion within a very small time range.
After temporal slicing, the following process involves pro-
jecting sliced 3D Gaussians onto the 2D image plane based
on depth order from specific view direction, and executing
the fast differentiable rasterization to render the final image.
Although this paradigm provides high-quality novel view
synthesis, it necessitates large amount of Gaussians to fully
reconstruct a dynamic scene, which brings unbearable stor-
age overhead. This challenge drives our memory-efficient
4D Gaussian Splatting design.
3.2. Memory-efficient 4D Gaussian Splatting
Overview. As illustrated in Fig. 3, we develop our memory-
efficient 4D Gaussian framework to significantly reduce the
number of per-Gaussian stored parameters and drive the
model to reconstruct dynamic scene with fewer 4D Gaus-
sians. During the rendering process, we utilize a set of opti-
mized 4D Gaussians and initially transform each Gaussian
based on specific time and view direction. This transforma-
tion procedure involves Gaussian color prediction and ge-
ometry deformation. By modifying the geometric structure
of each Gaussian, we effectively broaden its action range.
This expansion not only reduces the total number of Gaus-
sians required but also increases the rendering participation
rate of each Gaussian. Following the per-Gaussian transfor-
mation, we adhere to the established protocols of the origi-
nal 4DGS [46] to carry out temporal slicing, projection, and

---

## Page 5

0 10 20 30 40 50
Time Step01020304050607080Participation Ratio (%)
Birthday Scene
4DGS
Ours (Before Transformation)
Ours (After Transformation)(a)
0 5 10 15 20 25 30
Iteration (K)0.00.20.40.60.81.01.21.41.6Number of Gaussians1e7 Birthday Scene
Ours w/o opa (PSNR: 31.35)
Ours               (PSNR: 32.02) (b)
Figure 4. (a) The ratio of Gaussians involved in rendering the Birthday scene at different time steps. The blue line shows how many
Gaussians are involved in rendering in our MEGA model if we do not use per-Gaussian transformation. (b) Visualization of the varying
number of Gaussians on the Birthday scene during training.
differentiable rasterization, all critical for rendering high-
quality frames.
Memory-efficient 4D Gaussian. 4DGS introduces 4D
spherical harmonics hto model the temporal evolution of
view-dependent color in dynamic scenes, which typically
requires 144 of the total 161 parameters and contributes to
the main storage overhead. While [20] have explored the
use of a grid-based neural field to replace SH coefficients
h, we find that directly applying this method results in se-
vere performance loss compared to 4DGS (see Table 3).
To overcome this issue, we propose a compact DC-AC
color (DAC) representation. Specifically, we decouple the
color attribute as a per-Gaussian DC color component cdc∈
R3and a temporal-viewpoint aware AC color predictor Fϕ.
To predict the final color ct,vof each Gaussian, we first
compute the normalized view direction dv=µ3D−pv
||µ3D−pv||2
for each Gaussian according to the camera center point
pv∈R3at the viewpoint v. Then, we concatenate the 3D
position µ3D, view direction di, time t, and DC color cdc
and input them to a lightweight MLP network Fϕ:
ct,v= sigmoid( cdc+Fϕ(sg(µ3D),sg(dv), t,cdc)),(3)
where sg(·)indicates a stop-gradient operation. This hybrid
color composition method not only effectively preserves the
individual information using DC component and supple-
ments the missing viewpoint and time information using the
AC predictor to maintain high rendering quality, but also
reduces the storage overhead by up to 8×compared to the
original 4DGS [46].
Entropy-constrained Gaussian Deformation. For a spe-
cific time t, 4DGS [46] presupposes that the sliced 4D
Gaussians exhibit linear movement while their rotation and
scale remain constant. This strict assumption simplifiesthe movement representation and forces the model to com-
bine multiple extra Gaussians to present any complex non-
linear motions. Moreover, the sliced 4D Gaussian intro-
duces the temporal decay opacity σt. From its definition,
it is analyzed that a Gaussian gradually appears as time
tapproaches its temporal position µt, peaks in opacity at
t=µt, and gradually diminishes in density as tmoves away
from µt. As shown in Fig. 4 (a), this limited temporal op-
eration range results in more than 90% of Gaussians being
excluded at each time, causing the model to densify a large
amount of Gaussians for rendering high-quality scene.
To address these limitations, we advocate for improv-
ing flexibility in the motion representation and geometric
structure of each 4D Gaussian. Specifically, we introduce
a temporal-viewpoint aware deformation predictor to en-
large the action range of Gaussians. The 4D Gaussian
center µ4D, view direction di, and time tare mapped to
a high-dimensional space using a regular frequency posi-
tional encoding function γ[31], and then processed through
a lightweight MLP network Fθto predict the position de-
formation mt,v
µ4D∈R4, scale deformation mt,v
s4D∈R4, and
rotation deformations mt,v
ql∈R4,mt,v
qr∈R4as:
(mt,v
µ4D,mt,v
s4D,mt,v
ql,mt,v
qr) =
Fθ(γ(sg(µ4D)), γ(sg(dv)), γ(t)),(4)
where γis defined as (sin(2lπp),cos(2lπp))L−1
l=0. Based on
the estimated deformation for time tand viewpoint v, we
transform the original 4D Gaussian to a temporal-viewpoint
aware 4D Gaussian:
µt,v
4D=µ4D×mt,v
µ4D,st,v
4D=s4D×mt,v
s4D,
qt,v
l=ql×mt,v
ql, qt,v
r=qr×mt,v
qr.(5)

---

## Page 6

Table 1. Quantitative comparison with various competitive baselines on the Technicolor Dataset. “Storage” refers to the total model size
for 50 frames.
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
DyNeRF [22] 31.80 - 0.0210 0.1400 0.02 30.00MB
HyperReel [1] 32.70 0.0470 - 0.1090 4.00 60.00MB
Deformable 3DGS [44] 30.95 0.0696 0.0353 0.1553 76.09 61.36MB
STG [23] 33.35 0.0404 0.0187 0.0846 141.73 51.35MB
E-D3DGS [2] 32.89 0.0494 0.0231 0.1114 79.14 56.07MB
4DGS [46] 32.07 0.0535 0.0263 0.1189 55.26 6107.07MB
Ours 33.57 0.0442 0.0204 0.1014 83.14 32.45MB
Nonetheless, as depicted in Fig. 4 (b), without constraints
on the number of Gaussians, a significant proliferation oc-
curs where Gaussians are continuously split and cloned dur-
ing the densification process. To force the model to use
fewer Gaussians while accurately simulating complex scene
changes, we introduce a spatial opacity-based entropy loss
Lopathat encourages the spatial opacity oof each Gaussian
to approach one or zero:
Lopa=1
NNX
j=1(−ojlog(oj)), (6)
where Ndenotes the number of Gaussians. During op-
timization, we actively prune Gaussians that exhibit near-
zero opacity at every Kiterations, which ensures efficient
computation and maintains a low storage footprint through-
out the training phase. Furthermore, as shown in Fig. 4 (a),
with the opacity-based entropy loss Lopa, our deformation
field successfully enlarges the action range of each Gaus-
sian, increasing the Gaussian participation ratio from less
than 50% to about 75% under the same Gaussian points.
3.3. Training and Compression Pipeline
Loss Function. Following the original 4DGS [46], we
adopt the photometric loss Lphoto , consisting of L1loss and
structural similarity loss Lssim, to measure the distortion
between the rendered image and ground truth image. By
adding the loss for opacity regularization Lopa, the overall
lossLis defined as:
L=Lphoto +κLopa= (1−λ)L1+λLssim+κLopa,(7)
where both λandκare trade-off parameters to balance the
components.
Compression Pipeline. During the optimization phase, we
adopt half-precision training. After obtaining the optimized
MEGA representation, we store these learnable parameters
in the FP16 format, then apply the zip delta compression
algorithm. This lossless compression technique typically
reduces storage overhead by approximately 10%.4. Experiments
4.1. Experimental Setup
Datasets. We evaluate the effectiveness of our method us-
ing two real-world benchmarks that are representative of
various challenges in dynamic scene rendering: (1) Tech-
nicolor Light Field Dataset [37]: This dataset consists of
multi-view video data captured by a time-synchronized 4 ×4
camera rig. Following HyperReel [1], we exclude the cam-
era at the second row, second column and evaluate on five
scenes ( Birthday ,Fabien ,Painter ,Theater , and Trains ) at
2048×1088 full resolution. (2) Neural 3D Video Dataset
(Neu3DV) [22]: This dataset includes six indoor multi-view
video scenes captured by 18 to 21 cameras, each at a reso-
lution of 2704×2028 pixels. The scenes ( Coffee Martini ,
Cook Spinach ,Cut Roasted Beef ,Flame Salmon ,Flame
Steak ,Sear Steak ) vary in duration and feature dynamic
movements, some with multiple objects in motion. Con-
sistent with existing practices [22, 46], evaluations are con-
ducted at half resolution of 300-frame scenes.
Evaluation Metrics. To assess the quality of rendered
videos, we utilize three popular image quality assessment
metrics: Peak Signal-to-Noise Ratio (PSNR), Dissimilarity
Structural Similarity Index Measure (DSSIM), and Learned
Perceptual Image Patch Similarity (LPIPS) [47]. PSNR
quantifies the pixel color error between the rendered and
original frames. DSSIM evaluates the perceived dissim-
ilarity of the rendered image, while LPIPS measures the
higher-level perceptual similarity using an AlexNet back-
bone [19]. Given the inconsistency in DSSIM implementa-
tion noted across different methods [1, 13], we follow [23]
to distinguish DSSIM results into two categories: DSSIM 1
and DSSIM 2. DSSIM 1is calculated with a data range set
to 1.0, based on the structural similarity function from the
scikit-image library, whereas DSSIM uses a data range of
2.0. For rendering speed, we measure the performance in
frames per second (FPS).
Baselines. As we introduce MEGA, a novel method for
compressing 4DGS [46], our primary comparison focuses
on the baseline 4DGS method. Additionally, we bench-
mark MEGA against a range of NeRF-based baselines,
including DyNeRF [22], HyperReel [1], Neural V olume

---

## Page 7

Table 2. Quantitative comparisons with various competitive baselines on the Neural 3D Video Dataset. “Storage” refers to the total model
size for 300 frames.1: Only report the result on the Flame Salmon scene.2: Exclude the Coffee Martini scene.3: These methods train
each model with a 50-frame video sequence to prevent memory overflow, requiring six models to complete the overall evaluation.
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
Neural V olume1[25] 22.80 - 0.0620 0.2950 - -
LLFF1[30] 23.24 - 0.0200 0.2350 - -
DyNeRF1[22] 29.58 - 0.0200 0.0830 0.015 28.00MB
HexPlane2,3[3] 31.71 - 0.0140 0.0750 0.56 200.00MB
StreamRF [21] 28.26 - - - 10.90 5310.00MB
NeRFPlayer3[39] 30.69 0.0340 - 0.1110 0.05 5130.00MB
HyperReel [1] 31.10 0.0360 - 0.0960 2.00 360.00MB
K-Planes [13] 31.63 - 0.0180 - 0.30 311.00MB
MixV oxels-L [41] 31.34 - 0.0170 0.0960 37.70 500.00MB
MixV oxels-X [41] 31.73 - 0.0150 0.0640 4.60 500.00MB
Dynamic 3DGS [29] 30.46 0.0350 0.0190 0.0990 460.00 2772.00MB
C-D3DGS [16] 30.46 - - 0.1500 118.00 338.00MB
Deformable 3DGS [44] 30.98 0.0331 0.0191 0.0594 29.62 32.64MB
E-D3DGS [2] 31.20 0.0259 0.0151 0.0304 69.70 40.20MB
STG3[23] 32.04 0.0261 0.0145 0.0440 273.47 175.35MB
4DGS [46] 31.57 0.0290 0.0164 0.0573 96.69 3128.00MB
Ours 31.49 0.0290 0.0165 0.0568 77.42 25.05MB
[25], LLFF [30], HexPlane [3], StreamRF [21], NeRF-
Player [39], MixV oxels [41], and K-Planes [13]. Other re-
cent competitive Gaussian-based methods are also consid-
ered in our comparisons, including Dynamic 3DGS [29], C-
D3DGS [16], Deformable 3DGS [44], E-D3DGS [2], and
STG [23]. The numerical results of Deformable 3DGS, E-
D3DGS, STG, and 4DGS are produced by running their re-
leased codes on a single NVIDIA A800 GPU, while results
for other baselines are from their original papers.
Implementation Details. We train our MEGA model over
30k iterations and stop densification at the midpoint. We
use the Adam optimizer with a batch size of one, adopting
the hyperparameter settings from the original 4DGS [46]
framework, including loss weight, learning rate, and thresh-
old parameters. When rendering the view at time t, we fil-
ter out those Gaussians with σ(t)≤0.05. To ensure stable
training of our deformation predictor, we introduce weight
regularization and set it at 1e−6. The learning rate of the
deformation predictor undergoes exponential decay, start-
ing from 8e−4and reducing to 1.6e−6. For the AC color
predictor, we start with an initial learning rate of 0.01, in-
corporating a 100-step warm-up phase. Subsequently, its
learning rate is decreased by a factor of three at the 5k, 15k,
and 25k steps. Regarding the hyper-parameters in the loss
function, we set λandκas 0.2 and 0.0005, respectively, to
balance the contributions of different components.
4.2. Experimental Results
Table 1 details a quantitative evaluation of our MEGA
method on the Technicolor dataset. Notably, our method
surpasses the main baseline 4DGS [46], with PSNR,
GT Deformable 3DGS E-D3DGS
STG 4DGS Ours
Figure 5. Subjective comparison of various methods on Theater
scene from the Technicolor Dataset.
DSSIM 1, DSSIM 2, and LPIPS improvements by 1.2dB,
0.01, 0.006, and 0.018, respectively. Meanwhile, it sig-
nificantly reduces storage requirements, achieving a 190 ×
compactness and improving rendering speed by 50%.
When compared with the NeRF-based method HyperReel
[1], MEGA achieves a substantial improvement in represen-
tation, with an increase of about 0.87dB in PSNR and a 20 ×
faster rendering speed, while halving the storage overhead.
Moreover, our MEGA records a 0.22dB gain in visual fi-
delity over the state-of-the-art Gaussian-based method STG
[23], and reduces storage overhead by 40%. Fig. 5 offers
qualitative comparisons for the Theater scene, demonstrat-
ing that our results contain more vivid details and provide
artifact-less rendering.

---

## Page 8

Table 3. Ablation study of the proposed components. Ndenotes the number of Gaussians. The last row represents our final solution.
(a) Technicolor Dataset
VariantsBirthday Fabien
PSNR↑ DSSIM 1↓ N↓ Params ↓ PSNR↑ DSSIM 1↓ N↓ Params ↓
4DGS [46] 31.00 0.0383 13.00M 2093.56M 33.57 0.0582 5.43M 874.14M
w/ grid [20] 30.49 0.0410 16.33M 293.07M 32.99 0.0620 4.61M 93.77M
w/ DAC 31.60 0.0355 15.43M 308.65M 34.21 0.0587 4.57M 91.48M
w/ DAC+Deformation 31.35 0.0368 15.75M 315.36M 33.02 0.0604 11.56M 231.53M
w/ DAC+ Lopa 31.46 0.0370 9.15M 183.23M 33.96 0.0603 2.32M 46.40M
w/ DAC+Deformation+ Lopa 32.02 0.0309 0.91M 18.48M 34.89 0.0597 0.31M 6.43M
(b) Neural 3D Video Dataset
VariantsFlame Steak Sear Steak
PSNR↑ DSSIM 1↓ N↓ Params ↓ PSNR↑ DSSIM 1↓ N↓ Params ↓
4DGS [46] 33.19 0.0204 5.17M 831.88M 33.44 0.0204 3.52M 567.30M
w/ grid [20] 31.07 0.0279 4.82M 97.35M 31.313 0.0281 3.25M 70.76M
w/ DAC 33.34 0.0210 5.31M 106.33M 33.67 0.0206 3.61M 72.18M
w/ DAC+Deformation 33.47 0.0209 6.34M 127.16M 33.46 0.0208 4.17M 83.78M
w/ DAC+ Lopa 33.45 0.0208 2.76M 55.22M 33.58 0.0215 1.99M 39.74M
w/ DAC+Deformation+ Lopa 32.27 0.0242 0.87M 17.79M 33.67 0.0200 0.56M 11.50M
Besides, we report the quantitative comparisons on the
Neu3DV dataset in Table 2. Relative to 4DGS, our method
achieves up to a 125 ×compression ratio while preserving
similar visual quality and rendering speed. It is observed
that compared to the SOTA NeRF-based baseline MixV ox-
els [41], our method achieves a 20 ×storage reduction and
a 16×inference speed improvement, maintaining compara-
ble rendering quality. Furthermore, our approach exhibits
higher rendering quality and smaller storage overhead com-
pared to most Gaussian-based methods.
4.3. Ablation Study
To validate the effectiveness of various components within
our proposed method, we conduct ablation experiments on
on scenes from the Technicolor dataset ( Birthday ,Fabien )
and the Neu3DV dataset ( Flame Steak ,Sear Steak ). De-
tailed results are presented in Table 3.
Compact DC-AC Color Representation. Building on the
original 4DGS, we substitute the 4D SH coefficients with
a grid-based neural field representation [20], and our pro-
posed DAC representation, respectively. While the grid-
based approach, referred to as “w/ grid,” achieves a reduc-
tion of approximately 10 ×in parameters, it leads to a sig-
nificant performance degradation compared to 4DGS. This
performance loss may be attributed to the grid’s inability
to retain sufficient detail, thereby discarding critical infor-
mation. To address this issue, we use a DC component to
preserve essential color information inherently present in
the scene, and an AC predictor to encode the temporal-
viewpoint variations in color. This method allows us to
achieve a comparable reduction in storage as the grid-based
approach while maintaining high-quality rendering consis-
tent with 4DGS.
Entropy-constrained Gaussian Deformation. This part
of our ablation study evaluates the impact of Gaussian de-
formation and opacity-based entropy loss Lopa. Startingfrom the configuration “w/ DAC”, we observe that imple-
menting a deformation predictor alone (referred to as “w/
DAC+Deformation”) leads to an increased number of Gaus-
sians. Conversely, employing Lopawithout the deforma-
tion predictor (referred to as “w/ DAC+ Lopa”) limits the ac-
tion range of each Gaussian, inhibiting their efficacy. How-
ever, when combining our deformation predictor with Lopa,
this strategy significantly reduces the number of Gaussians
needed while maintaining rendering quality comparable to
that of 4DGS.
5. Conclusion
In this paper, we develop a novel, memory-efficient frame-
work tailored for 4D Gaussian Splatting. By decompos-
ing the color attribute into a per-Gaussian direct current
component and a shared, lightweight alternating current
color predictor, our approach significantly reduces the per-
Gaussian parameters without compromising performance.
Furthermore, to reduce redundancy among the 4D Gaus-
sians, we introduce entropy-constrained Gaussian deforma-
tion. This technique expands the action range of each Gaus-
sian to enhance the effective utilization rate, thereby en-
abling the model to render high-quality scenes with as few
Gaussians as possible. Extensive experimental results un-
derscore the efficacy of our approach, demonstrating more
than a hundredfold reduction in storage requirements while
maintaining high-quality reconstruction and real-time ren-
dering speeds in comparison to the original 4D Gaussian
Splatting. These advancements establish a new benchmark
in the field, combining high performance, compactness, and
real-time rendering capabilities.
Acknowledgments. This work was supported by the
Hong Kong Research Grants Council under the Areas of
Excellence scheme grant AoE/E-601/22-R and NSFC/RGC
Collaborative Research Scheme grant CRS HKUST603/22.

---

## Page 9

References
[1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael
Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil
Kim. HyperReel: High-fidelity 6-DoF video with ray-
conditioned sampling. In Conference on Computer Vision
and Pattern Recognition (CVPR) , 2023. 3, 6, 7, 13, 14
[2] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun
Bang, and Youngjung Uh. Per-gaussian embedding-based
deformation for deformable 3d gaussian splatting. In Euro-
pean Conference on Computer Vision , 2024. 3, 6, 7, 13, 14,
15
[3] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition ,
pages 130–141, 2023. 1, 3, 7, 14
[4] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance fields. In European
Conference on Computer Vision , pages 333–350. Springer,
2022. 3
[5] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xi-
aofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping
Liu, and Guosheng Lin. Gaussianeditor: Swift and control-
lable 3d editing with gaussian splatting. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 21476–21485, 2024. 3
[6] Yihang Chen, Qianyi Wu, Jianfei Cai, Mehrtash Harandi,
and Weiyao Lin. Hac: Hash-grid assisted context for 3d
gaussian splatting compression. In European Conference on
Computer Vision , 2024. 3
[7] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision , pages 370–386. Springer, 2024. 3
[8] Franc ¸ois Darmon and et al. Robust gaussian splatting. arXiv ,
2024. available online. 15
[9] Devikalyan Das, Christopher Wewer, Raza Yunus, Eddy
Ilg, and Jan Eric Lenssen. Neural parametric gaussians for
monocular non-rigid object reconstruction. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 10715–10725, 2024. 3
[10] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:
Towards efficient novel view synthesis for dynamic scenes.
InACM SIGGRAPH 2024 Conference Papers , pages 1–11,
2024. 1, 2, 3
[11] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3d
gaussian compression with 15x reduction and 200+ fps. In
European Conference on Computer Vision , 2024. 3
[12] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 5501–5510, 2022. 3
[13] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:Explicit radiance fields in space, time, and appearance. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 12479–12488, 2023. 3,
6, 7, 14
[14] Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, and
Houqiang Li. Motion-aware 3d gaussian splatting for effi-
cient dynamic scene reconstruction. In European Conference
on Computer Vision , 2024. 3
[15] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias
Zwicker, and Tom Goldstein. Speedy-splat: Fast 3d gaus-
sian splatting with sparse pixels and sparse primitives. In
Proceedings of the Computer Vision and Pattern Recognition
Conference , pages 21537–21546, 2025. 3
[16] Kai Katsumata, Duc Minh V o, and Hideki Nakayama. A
compact dynamic 3d gaussian representation for real-time
dynamic view synthesis. In European Conference on Com-
puter Vision , 2024. 3, 7, 14
[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics , 42
(4):139–1, 2023. 1, 3
[18] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis.
Dynmf: Neural motion factorization for real-time dynamic
view synthesis with 3d gaussian splatting. In European Con-
ference on Computer Vision , 2024. 3
[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
Imagenet classification with deep convolutional neural net-
works. In Advances in Neural Information Processing Sys-
tems, 2012. 6
[20] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park. Compact 3d gaussian representation for
radiance field. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 21719–
21728, 2024. 3, 5, 8
[21] Lingzhi Li, Zhen Shen, zhongshu wang, Li Shen, and Ping
Tan. Streaming radiance fields for 3d video synthesis. In
Advances in Neural Information Processing Systems , 2022.
3, 7, 14
[22] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 5521–5531, 2022. 1, 3,
6, 7, 13, 14
[23] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 8508–8520, 2024. 3,
6, 7, 13, 14, 15
[24] Zhening Liu, Yingdong Hu, Xinjie Zhang, Jiawei Shao, Ze-
hong Lin, and Jun Zhang. Dynamics-aware gaussian splat-
ting streaming towards fast on-the-fly training for 4d recon-
struction. arXiv preprint arXiv:2411.14847 , 2024. 3
[25] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel
Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural vol-
umes: Learning dynamic renderable volumes from images.

---

## Page 10

ACM Transactions on Graphics , 38(4):65:1–65:14, 2019. 3,
7, 14
[26] Jiahao Lu, Yifan Zhang, Qiuhong Shen, Xinchao Wang, and
Shuicheng YAN. Poison-splat: Computation cost attack on
3d gaussian splatting. In International Conference on Learn-
ing Representations , 2025. 3
[27] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 20654–20664, 2024. 3
[28] Zhicheng Lu, Xiang Guo, Le Hui, Tianrui Chen, Min Yang,
Xiao Tang, Feng Zhu, and Yuchao Dai. 3d geometry-aware
deformable gaussian splatting for dynamic view synthesis.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 8900–8910, 2024. 3
[29] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In International Conference
on 3D Vision , pages 800–809. IEEE, 2024. 7, 14
[30] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light field fusion: Practical view syn-
thesis with prescriptive sampling guidelines. ACM Transac-
tions on Graphics , 2019. 3, 7, 14
[31] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM , 65(1):99–106, 2021. 1,
3, 5
[32] Thomas M ¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. ACM Transactions on Graphics , 41
(4):1–15, 2022. 3
[33] KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi
Koohpayegani, and Hamed Pirsiavash. Compgs: Smaller and
faster gaussian splatting with vector quantization. In Euro-
pean Conference on Computer Vision , 2024. 3
[34] Simon Niedermayr, Josef Stumpfegger, and R ¨udiger West-
ermann. Compressed 3d gaussian splatting for accelerated
novel view synthesis. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
10349–10358, 2024. 3
[35] Long Peng, Anran Wu, Wenbo Li, Peizhe Xia, Xueyuan
Dai, Xinjie Zhang, Xin Di, Haoze Sun, Renjing Pei, Yang
Wang, et al. Pixel to gaussian: Ultra-fast continuous
super-resolution with 2d gaussian modeling. arXiv preprint
arXiv:2503.06617 , 2025. 3
[36] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer. D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
10318–10327, 2021. 1, 3
[37] Neus Sabater, Guillaume Boisson, Benoit Vandame, Paul
Kerbiriou, Frederic Babon, Matthieu Hog, Remy Gendrot,
Tristan Langlois, Olivier Bureller, Arno Schubert, et al.
Dataset and pipeline for multi-view light-field video. In Pro-ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition Workshops , pages 30–40, 2017. 6
[38] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Flashsplat:
2d to 3d gaussian splatting segmentation solved optimally. In
European Conference on Computer Vision , pages 456–472.
Springer, 2024. 3
[39] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele
Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerf-
player: A streamable dynamic scene representation with de-
composed neural radiance fields. IEEE Transactions on Visu-
alization and Computer Graphics , 29(5):2732–2742, 2023.
3, 7, 14
[40] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel
grid optimization: Super-fast convergence for radiance fields
reconstruction. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 5459–
5469, 2022. 3
[41] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei
Song, and Huaping Liu. Mixed neural voxels for fast multi-
view video synthesis. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision , pages 19706–
19716, 2023. 3, 7, 8, 14
[42] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jia-
jun Deng, Jiang Bian, and Zhibo Chen. End-to-end rate-
distortion optimized 3d gaussian representation. In European
Conference on Computer Vision , 2024. 3
[43] Yuxin Wang, Qianyi Wu, Guofeng Zhang, and Dan Xu.
Learning 3d geometry and feature consistent gaussian splat-
ting for object removal. In European Conference on Com-
puter Vision , pages 1–17. Springer, 2024. 3
[44] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 20310–20320, 2024.
3, 6, 7, 13, 14
[45] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pages 20331–20341, 2024. 3
[46] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-
time photorealistic dynamic scene representation and render-
ing with 4d gaussian splatting. In International Conference
on Learning Representations , 2024. 1, 2, 3, 4, 5, 6, 7, 8, 13,
14, 15
[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion, pages 586–595, 2018. 6
[48] Xinjie Zhang, Xingtong Ge, Tongda Xu, Dailan He, Yan
Wang, Hongwei Qin, Guo Lu, Jing Geng, and Jun Zhang.
Gaussianimage: 1000 fps image representation and compres-
sion by 2d gaussian splatting. In European Conference on
Computer Vision , pages 327–345. Springer, 2024. 3
[49] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang

---

## Page 11

Wang, and Achuta Kadambi. Feature 3dgs: Supercharging
3d gaussian splatting to enable distilled feature fields. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pages 21676–21685, 2024. 3
[50] Lingting Zhu, Guying Lin, Jinnan Chen, Xinjie Zhang,
Zhenchao Jin, Zhao Wang, and Lequan Yu. Large images
are gaussians: High-quality large image representation with
levels of 2d gaussian splatting. In Proceedings of the AAAI
Conference on Artificial Intelligence , pages 10977–10985,
2025. 3

---

## Page 12

MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes
Supplementary Material
A. Experimental Results
We provide the complete results on the Technicolor and
Neural 3D Video datasets in Table 4 and Table 5. More
visualizations are available in Fig. 6 and Fig. 7.
B. Network Structure
AC Color Predictor. Fig. 8 (a) shows the details of the AC
color predictor. After generating the AC color component
ct,v
ac, we combine the DC component cdcto produce the final
colorct,v.
Deformation Predictor. Fig. 8 (b) provides the details of
the deformation predictor. For the feature fusion module,
we apply two linear layers with ReLU activation function.
GT Deformable 3DGS E-D3DGS
STG 4DGS Ours
GT Deformable 3DGS E-D3DGS
STG 4DGS Ours
Figure 6. Subjective comparison of various methods on Cut
Roasted Beef scene (Top) and Sear Steak scene (Bottom) from
the Neural 3D Video Dataset.
GT
 Deformable 3DGS
 E-D3DGS
STG 4DGS Ours
GT Deformable 3DGS E-D3DGS
STG 4DGS Ours
GT Deformable 3DGS E-D3DGS
STG 4DGS Ours
Figure 7. Subjective comparison of various methods on Birthday
scene (Top), Trains scene (Medium) and Painter scene (Bottom)
from the Technicolor Dataset.

---

## Page 13

Linear (1 1, 64)
ReLU
Linear (64, 64)
ReLU
Linear (64, 64)
Sigmoid(a) AC Color Predictor
Linear (39, 256)
ReLU
Linear (256, 30)Linear (13, 256)
ReLU
Linear (256, 30)
Feature Fusion (104)
Feature Fusion(360)
Linear (256, 4) Linear (256, 4) Linear (256, 8)Linear (N, 256)
ReLU
Linear (256, 256)
ReLU (b) Deformation Predictor
Figure 8. The network structures of (a) AC color predictor, (b) Deformation predictor.
Table 4. Quantitative comparisons with various competitive baselines on the Technicolor Dataset.
Birthday Fabien
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
DyNeRF [22] 29.20 - 0.0240 0.0668 - - 32.76 - 0.0175 0.2417 - -
HyperReel [1] 29.99 0.0390 - 0.0531 - - 34.70 0.0525 - 0.1864 - -
Deformable 3DGS [44] 30.68 0.0440 0.0237 0.0775 52.83 90.61MB 33.33 0.0673 0.0273 0.1851 95.52 42.81MB
E-D3DGS [2] 31.88 0.0328 0.0172 0.0506 62.41 66.50MB 34.69 0.0612 0.0236 0.1689 124.71 20.02MB
STG [23] 31.65 0.0293 0.0156 0.0413 128.43 51.81MB 35.61 0.0468 0.0177 0.1140 138.03 40.23MB
4DGS [46] 31.00 0.0383 0.0211 0.0629 39.61 7986.31MB 33.57 0.0582 0.0226 0.1555 87.54 3334.57MB
Ours 32.02 0.0309 0.0163 0.0460 61.26 31.43MB 34.89 0.0597 0.0233 0.1760 147.58 10.26MB
Painter Theater
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
DyNeRF [22] 35.95 - 0.0140 0.1464 - - 29.53 - 0.0305 0.1881 - -
HyperReel [1] 35.91 0.0385 - 0.1173 - - 33.32 0.0525 - 0.1154 - -
Deformable 3DGS [44] 34.71 0.0497 0.0211 0.1302 84.37 51.56MB 29.65 0.0768 0.0382 0.1795 80.40 54.75MB
E-D3DGS [2] 35.97 0.0360 0.0149 0.0903 94.91 38.00MB 31.04 0.0643 0.0307 0.1493 56.88 77.61MB
STG [23] 35.73 0.0369 0.0148 0.0963 157.01 54.84MB 31.16 0.0595 0.0286 0.1332 137.48 48.52MB
4DGS [46] 35.73 0.0423 0.0176 0.1125 54.73 5667.79MB 31.29 0.0696 0.0341 0.1653 54.05 5770.69MB
Ours 36.73 0.0380 0.0154 0.1014 121.72 14.03MB 31.54 0.0622 0.0297 0.1475 56.91 34.31MB
Trains Average
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
DyNeRF [22] 31.58 - 0.0190 0.0670 - - 31.80 - 0.0210 0.1400 0.02 30.00MB
HyperReel [1] 29.74 0.0525 - 0.0723 - - 32.70 0.0470 - 0.1090 4.00 60.00MB
Deformable 3DGS [44] 26.39 0.1104 0.0663 0.2040 67.32 67.08MB 30.95 0.0696 0.0353 0.1553 76.09 61.36MB
E-D3DGS [2] 30.87 0.0525 0.0289 0.0976 56.81 78.23MB 32.89 0.0494 0.0231 0.1114 79.14 56.07MB
STG [23] 32.61 0.0296 0.0169 0.0380 147.70 61.34MB 33.35 0.0404 0.0187 0.0846 141.73 51.35MB
4DGS [46] 28.79 0.0590 0.0362 0.0985 40.36 7775.97MB 32.07 0.0535 0.0263 0.1189 55.26 6107.07MB
Ours 32.69 0.0301 0.0172 0.0362 28.25 72.21MB 33.57 0.0442 0.0204 0.1014 83.14 32.45MB
C. Ablation Study
MLP. As shown in Table 6, MLPs of various sizes exhibit
similar results, because FϕandFθonly provide tempo-
ral and viewpoint varying information, which can be effec-
tively captured by lightweight MLPs.Trade-off coefficients. Table 6 shows the results of various
trade-off coefficients λandκ. Our default trade-off coeffi-
cients are chosen empirically.
Opacity loss. As shown in Table 6, simply applying Lopa
to existing baselines does not improve performance. For
4DGS and STG, the reason may arise from their explicit

---

## Page 14

Table 5. Quantitative comparisons with various competitive baselines on the Neural 3D Video Dataset.1: Only report the result on the
Flame Salmon scene.2: Exclude the Coffee Martini scene.3: These methods train each model with a 50-frame video sequence to prevent
memory overflow, requiring six models to complete the overall evaluation.4: Only report the overall results.
Coffee Martini Cook Spinach
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
HexPlane2,3[3] - - - - - - 32.04 - 0.0150 0.0820 - -
NeRFPlayer3[39] 31.53 0.0245 - 0.085 - - 30.56 0.0355 - 0.1130 - -
HyperReel [1] 28.37 0.0540 - 0.1270 - - 32.30 0.0295 - 0.0890 - -
K-Planes [13] 29.99 - 0.0170 - - - 31.82 - 0.0170 - - -
MixV oxels-L [41] 29.63 - 0.0162 0.099 - - 32.40 - 0.0157 0.088 - -
MixV oxels-X [41] 30.39 - 0.0160 0.062 - - 32.63 - 0.0146 0.057 - -
Dynamic 3DGS [29] 26.49 0.0263 0.0129 0.087 - - 30.72 0.0295 0.0161 0.090 - -
Deformable 3DGS [44] 27.88 0.0470 0.0284 0.0855 26.89 33.84MB 33.06 0.0267 0.0142 0.0519 31.06 33.21MB
E-D3DGS [2] 29.56 0.0319 0.0193 0.0300 51.94 57.97MB 32.71 0.0219 0.0123 0.0255 74.11 36.82MB
STG3[23] 28.55 0.0418 0.0253 0.0692 221.76 214.52MB 33.18 0.0215 0.0113 0.0367 290.03 151.52MB
4DGS [46] 27.98 0.0435 0.0265 0.0847 78.79 3704.58MB 32.73 0.0245 0.0133 0.0489 111.77 2474.94MB
Ours 27.84 0.0440 0.0270 0.0770 75.66 24.90MB 33.08 0.0230 0.0125 0.0471 92.51 19.83MB
Cut Roasted Beef Flame Salmon
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
Neural V olume1[25] - - - - - - 22.80 - 0.0620 0.2950 - -
LLFF1[30] - - - - - - 23.24 - 0.0200 0.2350 - -
DyNeRF1[22] - - - - - - 29.58 - 0.0200 0.0830 0.015 28.00MB
HexPlane2,3[3] 32.55 - 0.0130 0.0800 - - 29.47 - 0.0180 0.0780 - -
NeRFPlayer3[39] 29.35 0.0460 - 0.1440 - - 31.65 0.0300 - 0.098 - -
HyperReel [1] 32.92 0.0275 - 0.084 - - 28.26 0.0590 - 0.136 - -
K-Planes [13] 31.82 - 0.0170 - - - 30.44 - 0.0235 - - -
MixV oxels-L [41] 32.40 - 0.0157 0.088 - - 29.81 - 0.0255 0.116 - -
MixV oxels-X [41] 32.63 - 0.0146 0.057 - - 30.60 - 0.0233 0.078 - -
Dynamic 3DGS [29] 30.72 0.0295 0.0161 0.0900 - - 26.92 0.0512 0.0302 0.1220 - -
Deformable 3DGS [44] 31.43 0.0333 0.0204 0.0551 28.43 33.14MB 28.70 0.0432 0.0255 0.0804 28.72 34.17MB
E-D3DGS [2] 33.02 0.0213 0.0116 0.0258 74.33 36.63MB 29.79 0.0363 0.0216 0.0535 61.03 45.08MB
STG3[23] 33.55 0.0207 0.0106 0.0367 299.98 135.28MB 29.48 0.0375 0.0224 0.0630 215.69 268.39MB
4DGS [46] 33.23 0.0226 0.0119 0.0470 109.11 2555.56MB 28.86 0.0425 0.0257 0.0832 64.31 4695.46MB
Ours 33.58 0.0217 0.0113 0.0489 75.22 25.20MB 28.48 0.0412 0.0251 0.0736 64.07 30.26MB
Flame Steak Sear Steak
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
HexPlane2,3[3] 32.08 - 0.0110 0.0660 - - 32.39 - 0.0110 0.0700 - -
NeRFPlayer3[39] 31.93 0.0250 - 0.0880 - - 29.13 0.0460 - 0.138 - -
HyperReel [1] 32.20 0.0255 - 0.078 - - 32.57 0.0240 - 0.077 - -
K-Planes [13] 32.38 - 0.0150 - - - 32.52 - 0.0130 - - -
MixV oxels-L [41] 31.83 - 0.0144 0.088 - - 32.10 - 0.0122 0.080 - -
MixV oxels-X [41] 32.10 - 0.0137 0.051 - - 32.33 - 0.0121 0.053 - -
Dynamic 3DGS [29] 33.24 0.0233 0.0113 0.0790 - - 33.68 0.0224 0.0105 0.079 - -
Deformable 3DGS [44] 31.83 0.0248 0.0137 0.0418 30.91 30.72MB 33.01 0.0237 0.0125 0.0416 31.73 30.74MB
E-D3DGS [2] 30.23 0.0241 0.0149 0.0243 76.92 32.244MB 31.91 0.0200 0.0110 0.0233 79.89 32.426MB
STG3[23] 33.59 0.0178 0.0088 0.0290 305.22 141.25MB 33.89 0.0174 0.0085 0.0295 308.15 141.16MB
4DGS [46] 33.19 0.0204 0.0106 0.0389 91.52 3173.37MB 33.44 0.0204 0.0105 0.0411 124.66 2164.07MB
Ours 32.27 0.0242 0.0129 0.0538 63.84 30.48MB 33.67 0.0200 0.0103 0.0403 93.21 19.62MB
Average
Method PSNR↑DSSIM 1↓DSSIM 2↓LPIPS↓ FPS↑ Storage ↓
Neural V olume1[25] 22.80 - 0.0620 0.2950 - -
LLFF1[30] 23.24 - 0.0200 0.2350 - -
DyNeRF1[22] 29.58 - 0.0200 0.0830 0.015 28.00MB
HexPlane2,3[3] 31.71 - 0.0140 0.0750 0.56 200.00MB
StreamRF4[21] 28.26 - - - 10.90 5310.00MB
NeRFPlayer3[39] 30.69 0.0340 - 0.1110 0.05 5130.00MB
HyperReel [1] 31.10 0.0360 - 0.0960 2.00 360.00MB
K-Planes [13] 31.63 - 0.0180 - 0.30 311.00MB
MixV oxels-L [41] 31.34 - 0.0170 0.0960 37.70 500.00MB
MixV oxels-X [41] 31.73 - 0.0150 0.0640 4.60 500.00MB
Dynamic 3DGS [29] 30.46 0.0350 0.0190 0.0990 460.00 2772.00MB
C-D3DGS4[16] 30.46 - - 0.1500 118.00 338.00MB
Deformable 3DGS [44] 30.98 0.0331 0.0191 0.0594 29.62 32.64MB
E-D3DGS [2] 31.20 0.0259 0.0151 0.0304 69.70 40.20MB
STG3[23] 32.04 0.0261 0.0145 0.0440 273.47 175.35MB
4DGS [46] 31.57 0.0290 0.0164 0.0573 96.69 3128.00MB
Ours 31.49 0.0290 0.0165 0.0568 77.42 25.05MB
modeling of motion as fixed low-order polynomials, e.g., linear or quadratic. Thus, enforcing sparsity in opacity may

---

## Page 15

Table 6. Ablation study on the Fabien scene.Fϕdenotes the color
MLP network, Fθdenotes the deformation MLP network. The
first row denotes our final solution with λ= 0.2andκ= 5e−4.
Variant PSNR↑ DSSIM 1↓ N↓ Params ↓
Ours 34.89 0.0597 0.31M 6.43M
LargeFϕ 33.59 0.0653 0.35M 7.30M
LargeFθ 34.71 0.0604 0.33M 8.24M
LargeFϕ+Fθ 34.27 0.0627 0.30M 7.85M
λ=0.1,κ=5e−434.09 0.0643 0.27M 5.74M
λ=0.3,κ=5e−433.99 0.0627 0.47M 9.77M
λ=0.2,κ=1e−433.99 0.0639 0.48M 10.00M
λ=0.2,κ=1e−334.01 0.0633 0.31M 6.41M
4DGS [46] 33.57 0.0582 5.43M 874.14M
4DGS+ Lopa 23.23 0.1037 8.41M 1353.13M
STG [23] 35.61 0.0468 0.30M 10.54M
STG+Lopa 33.78 0.0610 0.28M 26.03M
E-D3DGS [2] 34.69 0.0612 0.06M 5.25M
E-D3DGS+ Lopa 34.52 0.0623 0.11M 10.21M
Table 7. Effect of view direction on the Birthday scene.
Variant PSNR↑DSSIM 1↓N↓ Params ↓
Ours 32.02 0.0309 0.91M 18.48M
w/o view 27.35 0.0697 1.54M 31.44M
conflict with the motion priors, resulting in either over-
pruning or insufficient flexibility to model more complex
nonlinear temporal dynamics. For E-D3DGS, while it sup-
ports more flexible motion via multi-granularity embed-
dings, it induces locally similar deformation by regularizing
nearby per-Gaussian embeddings. Therefore, adding Lopa
may disrupt this smooth deformation by encouraging abrupt
temporal activation, degrading local coherence. In contrast,
our method jointly learns deformation and opacity in a uni-
fied way, enabling both localized adaptation and temporal
sparsity without conflict.
View direction. Since the same time frames may reveal
different visible content under different viewpoints, the
view direction input is critical for disambiguating view-
dependent geometry and motion, especially in sparse or oc-
cluded camera settings. As shown in Table 7, removing the
view input significantly degrades rendering quality and re-
quires more Gaussians, indicating less efficient and less ac-
curate scene modeling.
D. Limitations
First, MEGA lacks robustness to real-world noise such as
motion blur and color inconsistency. Integrating techniques
from Robust GS [8] is a promising future direction. Sec-
ond, while MEGA achieves significant compression, the in-
creased flexibility of 4D Gaussians can lead to artifacts in
cases of ultra-fast motion. A potential solution is to decom-
pose the scene into static and dynamic regions, and then use
3D Gaussians for static regions and 4D Gaussians for dy-
namic regions. Finally, MEGA struggles with novel viewextrapolation beyond the bound of training views, which
can introduce artifacts. Incorporating strong scene priors
could help improve generalization.