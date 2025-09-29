

---

## Page 1

1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering
Yuan Yuheng Qiuhong Shen Xingyi Yang Xinchao Wang
National University of Singapore
{yuhengyuan,qiuhong.shen,xyang }@u.nus.edu, xinchao@nus.edu.sg
0 200 400 600 800
FPS30.030.531.031.532.032.5PSNR(dB)Ours
4DGS
4DGaussians
Dynamic  3DGSSTG
E-D3DGS
Figure 1. Compressibility and Rendering Speed. We introduce 4DGS-1K , a novel compact representation with high rendering speed.
In contrast to 4D Gaussian Splatting (4DGS) [40], we can achieve rasterization at 1000+ FPS while maintaining comparable photorealistic
quality with only 2%of the original storage size. The right figure is the result tested on the N3V [18] datasets, where the radius of the dot
corresponds to the storage size.
Abstract
4D Gaussian Splatting (4DGS) has recently gained con-
siderable attention as a method for reconstructing dynamic
scenes. Despite achieving superior quality, 4DGS typically
requires substantial storage and suffers from slow render-
ing speed. In this work, we delve into these issues and iden-
tify two key sources of temporal redundancy. (Q1) Short-
Lifespan Gaussians : 4DGS uses a large portion of Gaus-
sians with short temporal span to represent scene dynam-
ics, leading to an excessive number of Gaussians. (Q2)
Inactive Gaussians : When rendering, only a small sub-
set of Gaussians contributes to each frame. Despite this,
all Gaussians are processed during rasterization, resulting
in redundant computation overhead. To address these re-
dundancies, we present 4DGS-1K , which runs at over 1000
FPS on modern GPUs. For Q1, we introduce the Spatial-
Temporal Variation Score, a new pruning criterion that ef-
fectively removes short-lifespan Gaussians while encourag-
ing 4DGS to capture scene dynamics using Gaussians with
longer temporal spans. For Q2, we store a mask for ac-
tive Gaussians across consecutive frames, significantly re-
ducing redundant computations in rendering. Compared
to vanilla 4DGS, our method achieves a 41×reduction in
storage and 9×faster rasterization speed on complex dy-
namic scenes, while maintaining comparable visual quality.
Please see our project page at 4DGS-1K.1. Introduction
Novel view synthesis for dynamic scenes allows for the cre-
ation of realistic representations of 4D environments, which
is essential in fields like computer vision, virtual reality,
and augmented reality. Traditionally, this area has been
led by neural radiance fields (NeRF) [2, 12, 18, 21, 25],
which model opacity and color over time to depict dynamic
scenes. While effective, these NeRF-based methods come
with high training and rendering costs, limiting their prac-
ticality, especially in real-time applications and on devices
with limited resources.
Recently, point-based representations like 4D Gaussian
Splatting (4DGS) [40] have emerged as strong alternatives.
4DGS models a dynamic scene using a set of 4D Gaussian
primitives, each with a 4-dimensional mean and a 4×4co-
variance matrix. At any given timestamp, a 4D Gaussian
is decomposed into a set of conditional 3D Gaussians and
a marginal 1D Gaussian, the latter controlling the opacity
at that moment. This mechanism allows 4DGS to effec-
tively capture both static and dynamic features of a scene,
enabling high-fidelity dynamic scene reconstruction.
However, representing dynamic scenes with 4DGS is
both storage-intensive and slow. Specifically, 4DGS of-
ten requires millions of Gaussians, leading to significant
storage demands (averaging 2GB for each scene on the
N3V [18] dataset) and suboptimal rendering speed. In com-
1arXiv:2503.16422v1  [cs.CV]  20 Mar 2025

---

## Page 2

parison, mainstream deformation field methods [39] require
only about 90MB for the same dataset. Therefore, reduc-
ing the storage size of 4DGS [40] and improving rendering
speed are essential for efficiently representing complex dy-
namic scenes.
We look into the cause of such an explosive number of
Gaussian and place a specific emphasis on two key issues.
(Q1) A large portion of Gaussians exhibit a short temporal
span. In empirical experiments, 4DGS tends to favor “flick-
ing” Gaussians to fit complex dynamic scenes, which just
influence a short portion of the temporal domain. This ne-
cessitates that 4DGS relies on a large number of Gaussians
to reconstruct a high-fidelity scene. As a result, substan-
tial storage is needed to record the attributes of these Gaus-
sians: (Q2) Inactive Gaussians lead to redundant computa-
tion. During rendering, 4DGS needs to process all Gaus-
sians. However, only a very small portion of Gaussians
are active at that moment. Therefore, most of the computa-
tion time is spent on inactive Gaussians. This phenomenon
greatly hampers the rendering speed. In this paper, we in-
troduce 4DGS-1K , a framework that significantly reduces
the number of Gaussians to minimize storage requirements
and speedup rendering while maintaining high-quality re-
construction. To address these issues, 4DGS-1K introduces
a two-step pruning approach:
•Pruning Short-Lifespan Gaussians. We propose a
novel pruning criterion called the spatial-temporal vari-
ation score , which evaluates the temporal impact of each
Gaussian. Gaussians with minimal influence are identi-
fied and pruned, resulting in a more compact scene repre-
sentation with fewer Gaussians with short temporal span.
•Filtering Inactive Gaussians. To further reduce redun-
dant computations during rendering, we use a key-frame
temporal filter that selects the Gaussians needed for each
frame. On top of this, we share the masks for adjacent
frames. This is based on our observation that Gaussians
active in adjacent frames often overlap significantly.
Besides, the pruning in step 1 enhances the masking pro-
cess in step 2. By pruning Gaussians, we increase the tem-
poral influence of each Gaussian, which allows us to select
sparser key frames and further reduce storage requirements.
We have extensively tested our proposed model on var-
ious dynamic scene datasets including real and synthetic
scenes. As shown in Fig. 1, 4DGS-1K reduces storage costs
by 41×on the Neural 3D Video datasets [18] while main-
taining equivalent scene representation quality. Crucially, it
enables real-time rasterization speeds exceeding 1,000 FPS.
These advancements collectively position 4DGS-1K as a
practical solution for high-fidelity dynamic scene modeling
without compromising efficiency.
In summary, our contributions are three-fold:
• We delve into the temporal redundancy of 4D Gaussian
Splatting, and explain the main reason for the storagepressure and suboptimal rendering speed.
• We introduce 4DGS-1K , a compact and memory-efficient
framework to address these issues. It consists of two
key components, a spatial-temporal variation score-based
pruning strategy and a temporal filter.
• Extensive experiments demonstrate that 4DGS-1K not
only achieves a substantial storage reduction of approx-
imately 41 ×but also accelerates rasterization to 1000+
FPS while maintaining high-quality reconstruction.
2. Related Work
2.1. Novel view synthesis for static scenes
Recently, neural radiance fields(NeRF) [25] have achieved
encouraging results in novel view synthesis. NeRF [25] rep-
resents the scene by mapping 3D coordinates and view de-
pendency to color and opacity. Since NeRF [25] requires
sampling each ray by querying the MLP for hundreds of
points, this significantly limits the training and rendering
speed. Subsequent studies [5, 11, 26, 31, 32, 35, 37, 38]
have attempted to speed up the rendering by introducing
specialized designs. However, these designs also constrain
the widespread application of these models. In contrast, 3D
Gaussian Splatting(3DGS) [14] has gained significant at-
tention, which utilizes anisotropic 3D Gaussians to repre-
sent scenes. It achieves high-quality results with intricate
details, while maintaining real-time rendering performance.
2.2. Novel view synthesis for dynamic scenes
Dynamic NVS poses new challenges due to the temporal
variations in the input images. Previous NeRF-based dy-
namic scene representation methods [2, 4, 12, 17, 18, 21,
24, 30, 34, 36] handle dynamic scenes by learning a map-
ping from spatiotemporal coordinates to color and density.
Unfortunately, these NeRF-based models are constrained in
their applications due to low rendering speeds. Recently,
3D Gaussians Splatting [14] has emerged as a novel ex-
plicit representation, with many studies [3, 6, 13, 22, 39, 41]
attempting to model the dynamic scenes based on it. 4D
Gaussian Splatting(4DGS) [40] is one of the representa-
tives. It utilizes a set of 4D Gaussian primitives. However,
4DGS often requires a huge redundant number of Gaussians
for dynamic scenes. These Gaussians lead to tremendous
storage and suboptimal rendering speed. To this end, we
focus on analyzing the temporal redundancy of 4DGS [40]
in hopes of developing a novel framework to achieve lower
storage requirements and higher rendering speeds.
2.3. Gaussian Splatting Compression
3D Gaussian-based large-scale scene reconstruction typi-
cally requires millions of Gaussians, resulting in the re-
quirement of up to several gigabytes of storage. There-
fore, subsequent studies have attempted to tackle these is-
2

---

## Page 3

sues. Specifically, Compgs [27] and Compact3D [16] em-
ploy vector quantization to store Gaussians within code-
books. Concurrently, inspired by model pruning, some
studies [1, 8, 9, 20, 28, 29] have proposed criterion to
prune Gaussians by a specified ratio. However, compared
to 3DGS [14], 4DGS [40] introduces an extra temporal di-
mension to enable dynamic representation. Previous 3DGS-
based methods may therefore be unsuitable for 4DGS. Con-
sequently, we first identify a key limitation leading to this
problem, referred as temporal redundancy . Furthermore,
we propose a novel pruning criterion leveraging spatial-
temporal variation, and a temporal filter to achieve more
efficient storage requirements and higher rendering speed.
3. Preliminary of 4D Gaussian Splatting
Our framework builds on 4D Gaussian Splatting
(4DGS) [40], which reconstructs dynamic scenes by
optimizing a collection of anisotropic 4D Gaussian primi-
tives. For each Gaussian, it is characterized by a 4D mean
µ= (µx, µy, µz, µt)∈R4coupled with a covariance
matrix Σ∈R4×4.
By treating time and space dimensions equally, the 4D
covariance matrix Σcan be decomposed into a scaling ma-
trixS4D= (sx, sy, sz, st)∈R4and a rotation matrix
R4D∈R4×4.R4Dis represented by a pair of left quater-
nionql∈R4and right quaternion qr∈R4.
During rendering, each 4D Gaussian is decomposed into
a conditional 3D Gaussian and a 1D Gaussian at a specific
timet. Moreover, the conditional 3D Gaussian can be de-
rived from the properties of the multivariate Gaussian with:
µxyz|t=µ1:3+ Σ 1:3,4Σ−1
4,4(t−µt)
Σxyz|t= Σ 1:3,1:3−Σ1:3,4Σ−1
4,4Σ4,1:3(1)
Here, µ1:3∈R3andΣ1:3,1:3∈R3×3denote the spatial
mean and covariance, while µtandΣ4,4are scalars rep-
resenting the temporal components. To perform rasteriza-
tion, given a pixel under view Iand timestamp t, its color
I(u, v, t )can be computed by blending visible Gaussians
that are sorted by their depth:
I(u, v, t ) =NX
ici(d)αii−1Y
j=1(1−αj) (2)
with
αi=pi(t)pt(u, v|t)σi
pi(t)∼ N(t;µt,Σ4,4)(3)
where ci(d)is the color of each Gaussian, and αiis given by
evaluating a 2D Gaussian with covariance Σ2Dmultiplied
with a learned per-point opacity σiand temporal Gaussian
distribution pi(t). In the following discussion, we denote
Σ4,4asΣtfor simplicity.Temporal Redundancy. Despite achieving high quality,
4DGS requires a huge number of Gaussians to model dy-
namic scenes. We identify a key limitation leading to this
problem: 4DGS represents scenes through temporally inde-
pendent Gaussians that lack explicit correlation across time.
This means that, even static objects are redundantly rep-
resented by hundreds of Gaussians, which inconsistently
appear or vanish across timesteps. We refer to this phe-
nomenon as temporal redundancy . As a result, scenes end
up needing more Gaussians than they should, leading to ex-
cessive storage demands and suboptimal rendering speeds.
In Sec. 4, we analyze the root causes of this issue and pro-
pose a set of solutions to reduce the count of Gaussians.
4. Methodology
Our goal is to compress 4DGS by reducing the number of
Gaussians while preserving rendering quality. To achieve
this, we first analyze the redundancies present in 4DGS, as
detailed in Sec. 4.1. Building on this analysis, we introduce
4DGS-1K in Sec. 4.2, which incorporates a set of compres-
sion techniques designed for 4DGS. 4DGS-1K enables ren-
dering speeds of over 1,000 FPS on modern GPUs.
4.1. Understanding Redundancy in 4DGS
This section investigates why 4DGS requires an excessive
number of Gaussians to represent dynamic scenes. In par-
ticular, we identify two key factors. First, 4DGS models
object motion using a large number of transient Gaussians
that inconsistently appear and disappear across timesteps,
leading to redundant temporal representations. Second, for
each frame, only a small fraction of Gaussians actually con-
tribute to the rendering. We discuss those problems below.
Massive Short-Lifespan Gaussians . We observe that
4DGS tends to store numerous Gaussians that flicker in
time. We refer to these as Short-Lifespan Gaussians . To in-
vestigate this property, we analyze the Gaussians’ opacity,
which controls visibility. Intuitively, Short-Lifespan Gaus-
sians exhibit an opacity pattern that rapidly increases and
then suddenly decreases. In 4DGS, this behavior is typi-
cally reflected in the time variance parameter Σt—small Σt
values indicate a short lifespan.
Observations. Specifically, we plot the distribution of Σtfor
all Gaussians in the Sear Steak scene. As shown in Fig. 2a,
most of Gaussians has small Σtvalues (e.g. 70% have
Σt<0.25). Moreover, as shown in Fig. 3, we visualize
the spatial distribution of Σtvalues. We take the reciprocal
ofΣtand then normalize it. Therefore, brighter regions in
the image indicate smaller Σt. Most of these Gaussians are
concentrated along the edges of moving objects.
Therefore, in 4DGS, nearly all Gaussians have a short
lifespan, especially around the fast-moving objects. This
property leads to high storage needs and slower rendering.
3

---

## Page 4

0.5 1.0 1.5
the Val e of Σ
t0.00.20.40.60.81.0PercentageO rs(w . F ilter)
O rs(w .o F ilter)
4DG S(a)
0 100 200 300
Timestamp0.00.20.40.60.81.0Active Ratio4DGS
Ours(w .o Filter)
Ours(w . Filter) (b)
0 50 100 150
Timestamp0.00.20.40.60.81.0Activation IoU4DGS
Ours(w .o F ilter)
Ours(w . F ilter) (c)
Figure 2. Temporal redundancy Study. (a) The Σtdistribution of 4DGS. The red line shows the result of vanilla 4DGS. The other two
lines represent our model has effectively reduced the number of transient Gaussians with small Σt. (b) The active ratio during rendering
at different timestamps. It demonstrates that most of the computation time is spent on inactive Gaussians in vanilla 4DGS. However,
4DGS-1K can significantly reduce the occurrence of inactive Gaussians during rendering to avoid unnecessary computations. (c) This
figure shows the IoU between the set of active Gaussians in the first frame and frame t. It proves that active Gaussians tend to overlap
significantly across adjacent frames.
Figure 3. Visualizations of Distribution of Σt.Most of these
Gaussians are concentrated along the edges of moving objects.
Inactive Gaussians. Another finding is that, during the for-
ward rendering, actually, only a small fraction of Gaussians
are contributing. Interestingly, active ones tend to overlap
significantly across adjacent frames. To quantify this, we
introduce two metrics: (1) Active ratio . This ratio is defined
as the proportion of the total number of active Gaussians
across all views at any moment relative to the total number
of Gaussians. (2) Activation Intersection-over-Union (IoU) .
This is computed as IoU between the set of active Gaussians
in the first frame and in frame t.
Observations. Again, we plot the two metrics from Sear
Steak scene. As shown in Fig. 2b, nearly 85% of Gaus-
sians are inactive at each frame, even though all Gaussians
are processed during rendering. Moreover, Fig. 2c demon-
strates that the active Gaussians remain quite consistent
over time, with an IoU above 80% over a 20-frame window.
The inactive gaussians bring a significant issue in 4DGS,
because each 4D Gaussian must be decomposed into a
3D Gaussian and a 1D Gaussian before rasterization (see
Eq. (1)). Therefore, a large portion of computational re-
sources is wasted on inactive Gaussians.
In summary, redundancy in 4DGS comes from massive
Short-Lifespan Gaussians and inactive Gaussians. These
insights motivate our compression strategy to eliminate re-dundant computations while preserving rendering quality.
4.2. 4DGS-1K for Fast Dynamic Scene Rendering
Building on the analysis above, we introduce 4DGS-1K, a
suite of compression techniques specifically designed for
4DGS to eliminate redundant Gaussians. As shown in
Fig. 4, this process involves two key steps. First, we identify
and globally prune unimportant Gaussians with low Spatial-
Temporal Variation Score in Sec. 4.2.1. Second, we apply
local pruning using a temporal filter to inactive Gaussians
that are not needed at each timestep in Sec. 4.2.2.
4.2.1. Pruning with Spatial-Temporal Variation Score
We first prune unimportant 4D Gaussians to improve effi-
ciency. Like 3DGS, we remove those that have a low im-
pact on rendered pixels. Besides, we additionally remove
short-lifespan Gaussians—those that persist only briefly
over time. To achieve this, we introduce a novel spatial-
temporal variation score as the pruning criterion for 4DGS.
It is composed of two parts, spatial score that measures the
Gaussians contributions to the pixels in rendering, and tem-
poral score considering the lifespan of Gaussians.
Spatial score. Inspired by the previous method [8, 9] and
α-blending in 3DGS [14], we define the spatial score by
aggregating the ray contribution of Gaussian gialong all
raysracross all input images at a given timestamp. It can
accurately capture the contribution of each Gaussian to one
pixel. Consequently, the spatial contribution score SSis
obtained by traversing all pixels:
SS
i=NHWX
k=1αii−1Y
j=1(1−αj) (4)
where αiQi−1
j=1(1−αj)reflects the contribution of ith
Gaussian to the final color of all pixels according to the al-
pha composition in Eq. (2).
4

---

## Page 5

Y
FilterTraining V iewTesting V iew
(a) Transient Gaussian Pruning (b) Temporal FilterXInactive Gaussian
Figure 4. Overview of 4DGS-1K. (a) We first calculate the spatial-temporal variation score for each 4D Gaussian on training views, to
prune Gaussians with short lifespan (The Red Gaussian). (b) The temporal filter is introduced to filter out inactive Gaussians before the
rendering process to alleviate suboptimal rendering speed. At a given timestamp t, the set of Gaussians participating in rendering is derived
from the two adjacent key-frames, t0and t0+∆ t.
Temporal score. It is expected to assign a higher tempo-
ral score to Gaussians with a longer lifespan. To quantify
this, we compute the second derivative of temporal opac-
ity function pi(t)defined in Eq. (3). The second derivative
p(2)
i(t)is computed as
p(2)
i(t) = ((t−µt)2
Σ2
t−1
Σt)pi(t) (5)
Intuitively, large second derivative magnitude corresponds
to unstable, short-lived Gaussians, while low second deriva-
tive indicates smooth, persistent ones.
Moreover, since the second derivative spans the real
number domain R, we apply tanh function to map it to the
interval (0,1). Consequently, the score for opacity varia-
tion,STV
i, of each Gaussian gi,tis expressed as:
STV
i=TX
t=01
0.5·tanh(p(2)
i(t)) + 0.5. (6)
In addition to the opacity range rate, the volume of 4D
Gaussians is necessary to be considered, as described in
Eq. (1). The volume should be normalized following the
method in [8], denoted as γ(S4D) = Norm (V(S4D)).
Therefore, the final temporal score ST
i=STV
iγ(S4D
i)
Finally, by aggregating both spatial and temporal score,
the spatial-temporal variation score Sican be written as:
Si=TX
t=0ST
iSS
i (7)
Pruning. All 4D Gaussians are ranked based on their
spatial-temporal variation score Si, and Gaussians withlower scores are pruned to reduce the storage burden of
4DGS [40]. The remaining Gaussians are optimized over
a set number of iterations to compensate for minor losses
resulting from pruning.
4.2.2. Fast rendering with temporal filtering
Our analysis reveals that inactive Gaussians induces unnec-
essary computations in 4DGS, significantly slowing down
rendering. To address this issue, we introduce a tempo-
ral filter that dynamically selects active Gaussians. We ob-
served that active Gaussians in adjacent frames overlap con-
siderably (as detailed in Sec. 4.1), which allows us to share
their corresponding masks across a window of frames.
Key-frame based Temporal Filtering. Based on this ob-
servation, we design a key-frame based temporal filtering
for active Gaussians. We select sparse key-frames at even
intervals and share their masks with surrounding frames.
Specifically, we select a list of key-frame timestamps
{ti}T
i=0, where Tdepends on the chosen interval ∆t. For
eachti, we render the images from all training views at cur-
rent timestamp and calculate the visibility list {mi,j}N
j=1,
where mi,jis the visibility mask obtained by Eq. (2) from
thejthtraining viewpoint at timestamp tiandNis the num-
ber of training views at current timestamp. The final set of
active Gaussian masks is given bynSN
j=1mi,joT
i=0.
Filter based Rendering. To render the images from any
viewpoint at a given timestamp ttest, we consider its two
nearest key-frames, denoted as tlandtr. Then, we perform
rasterization while only considering the Gaussians marked
by masknSN
j=1mi,jo
i=l,r. This method explicitly filters
5

---

## Page 6

Table 1. Quantitative comparisons on the Neural 3D Video Dataset.
Method PSNR ↑ SSIM↑ LPIPS↓ Storage(MB) ↓ FPS↑ Raster FPS ↑ #Gauss↓
Neural V olume1[21] 22.80 - 0.295 - - - -
DyNeRF1[18] 29.58 - 0.083 28 0.015 - -
StreamRF[17] 28.26 - - 5310 10.90 - -
HyperReel[2] 31.10 0.927 0.096 360 2.00 - -
K-Planes[12] 31.63 - 0.018 311 0.30 - -
Dynamic 3DGS[23] 30.67 0.930 0.099 2764 460 - -
4DGaussian[39] 31.15 0.940 0.049 90 30 - -
E-D3DGS[3] 31.31 0.945 0.037 35 74 - -
STG[19] 32.05 0.946 0.044 200 140 - -
4D-RotorGS[7] 31.62 0.940 0.140 - 277 - -
MEGA[43] 31.49 - 0.056 25 77 - -
Compact3D[16] 31.69 0.945 0.054 15 186 - -
4DGS[40] 32.01 - 0.055 - 114 - -
4DGS2[40] 31.91 0.946 0.052 2085 90 118 3333160
Ours 31.88 0.946 0.052 418 805 1092 666632
Ours-PP 31.87 0.944 0.053 50 805 1092 666632
1The metrics of the model are tested without “coffee martini” and the resolution is set to 1024 × 768.
2The retrained model from the official implementation.
Table 2. Quantitative comparisons on the D-NeRF Dataset.
Method PSNR ↑ SSIM↑ LPIPS↓ Storage(MB) ↓ FPS↑ Raster FPS ↑ #Gauss↓
DNeRF[30] 29.67 0.95 0.08 - 0.1 - -
TiNeuV ox[10] 32.67 0.97 0.04 - 1.6 - -
K-Planes[12] 31.07 0.97 0.02 - 1.2 - -
4DGaussian[39] 32.99 0.97 0.05 18 104 - -
Deformable3DGS[41] 40.43 0.99 0.01 27 70 - 131428
4D-RotorGS[7] 34.26 0.97 0.03 112 1257 - -
4DGS[40] 34.09 0.98 0.02 - - - -
4DGS1[40] 32.99 0.97 0.03 278 376 1232 445076
Ours 33.34 0.97 0.03 42 1462 2482 66460
Ours-PP 33.37 0.97 0.03 7 1462 2482 66460
1The retrained model from the official implementation.
out inactive Gaussians to speed up rendering.
Note that using long intervals may overlook some Gaus-
sians, reducing rendering quality. Therefore, we fine-tune
Gaussians recorded by the masks to compensate for losses.
5. Experiment
5.1. Experimental Settings
Datasets. We utilize two dynamic scene datasets to demon-
strate the effectiveness of our method: (1) Neural 3D Video
Dataset (N3V) [18]. This dataset consists of six dynamic
scenes, and the resolution is 2704×2028 . For a fair com-
parison, we align with previous work [19, 40] by conducting
evaluations at a half-resolution of 300 frames. (2) D-NeRF
Dataset [30]. This dataset is a monocular video dataset
comprising eight videos of synthetic scenes. We choose
standard test views that originate from novel camera posi-
tions not encountered during the training process.
Evaluation Metrics. To evaluate the quality of rendering
dynamic scenes, we employ several commonly used im-age quality assessment metrics: Peak Signal-to-Noise Ratio
(PSNR), Structural Similarity Index Measure (SSIM), and
Learned Perceptual Image Patch Similarity (LPIPS) [42].
Following the previous work, LPIPS [42] is computed us-
ing AlexNet [15] and VGGNet [33] on the N3V dataset and
the D-NeRF dataset, respectively. Moreover, we report the
number of Gaussians and storage. To demonstrate the im-
provement in rendering speed, we report two types of FPS:
(1)FPS. It considers the entire rendering function. Due
to interference from other operations, it can’t effectively
demonstrate the acceleration achieved by our method. (2)
Raster FPS. It only considers the rasterization, the most
computationally intensive component during rendering.
Baselines. Our primary baseline for comparison is
4DGS [40], which serves as the foundation of our
model. Moreover, we compare 4DGS-1K with two con-
current works on 4D compression, MEGA [43] and Com-
pact3D [16]. Certainly, we conduct comparisons with 4D-
RotorGS [7] which is another form of representation for 4D
Gaussian Splatting with the capability for real-time render-
6

---

## Page 7

Ground Truth 4DGS Ours Ours-PP
(a) Results on Sear Steak Scene.
Ground Truth 4DGS Ours Ours-PP
(b) Results on Trex Scene.
Figure 5. Qualitative comparisons of 4DGS and our method.
Table 3. Ablation study of per-component contribution.
ID Method \DatasetPSNR↑SSIM↑LPIPS↓Storage(MB) ↓FPS↑Raster FPS ↑#Gauss ↓Filter Pruning PP
a vanilla 4DGS131.91 0.9458 0.0518 2085 90 118 3333160
b✓1,231.51 0.9446 0.0539 2091 242 561 3333160
c✓229.56 0.9354 0.0605 2091 300 561 3333160
d ✓ 31.92 0.9462 0.0513 417 312 600 666632
e✓ ✓ 31.88 0.9457 0.0524 418 805 1092 666632
f✓2✓ 31.63 0.9452 0.0524 418 789 1080 666632
g✓ ✓ ✓ 31.87 0.9444 0.0532 50 805 1092 666632
1The result with environment map.2The result without finetuning.
ing speed and high-fidelity rendering results. In addition,
we also compare our work against NeRF-based methods,
like Neural V olume [21], DyNeRF [18], StreamRF [17],
HyperReel [2], DNeRF [30] and K-Planes [12]. Further-
more, other recent competitive Gaussian-based methods
are also considered in our comparison, including Dynamic
3DGS [23], STG [19], 4DGaussian [39], and E-D3DGS [3].
Implementation Details. Our method is tested in a single
RTX 3090 GPU. We train our model following the experi-
ment setting in 4DGS [40]. After training, we perform the
pruning and filtering strategy. Then, we fine-tune 4DGS-1K
for 5,000 iterations while disabling additional clone/splitoperations. For pruning strategy, the pruning ratio is set to
80% on the N3V Dataset, and 85% on the D-NeRF Dataset.
For the temporal filtering, we set the interval ∆tbetween
key-frames to 20frames on the N3V Dataset. Considering
the varying capture speeds on the D-NeRF dataset, we se-
lect6key-frames rather than a specific frame interval. Ad-
ditionally, to further compress the storage of 4DGS [40],
we implement post-processing techniques in our model, de-
noted as Ours-PP. It includes vector quantization [27] on SH
of Gaussians and compressing the mask of filter into bits.
Note that we don’t apply environment maps imple-
mented by 4DGS on Coffee Martini and Flame Salmon
7

---

## Page 8

scenes, which significantly affects the rendering speed.
Subsequent results indicate that removing it for 4DGS-1K
does not significantly degrade the rendering quality.
5.2. Results and Comparisons
Comparisons on real-world dataset. Tab. 1 presents a
quantitative evaluation on the N3V dataset. 4DGS-1K
achieves rendering quality comparable to the current base-
line. Compared to 4DGS [40], we achieve a 41×compres-
sion and 9×faster in rendering speed at the cost of a 0.04dB
reduction in PSNR. In addition, compared to MEGA [43]
and Compact3D [16], two concurrent works on 4D com-
pression, the rendering speeds are 10 ×and 4×faster re-
spectively while maintaining a comparable storage require-
ment and high quality reconstruction. Moreover, the FPS
of 4DGS-1K far exceeds the current state-of-the-art levels.
It is nearly twice as fast as the current fastest model, Dy-
namic 3DGS [23] while requiring only 1%of the storage
size. Additionally, 4DGS-1K achieves better visual quality
than that of Dynamic 3DGS [23], with an increase of about
1.2dBin PSNR. Compared to the storage-efficient model,
E-D3DGS [3] and DyNeRF [18] we achieve an increase of
over0.5dBin PSNR and fast rendering speed. Fig. 5 offers
qualitative comparisons for the Sear Steak, demonstrating
that our results contain more vivid details.
Comparisons on synthetic dataset. In our experiments,
we benchmarked 4DGS-1K against several baselines us-
ing the monocular synthetic dataset introduced by D-
NeRF [30]. The result is shown in Tab. 2. Compared
to 4DGS [40], our method achieves up to 40×compres-
sion and 4×faster rendering speed. It is worth noting that
the rendering quality of our model even surpasses that of
the original 4DGS, with an increase of about 0.38dBin
PSNR. Furthermore, our approach exhibits higher render-
ing quality and smaller storage overhead compared to most
Gaussian-based methods. We provide qualitative results
in Fig. 5 for a more visual assessment.
5.3. Ablation Study
To evaluate the contribution of each component and the ef-
fectiveness of the pruning strategy for temporal filtering, we
conducted ablation experiments on the N3V dataset [18].
More ablations are provided in the supplement(See Sec. 8).
Table 4. Ablation study of Spatial-Temporal Variation Score.
We compare our Spatial-Temporal Variation Score with other vari-
ants, and report the PSNR score of each scene.
ID Model Sear Steak Flame Salmon
a 4DGS w/o Prune 33.60 29.10
b SS
iOnly 33.62 28.75
c ST
iOnly 33.59 28.79
dSi(w.p(1)
i(t)) 33.67 28.81
e Si(w.Σt) 33.47 28.71
f Ours 33.76 28.90Pruning. As shown in Tab. 3, our pruning strategy re-
duces the number of Gaussians by 80%, and achieves 5×
compression ratio and 5×faster rasterization speed while
slightly improving rendering quality. As shown in Fig. 2a,
our pruning strategy also reduces the presence of Gaussians
with short lifespan. As such, 4DGS-1k processes far fewer
unnecessary Gaussians (See Fig. 2b) during rendering.
Furthermore, we compare our Spatial-Temporal Varia-
tion Score with serveral variants. Specific settings are de-
scribed in Sec. 8. As shown in Tab. 4, using spatial and tem-
poral scores separately reduce the PSNR. This occurs be-
cause separate scores can amplify extreme Gaussians. For
instance, using only the spatial score (b) may retain Gaus-
sians that cover just a single frame but occupy a large spatial
volume. Our combined score balances these factors. For
variant d, using the first derivative may cause some small
Gaussians to have large ST
icompared to ours. Moreover,
since most Gaussians have small Σt, it is difficult to distin-
guish them by using Σtalong (See e). Moreover, as shown
in Fig. 2c, the pruning process expands the range of adja-
cent frames. It allows larger intervals for the temporal filter.
We will discuss it in the next part.
Temporal Filtering. As illustrated in Tab. 3, the results of
b and c are obtained by directly applying the filter to 4DGS
without fine-tuning. It proves that this component can en-
hance the rendering speed of 4DGS. However, as mentioned
in Sec. 4.1, the 4DGS contains a huge number of short lifes-
pan Gaussians. It results in some Gaussians being over-
looked in the filter, causing a slight decrease in rendering
quality. However, through pruning, most Gaussians are en-
sured to have long lifespan, making them visible even at
large intervals. Therefore, it alleviates the issue of Gaus-
sians being overlooked (See f). Furthermore, appropriate
fine-tuning allows the Gaussians in the active Gaussians list
to relearn the scene features to compensate for the loss in-
curred by the temporal filter (See e and f).
6. Conclusion
In this paper, we present 4DGS-1K , a compact and
memory-efficient dynamic scene representation capable of
running at over 1000 FPS on modern GPUs. We introduce
a novel pruning criterion called the spatial-temporal varia-
tion score, which eliminates a significant number of redun-
dant Gaussian points in 4DGS, drastically reducing storage
requirements. Additionally, we propose a temporal filter
that selectively activates only a subset of Gaussians during
each frame’s rendering. This approach enables our render-
ing speed to far surpass that of existing baselines. Com-
pared to vanilla 4DGS, 4DGS-1K achieves a 41×reduction
in storage and 9×faster rasterization speed while maintain-
ing high-quality reconstruction.
8

---

## Page 9

1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering
Supplementary Material
The Supplementary material is organized as follows:
• Sec. 7: provides additional visualization results and quan-
titative results. Furthermore, it also shows the resource
consumption which reveals the potential of 4DGS-1K for
deployment on low-performance hardware.
• Sec. 8: provides additional ablation study. It firstly pro-
vides the variant settings in the main text, then it presents
more additional ablation study to illustrate that our pa-
rameter selection is the result of a trade-off between ren-
dering quality and storage size.
• Sec. 9: discusses the reason of improved performance for
4DGS-1K. Furthermore, we introduce the limitations and
potential future directions of 4DGS-1K.
7. Experimental Results
7.1. Per scene result
We provide per-scene quantitative comparisons on the N3V
Dataset [18]( Tab. 5) and D-NeRF Dataset [30]( Tab. 6).
Compared to vanilla 4DGS [40], our model significantly
reduces the storage requirements and enhances render-
ing speed while maintaining high-quality reconstruction.
Fig. 12 and Fig. 13 show more visual comparisons on the
N3V Dataset. Fig. 14, Fig. 15 and Fig. 16 show visual
comparisons on the D-NeRF Dataset.
7.2. Resource consumption
We present the resource consumption metrics, including
training time, GPU memory allocation and additional stor-
age space. On the N3V dataset [18], 4DGS-1K only takes
approximately 30 minutes to fine-tune, with GPU memory
allocation of 10.54GB. During rendering, it only consumes
1.62GB of GPU memory. For storage requirement, 4DGS-
1K requires additional storage for the mask of filter and
codebook; however, these occupy only a minimal portion
of the total storage, approximately 1 MB per scene. These
parts are also included in the final experiment results.
The above results demonstrate the potential of 4DGS-
1K for deployment on low-performance hardware. Conse-
quently, we further test 4DGS-1K on TITAN X GPU, where
4DGS-1K maintains 200+ FPS on the N3V dataset, still far
outperforming vanilla 4DGS (20 FPS).
7.3. Additional experiments for redundancy
In this section, we provide additional experiments for re-
dundancy study as a supplement to Sec. 4.1. It is composed
of two parts: first, the visualization of the Gaussian with
short lifespan distribution, and secondly, the relationship
between FPS and the number of inactive Gaussians.Visualization of Gaussians with small lifespan. In
Sec. 4.1, we argue that in vanilla 4DGS, nearly all Gaus-
sians have a short lifespan, especially around the edge of
fast-moving objects. Therefore, we visualize the spatial
distribution of Σtto better support our redundancy study
in Sec. 4.1.
Specifically, we visualize the distribution of Σtat several
timestamps in Sear Steak Scene. The visualization results
are shown in Fig. 6. For visualization, we take the reciprocal
ofΣtduring rendering and then normalize it. Therefore,
brighter regions in the rendered image indicate smaller Σt.
As shown in Fig. 6, Gaussians with short lifespan are pri-
marily concentrated in regions of object motion, such as the
moving person and dog. Moreover, we observe that Gaus-
sians with small Σtalso appear on the edges of some objects
which exhibit significant color variation. This is because
small Gaussians are preferred in these regions to capture the
high-frequency details in the spatial dimension. As vanilla
4DGS [40] treats time and space dimensions equally, these
Gaussians also have short lifespan in the temporal dimen-
sion.
Relationship between FPS and the number of inactive
Gaussians. In Sec. 4.1, our primary prior assumption is that
the number of inactive Gaussians affects the FPS. There-
fore, we visualize the relationship between FPS and the
number of inactive Gaussians.
However, only limiting the total number of Gaussians is
incorrect in this task. As the total number increases, the
number of active Gaussians and inactive Gaussians also in-
creases, which cannot clarify whether the FPS variation is
caused by active or inactive Gaussians. Consequently, we
first identify the active Gaussians by rendering and then add
a mount of inactive Gaussians among these Gaussians.
We visualize the result in the Sear Steak Scene(See
Fig. 7). The FPS decreases as the number of inactive Gaus-
sians increases. This phenomenon strongly supports our re-
dundancy study in Sec. 4.1.
7.4. Visualizations of Pruned Gaussians
We provide the visualization of pruned Gaussians in the
Sear Steak Scene, as shown in Fig. 8. Our pruning strategy
can accurately identify Gaussians with short lifespan(See
Fig. 8c) while maintaining the high quality reconstruc-
tion(See Fig. 8d). The quantized results after pruning are
presented in Tab. 3. Our pruning technique achieves 5×
compression ratio and 5×faster rasterization speed while
slightly improving rendering quality.
9

---

## Page 10

Figure 6. Visualizations of Distribution of Σt.
0 500 1000 1500 2000
Number of Gaussians (K)0100200300400FPS
V anilla 4DGS
Figure 7. Relationship between rendering speed and the num-
ber of inactive Gaussians.
7.5. Video result
In this work, we propose a novel framework for dynamic 3D
reconstruction. Therefore, we provide several videos that
are rendered from testing viewpoints on the N3V datasets
and D-NeRF datasets to show the reconstruction quality
and temporal consistency of 4DGS-1K. These videos are
composed by concatenating each frame of 4DGS and our
method.
8. Additional ablation study
In this section, we firstly provide the variant settings
of Tab. 4. Furthermore, in addition to the ablation study in
the main text, we also investigate the impact of the pruning
ratio and different key-frames intervals on rendering quality.
We select three distinct scenes, Cook Spinach ,Cut Roasted
Beef, and Sear Steak on the N3V dataset [18] due to the
varying performance across different scenes resulting from
their unique characteristics. These results show that our de-
fault configuration is a well-rounded choice for a wide range
of scenes.
Variant Settings. As described in Sec. 4.2.1, our Spatial-
Temporal Variation Score is composed of two parts, spatial
score that measures the Gaussians contributions to the pix-els in rendering, and temporal score considering the lifes-
pan of Gaussians. By aggregating both spatial and temporal
score, our score Sican be written as:
Si=TX
t=0ST
iSS
i (8)
Therefore, the variant scores in Tab. 4 can be written as
follow.
• (b)SS
iOnly: only considering the spatial part of our
score. It can be written as:
Si=TX
t=0SS
i (9)
• (c)ST
iOnly: only considering the temporal contribution
part of our score. It can be written as:
Si=TX
t=0ST
i (10)
• (b)Si(w.p(1)
i(t)): Replace the p(2)
i(t)withp(1)
i(t)in
temporal score ST
i. It can be written as:
Si=TX
t=0ST
iSS
i
=TX
t=0STV
iγ(S4D
i)SS
i
=TX
t=01
0.5·tanh(p(1)
i(t)) + 0.5γ(S4D
i)SS
i.(11)
• (c)Si(w.Σt) Replace the STV
iwithΣt. It can be written
as:
Si=TX
t=0ST
iSS
i
=TX
t=0Σtγ(S4D
i)SS
i(12)
10

---

## Page 11

(a) Ground Truth
 (b) Distribution of Σt
 (c) Pruned Gaussians
 (d) Ours
Figure 8. Visualization of Pruned Gaussians.
Performance change with pruning ratio. As illustrated
in Fig. 10, we analyze the relationship between the pruning
ratio and rendering quality. This reveals that our spatial-
temporal variation score based pruning can even improve
scene rendering quality when the pruning ratio is relatively
low in the Cook Spinach andSear Steak scenes. Moreover,
at higher thresholds, it can maintain results comparable to
the vanilla 4DGS [40]. Our default setting represents a bal-
anced trade-off between rendering quality and storage size.
This setting allows us to achieve a 5×compression ratio
while still maintaining high-quality reconstruction.
Performance change with key-frames intervals. As
shown in Fig. 11, although the temporal filter effectively
improves rendering speed, its performance degrades signifi-
cantly when the filter is with long-interval keyframes. How-
ever, by integrating the temporal filter into the fine-tuning
process, this limitation can be mitigated. The fundamen-
tal reason is that some Gaussians which may carry critical
scene information are being overlooked due to overly long
intervals. However, the fine-tuning process effectively com-
pensates for the loss of this portion of information. This al-
lows us to utilize longer intervals to reduce the additional
computational overhead caused by mask calculations.
9. Discussion
(a) Ground Truth (b) 4DGS (c) Ours
Figure 9. Visualization of improved performance.Improved performance. As shown in Tab. 2, our model
achieves a slight PSNR improvement on the D-NeRF
Dataset [30]. This is because vanilla 4DGS often suffers
from floaters and artifacts, due to the limited training view-
points on the D-NeRF Datasets. However, in our study,
4DGS-1K not only can prune the Gaussians with short lifes-
pan, but also reduce the occurrence of floaters and artifacts,
as shown in Fig. 9. We visualize two scenes, Bouncing-
balls andJumpingjacks , on the D-NeRF Dataset. These
two scenes exhibit floaters and artifacts issues due to limited
training viewpoints, as shown in the red box. However, this
issue does not appear in 4DGS-1K. Through pruning and
filtering, 4DGS-1K successfully mitigates the occurrence of
this phenomenon.
Limitations and Future work. As shown in Tab. 5 and
Tab. 6, due to the acceleration provided by the temporal fil-
ter, the proportion of time spent on the rasterization pro-
cess sharply decreases relative to the total rendering time.
Therefore, the time consumed by preliminary preparation
stages has not gradually become negligible. We hope that
future work will focus on optimizing these additional op-
erations within the rendering module to improve its com-
putational performance. Moreover, during the pruning pro-
cess, we specified a predefined pruning ratio. This prun-
ing ratio is influenced by the inherent characteristics of the
scene. As shown in Fig. 10, an improper pruning ratio will
cause a sharp drop in rendering quality. Therefore, identi-
fying the minimal number of Gaussians required to main-
tain high-quality rendering across different scenes remains
a challenge. Lastly, there is a significant amount of existing
work on Gaussian-based novel view synthesis for dynamic
scenes, whereas our model is specifically tailored to a par-
ticular model, 4DGS [40]. Therefore, developing a univer-
sal compression method for these Gaussian-based models is
a promising direction for subsequent research endeavors.
11

---

## Page 12

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
Pruning Ratio32.7032.8032.9033.0033.1033.2033.30PSNR(dB)
Default Setting(a) Cook Spinach
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
Pruning Ratio33.2033.3033.4033.5033.6033.7033.8033.90PSNR(dB)
Default Setting (b) Cut Roasted Beef
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
Pruning Ratio33.2033.3033.4033.5033.6033.7033.80PSNR(dB)
Default Setting (c) Sear Steak
Figure 10. Rate-distortion curves evaluated on diverse scenes with different pruning ratios.
0 10 20 30 40 50 60
Interval31.5031.7532.0032.2532.5032.7533.0033.25PSNR(dB)
F ilter(w .  F inetuning)
F ilter(w .o F inetuning)
Default Setting
(a) Cook Spinach
0 10 20 30 40 50 60
Interval32.0032.2532.5032.7533.0033.2533.5033.75PSNR(dB)
F ilter(w .  F inetuning)
F ilter(w .o F inetuning)
Default Setting (b) Cut Roasted Beef
0 10 20 30 40 50 60
Interval32.8033.0033.2033.4033.6033.80PSNR(dB)
F ilter(w .  F inetuning)
F ilter(w .o F inetuning)
Default Setting (c) Sear Steak
Figure 11. Rate-distortion curves evaluated on diverse scenes with different key-frames interval.
Table 5. Per-scene results of N3V datasets.
Scene Coffee Martini Cook Spinach Cut Roasted Beef Flame Salmon Flame Steak Sear Steak Average
4DGSPSNR 27.9286 33.1651 33.8849 29.1009 33.7970 33.6031 31.9133
SSIM 0.9160 0.9545 0.9589 0.9236 0.9615 0.9607 0.9459
LPIPS 0.0759 0.0449 0.0408 0.0691 0.0383 0.0418 0.0518
Storage(MB) 2764 2211 1863 2969 1536 1167 2085
FPS 43 89 103 31 122 152 90
Raster FPS 75 103 122 70 148 195 118
#NUM 4441271 3530165 2979832 4719443 2457356 1870891 3333160
OursPSNR 28.5780 33.2613 33.6092 28.8488 33.2804 33.7150 31.8821
SSIM 0.9185 0.9553 0.9570 0.9221 0.9598 0.9615 0.9457
LPIPS 0.0726 0.0459 0.0435 0.0707 0.0417 0.0401 0.0524
Storage(MB) 557.4 443.11 374.05 592.4 308.4 234.8 418.36
FPS 696 803 853 680 864 935 805
Raster FPS 901 1088 1163 879 1189 1332 1092
#NUM 888254 706033 595967 943889 491471 374178 666632
Ours-PPPSNR 28.5472 33.0641 33.7767 28.9878 33.2519 33.6053 31.8722
SSIM 0.9166 0.9540 0.9562 0.9209 0.9581 0.9604 0.9444
LPIPS 0.0744 0.0467 0.0445 0.0712 0.0421 0.0402 0.0532
Storage(MB) 64.94 52.04 44.54 69.24 36.94 29.34 49.50
FPS 696 803 853 680 864 935 805
Raster FPS 901 1088 1163 879 1189 1332 1092
#NUM 888254 706033 595967 943889 491471 374178 666632
12

---

## Page 13

Ground Truth 4DGS Ours Ours-PP
(a) Results on Coffee Martini Scene.
(b) Results on Cook Spinach Scene.
(c) Results on Cut Roasted Beef Scene.
Figure 12. Qualitative comparisons of 4DGS and our method on the N3V dataset. To be continued in the next page.
13

---

## Page 14

Ground Truth 4DGS Ours Ours-PP
(a) Results on Flame Salmon Scene.
(b) Results on Flame Steak Scene.
(c) Results on Sear Steak Scene.
Figure 13. Qualitative comparisons of 4DGS and our method on the N3V dataset.
14

---

## Page 15

Ground Truth 4DGS Ours Ours-PP
(a) Results on Bouncingballs Scene.
(b) Results on Hellwarrior Scene.
(c) Results on Hook Scene.
Figure 14. Qualitative comparisons of 4DGS and our method on the D-nerf dataset. To be continued in the next page.
15

---

## Page 16

Ground Truth 4DGS Ours Ours-PP
(a) Results on Jumpingjacks Scene.
(b) Results on Lego Scene.
(c) Results on Mutant Scene.
Figure 15. Qualitative comparisons of 4DGS and our method on the D-nerf dataset. To be continued in the next page.
16

---

## Page 17

Ground Truth 4DGS Ours Ours-PP
(a) Results on Standup Scene.
(b) Results on Trex Scene.
Figure 16. Qualitative comparisons of 4DGS and our method on the D-nerf dataset.
Table 6. Per-scene results of D-NeRF datasets.
Scene Bouncingballs Hellwarrior Hook Jumpingjacks Lego Mutant Standup Trex Average
4DGSPSNR 33.3472 34.7296 31.9369 30.8247 25.3320 38.9257 39.0411 29.8542 32.9989
SSIM 0.9821 0.9516 0.9635 0.9684 0.9178 0.9903 0.9896 0.9795 0.9678
LPIPS 0.0252 0.0652 0.0385 0.0340 0.0819 0.0090 0.0094 0.0193 0.0353
Storage(MB) 83.69 156.53 164.91 510.99 351.19 73.24 95.38 791.66 278.45
FPS 462 426 414 267 317 463 457 202 376
Raster FPS 1951 1433 1309 489 634 1861 1878 302 1232
#NUM 133762 250201 263593 816773 561357 117062 152454 1265408 445076
OursPSNR 33.4532 35.0316 32.5118 31.8045 26.8319 37.1916 39.3990 30.4726 33.3370
SSIM 0.9826 0.9530 0.9653 0.9716 0.9280 0.9886 0.9896 0.9811 0.9699
LPIPS 0.0248 0.0644 0.035 0.0322 0.0674 0.0124 0.0099 0.0180 0.0330
Storage(MB) 12.56 23.38 24.63 76.19 52.45 10.97 14.25 118.24 41.58
FPS 1509 1517 1444 1491 1318 1518 1539 1361 1462
Raster FPS 2600 2665 2634 2476 2067 2598 2644 2174 2482
#NUM 20065 37368 39360 121776 83837 17527 22768 188986 66460
Ours-PPPSNR 33.4592 35.1570 32.5498 31.8467 27.2850 37.0218 39.0713 30.6063 33.3746
SSIM 0.9821 0.9537 0.9671 0.9728 0.9315 0.9883 0.9896 0.9821 0.9709
LPIPS 0.0259 0.0629 0.0345 0.0309 0.0646 0.0139 0.0109 0.0173 0.0326
Storage(MB) 4.12 5.29 5.39 11.04 8.48 3.56 3.88 16.11 7.23
FPS 1509 1517 1444 1491 1318 1518 1539 1361 1462
Raster FPS 2600 2665 2634 2476 2067 2598 2644 2174 2482
#NUM 20065 37368 39360 121776 83837 17527 22768 188986 66460
17

---

## Page 18

References
[1] Muhammad Salman Ali, Maryam Qamar, Sung-Ho
Bae, and Enzo Tartaglione. Trimming the fat: Effi-
cient compression of 3d gaussian splats through prun-
ing. arXiv preprint arXiv:2406.18214 , 2024. 3
[2] Benjamin Attal, Jia-Bin Huang, Christian Richardt,
Michael Zollhoefer, Johannes Kopf, Matthew
O’Toole, and Changil Kim. Hyperreel: High-fidelity
6-dof video with ray-conditioned sampling. In Pro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 16610–16620,
2023. 1, 2, 6, 7
[3] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun
Lee, Gun Bang, and Youngjung Uh. Per-gaussian
embedding-based deformation for deformable 3d
gaussian splatting. arXiv preprint arXiv:2404.03613 ,
2024. 2, 6, 7, 8
[4] Ang Cao and Justin Johnson. Hexplane: A fast rep-
resentation for dynamic scenes. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 130–141, 2023. 2
[5] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu,
and Hao Su. Tensorf: Tensorial radiance fields. In
European conference on computer vision , pages 333–
350. Springer, 2022. 2
[6] Devikalyan Das, Christopher Wewer, Raza Yunus,
Eddy Ilg, and Jan Eric Lenssen. Neural parametric
gaussians for monocular non-rigid object reconstruc-
tion. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages
10715–10725, 2024. 2
[7] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He,
Wenzheng Chen, and Baoquan Chen. 4d-rotor gaus-
sian splatting: towards efficient novel view synthesis
for dynamic scenes. In ACM SIGGRAPH 2024 Con-
ference Papers , pages 1–11, 2024. 6
[8] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu,
Dejia Xu, and Zhangyang Wang. Lightgaussian: Un-
bounded 3d gaussian compression with 15x reduction
and 200+ fps. arXiv preprint arXiv:2311.17245 , 2023.
3, 4, 5
[9] Guangchi Fang and Bing Wang. Mini-splatting: Rep-
resenting scenes with a constrained number of gaus-
sians. arXiv preprint arXiv:2403.14166 , 2024. 3, 4
[10] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie,
Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, and
Qi Tian. Fast dynamic radiance fields with time-aware
neural voxels. In SIGGRAPH Asia 2022 Conference
Papers , pages 1–9, 2022. 6
[11] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qin-
hong Chen, Benjamin Recht, and Angjoo Kanazawa.
Plenoxels: Radiance fields without neural networks.InProceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition , pages 5501–
5510, 2022. 2
[12] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rah-
bæk Warburg, Benjamin Recht, and Angjoo
Kanazawa. K-planes: Explicit radiance fields in
space, time, and appearance. In Proceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 12479–12488, 2023. 1, 2,
6, 7
[13] Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, and
Houqiang Li. Motion-aware 3d gaussian splatting for
efficient dynamic scene reconstruction. arXiv preprint
arXiv:2403.11447 , 2024. 2
[14] Bernhard Kerbl, Georgios Kopanas, Thomas
Leimk ¨uhler, and George Drettakis. 3d gaussian
splatting for real-time radiance field rendering. ACM
Trans. Graph. , 42(4):139–1, 2023. 2, 3, 4
[15] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hin-
ton. Imagenet classification with deep convolutional
neural networks. Advances in neural information pro-
cessing systems , 25, 2012. 6
[16] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan
Ko, and Eunbyung Park. Compact 3d gaussian rep-
resentation for radiance field. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 21719–21728, 2024. 3, 6, 8
[17] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and
Ping Tan. Streaming radiance fields for 3d video syn-
thesis. Advances in Neural Information Processing
Systems , 35:13485–13498, 2022. 2, 6, 7
[18] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Si-
mon Green, Christoph Lassner, Changil Kim, Tan-
ner Schmidt, Steven Lovegrove, Michael Goesele,
Richard Newcombe, et al. Neural 3d video synthe-
sis from multi-view video. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 5521–5531, 2022. 1, 2, 6, 7,
8, 9, 10
[19] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Space-
time gaussian feature splatting for real-time dynamic
view synthesis. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition ,
pages 8508–8520, 2024. 6, 7
[20] Wenkai Liu, Tao Guan, Bin Zhu, Lili Ju, Zikai
Song, Dan Li, Yuesong Wang, and Wei Yang. Ef-
ficientgs: Streamlining gaussian splatting for large-
scale high-resolution scene representation. arXiv
preprint arXiv:2404.12777 , 2024. 3
[21] Stephen Lombardi, Tomas Simon, Jason Saragih,
Gabriel Schwartz, Andreas Lehrmann, and Yaser
Sheikh. Neural volumes: Learning dynamic ren-
18

---

## Page 19

derable volumes from images. arXiv preprint
arXiv:1906.07751 , 2019. 1, 2, 6, 7
[22] Zhicheng Lu, Xiang Guo, Le Hui, Tianrui Chen,
Min Yang, Xiao Tang, Feng Zhu, and Yuchao Dai.
3d geometry-aware deformable gaussian splatting for
dynamic view synthesis. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 8900–8910, 2024. 2
[23] Jonathon Luiten, Georgios Kopanas, Bastian Leibe,
and Deva Ramanan. Dynamic 3d gaussians: Tracking
by persistent dynamic view synthesis. arXiv preprint
arXiv:2308.09713 , 2023. 6, 7, 8
[24] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-
Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi,
Ren Ng, and Abhishek Kar. Local light field fusion:
Practical view synthesis with prescriptive sampling
guidelines. ACM Transactions on Graphics (ToG) , 38
(4):1–14, 2019. 2
[25] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng.
Nerf: Representing scenes as neural radiance fields for
view synthesis. Communications of the ACM , 65(1):
99–106, 2021. 1, 2
[26] Thomas M ¨uller, Alex Evans, Christoph Schied, and
Alexander Keller. Instant neural graphics primitives
with a multiresolution hash encoding. ACM transac-
tions on graphics (TOG) , 41(4):1–15, 2022. 2
[27] KL Navaneet, Kossar Pourahmadi Meibodi,
Soroush Abbasi Koohpayegani, and Hamed Pir-
siavash. Compgs: Smaller and faster gaussian
splatting with vector quantization. In European
Conference on Computer Vision , 2024. 3, 7
[28] Michael Niemeyer, Fabian Manhardt, Marie-Julie
Rakotosaona, Michael Oechsle, Daniel Duckworth,
Rama Gosula, Keisuke Tateno, John Bates, Do-
minik Kaeser, and Federico Tombari. Radsplat: Ra-
diance field-informed gaussian splatting for robust
real-time rendering with 900+ fps. arXiv preprint
arXiv:2403.13806 , 2024. 3
[29] Panagiotis Papantonakis, Georgios Kopanas, Bern-
hard Kerbl, Alexandre Lanvin, and George Drettakis.
Reducing the memory footprint of 3d gaussian splat-
ting. Proceedings of the ACM on Computer Graphics
and Interactive Techniques , 7(1):1–17, 2024. 3
[30] Albert Pumarola, Enric Corona, Gerard Pons-Moll,
and Francesc Moreno-Noguer. D-nerf: Neural radi-
ance fields for dynamic scenes. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 10318–10327, 2021. 2, 6, 7,
8, 9, 11
[31] Christian Reiser, Songyou Peng, Yiyi Liao, and An-
dreas Geiger. Kilonerf: Speeding up neural radiance
fields with thousands of tiny mlps. In Proceedings ofthe IEEE/CVF international conference on computer
vision , pages 14335–14345, 2021. 2
[32] Katja Schwarz, Axel Sauer, Michael Niemeyer, Yiyi
Liao, and Andreas Geiger. V oxgraf: Fast 3d-aware
image synthesis with sparse voxel grids. Advances
in Neural Information Processing Systems , 35:33999–
34011, 2022. 2
[33] Karen Simonyan and Andrew Zisserman. Very deep
convolutional networks for large-scale image recogni-
tion. arXiv preprint arXiv:1409.1556 , 2014. 6
[34] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen,
Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger.
Nerfplayer: A streamable dynamic scene representa-
tion with decomposed neural radiance fields. IEEE
Transactions on Visualization and Computer Graph-
ics, 29(5):2732–2742, 2023. 2
[35] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct
voxel grid optimization: Super-fast convergence for
radiance fields reconstruction. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition , pages 5459–5469, 2022. 2
[36] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian,
Yafei Song, and Huaping Liu. Mixed neural voxels
for fast multi-view video synthesis. In Proceedings of
the IEEE/CVF International Conference on Computer
Vision , pages 19706–19716, 2023. 2
[37] Huan Wang, Jian Ren, Zeng Huang, Kyle Olszewski,
Menglei Chai, Yun Fu, and Sergey Tulyakov. R2l:
Distilling neural radiance field to neural light field for
efficient novel view synthesis. In European Confer-
ence on Computer Vision , pages 612–629. Springer,
2022. 2
[38] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang
Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu,
Jingyi Yu, and Lan Xu. Fourier plenoctrees for dy-
namic radiance field rendering in real-time. In Pro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 13524–13534,
2022. 2
[39] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie,
Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian,
and Xinggang Wang. 4d gaussian splatting for real-
time dynamic scene rendering. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 20310–20320, 2024. 2, 6, 7
[40] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang.
Real-time photorealistic dynamic scene representa-
tion and rendering with 4d gaussian splatting. arXiv
preprint arXiv:2310.10642 , 2023. 1, 2, 3, 5, 6, 7, 8, 9,
11
[41] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao,
Yuqing Zhang, and Xiaogang Jin. Deformable 3d
gaussians for high-fidelity monocular dynamic scene
19

---

## Page 20

reconstruction. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition ,
pages 20331–20341, 2024. 2, 6
[42] Richard Zhang, Phillip Isola, Alexei A Efros, Eli
Shechtman, and Oliver Wang. The unreasonable ef-
fectiveness of deep features as a perceptual metric. In
Proceedings of the IEEE conference on computer vi-
sion and pattern recognition , pages 586–595, 2018. 6
[43] Xinjie Zhang, Zhening Liu, Yifan Zhang, Xingtong
Ge, Dailan He, Tongda Xu, Yan Wang, Zehong Lin,
Shuicheng Yan, and Jun Zhang. Mega: Memory-
efficient 4d gaussian splatting for dynamic scenes.
arXiv preprint arXiv:2410.13613 , 2024. 6, 8
20