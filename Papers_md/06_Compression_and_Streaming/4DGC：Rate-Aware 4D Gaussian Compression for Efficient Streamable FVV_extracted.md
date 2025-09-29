

---

## Page 1

4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable
Free-Viewpoint Video
Qiang Hu1∗Zihan Zheng1∗Houqiang Zhong2Sihua Fu1Li Song2
XiaoyunZhang1†Guangtao Zhai2Yanfeng Wang3†
1Cooperative Medianet Innovation Center, Shanghai Jiao Tong University
2School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University
3School of Artificial Intelligence, Shanghai Jiao Tong University
TeTriRF32.13dB/0.7MB
Ours33.60dB/0.5MB3DGStream33.58dB/8.1MBReRF30.03dB/0.6MB
PSNR(dB)
Bitrate (MB/Frame)MixvoxelsK-PlanesOursStreamRF
Ours28.53dB/0.3MB
Ours29.67dB/0.9MBOurs29.13dB/0.4MB
Ours29.82dB/1.1MBD-3DGHyperReel3DGStream
ReRFTeTriRF
TeTriRF32.03dB/0.6MB
Ours33.60dB/0.5MB3DGStream33.58dB/8.1MBReRF30.03dB/0.5MB
Figure 1. Left: 4DGC results, showcasing flexible quality levels across various bitrates. Middle : Comparison of visual quality and bitrate
with state-of-the-art methods. Right : The RD performance of our approach surpasses that of prior work (e.g. 3DGStream [61], ReRF [64],
TeTriRF [69]).
Abstract
3D Gaussian Splatting (3DGS) has substantial potential
for enabling photorealistic Free-Viewpoint Video (FVV) ex-
periences. However, the vast number of Gaussians and their
associated attributes poses significant challenges for stor-
age and transmission. Existing methods typically handle
dynamic 3DGS representation and compression separately,
neglecting motion information and the rate-distortion (RD)
trade-off during training, leading to performance degra-
dation and increased model redundancy. To address this
gap, we propose 4DGC, a novel rate-aware 4D Gaussian
compression framework that significantly reduces storage
size while maintaining superior RD performance for FVV .
Specifically, 4DGC introduces a motion-aware dynamic
Gaussian representation that utilizes a compact motion grid
combined with sparse compensated Gaussians to exploit
inter-frame similarities. This representation effectively han-
dles large motions, preserving quality and reducing tem-
∗These authors contributed equally.†The corresponding au-
thors are Xiaoyun Zhang (xiaoyun.zhang@sjtu.edu.cn) and Yanfeng Wang
(wangyanfeng622@sjtu.edu.cn).poral redundancy. Furthermore, we present an end-to-end
compression scheme that employs differentiable quantiza-
tion and a tiny implicit entropy model to compress the mo-
tion grid and compensated Gaussians efficiently. The en-
tire framework is jointly optimized using a rate-distortion
trade-off. Extensive experiments demonstrate that 4DGC
supports variable bitrates and consistently outperforms ex-
isting methods in RD performance across multiple datasets.
1. Introduction
Free-Viewpoint Video (FVV) enables immersive real-time
navigation of scenes from any perspective, enhancing user
engagement with high interactivity and realism. This makes
FVV ideal for applications in entertainment, virtual reality,
sports broadcasting, and telepresence. However, streaming
and rendering high-quality FVV remains challenging, par-
ticularly for sequences with large motions, complex back-
grounds, and extended durations. The primary difficulty lies
in developing an efficient representation and compression
method for FVV that supports streaming with limited bi-
trate while maintaining high fidelity.
Traditional approaches to FVV reconstruction have pri-
1arXiv:2503.18421v1  [cs.CV]  24 Mar 2025

---

## Page 2

marily relied on point cloud-based methods [22] and depth-
based techniques [9], which struggle to deliver high render-
ing quality and realism, especially in complex scenes. Neu-
ral Radiance Fields (NeRF) and its variants [11, 18, 20, 26,
30, 45, 54] have demonstrated impressive results in recon-
structing FVV by learning continuous 3D scene representa-
tions, yet they face limitations in supporting long sequences
and streaming. Recent approaches [64, 65, 69, 76, 77] ad-
dress these issues by compressing explicit features of dy-
namic NeRF. For example, TeTriRF [69] employs a hy-
brid representation with tri-planes to model dynamic scenes
and applies a traditional video codec to further reduce re-
dundancy. However, these methods often suffer from slow
training and rendering speeds.
Recently, 3D Gaussian Splatting (3DGS) [29] has
demonstrated exceptional rendering speed and quality com-
pared to NeRF-based approaches for static scenes. Several
methods [35, 68] have attempted to extend 3DGS to dy-
namic settings by incorporating temporal correspondence
or time-dependency, but these approaches require loading
all frames into memory for training and rendering, limiting
their practicality for streaming applications. 3DGStream
[61] models the inter-frame transformation and rotation of
3D Gaussians as a neural transformation cache, which re-
duces per-frame storage requirements for FVV . However,
the overall data volume remains substantial, hindering its
ability to support efficient FVV transmission. Although
a few studies [27, 28] have explored the compression of
dynamic 3DGS, these methods heavily struggle with real-
world dynamic scenes containing backgrounds, which lim-
its their practical effectiveness. Additionally, they optimize
representation and compression independently, overlooking
the rate-distortion (RD) trade-off during training, which ul-
timately restricts compression efficiency.
In this paper, we propose 4DGC, a novel rate-aware
compression method tailored for Gaussian-based FVV . Our
key idea is to explicitly model motion between adjacent
frames, estimate the bitrate of the 4D Gaussian represen-
tation during training, and incorporate rate and distortion
terms into the loss function to enable end-to-end optimiza-
tion. This allows us to achieve a compact and compression-
friendly representation with optimal RD performance. We
realize this through two main innovations. First, we intro-
duce a motion-aware dynamic Gaussian representation for
inter-frame modeling within long sequences. We train the
3DGS on the first frame (keyframe) to obtain the initial ref-
erence Gaussians. For each subsequent frame, 4DGC uti-
lizes a compact multi-resolution motion grid to estimate the
rigid motion of each Gaussian from the previous frame to
the current one. Additionally, compensated Gaussians are
sparsely added to account for newly observed regions or ob-
jects, further enhancing the accuracy of the representation.
By leveraging inter-frame similarities, 4DGC effectively re-duces temporal redundancy and ensures that the representa-
tion remains compact while maintaining high visual fidelity.
Second, we propose a unified end-to-end compression
scheme that efficiently encodes the initial Gaussian spher-
ical harmonics (SH) coefficients, the motion grid, and the
compensated Gaussian SH coefficients. This compression
approach incorporates differentiable quantization to facil-
itate gradient back-propagation and a tiny implicit entropy
model for accurate bitrate estimation. By optimizing the en-
tire scheme through a rate-distortion trade-off during train-
ing, we significantly enhance compression performance.
Experimental results show that our 4DGC supports vari-
able bitrates and achieves state-of-the-art RD performance
across various datasets. Compared to the SOTA method
3DGStream [61], our approach achieves approximately a
16xcompression rate without quality degradation, as illus-
trated in Fig. 1.
In summary, our key contributions include:
• We present a compact and compression-friendly 4D
Gaussian representation for streamable FVV that effec-
tively captures dynamic motion and compensates for
newly emerging objects, minimizing temporal redun-
dancy and enhancing reconstruction quality.
• We introduce an end-to-end 4D Gaussian compression
framework that jointly optimizes representation and en-
tropy models using a rate-distortion loss function, ensur-
ing a low-entropy 4D Gaussian representation and signif-
icantly enhancing RD performance.
• Extensive experimental results on the real-world datasets
demonstrate that our 4DGC achieves superior reconstruc-
tion quality, bitrate efficiency, training time, and render-
ing speed compared to existing state-of-the-art dynamic
scene compression methods.
2. Related Work
Dynamic Modeling with NeRF. Building on NeRF’s suc-
cess in static scene synthesis [5–7, 13, 43, 45, 47, 50, 55,
56], several works have extended these methods to dynamic
scenes. Flow-based methods [33, 34] construct 3D features
from monocular videos with impressive results, at the cost
of more extra priors like depth and motion for complex
scenes. Deformation field methods [16, 30, 49, 54, 60] warp
frames to a canonical space to capture temporal features but
suffer from slow training and rendering speeds. To accel-
erate the speeds, some methods [11, 18, 20, 26, 32, 51, 58,
62, 63] extend the radiance field into four dimensions us-
ing grid representation, plane-based representation or ten-
sor factorization. However, these methods typically suffer
from storage efficiency and are not suitable for streaming.
Dynamic Modeling with 3DGS. Recent advancements
in 3DGS [29] and its variants [12, 19, 21, 24, 25] have
achieved photorealistic static scene rendering with high effi-
ciency. However, for dynamic scenes, the per-frame 3DGS
2

---

## Page 3

Figure 2. Illustration of the 4DGC Framework. The reconstructed Gaussians from the previous frame, ˆGt−1, are retrieved from the
reference buffer and combined with the input images of the current frame to facilitate learning of the motion grid Mtand the compensated
Gaussians ∆Gtthrough a two-stage training process. In the first stage, the motion grid and its associated entropy model are optimized. In
the second stage, the compensated Gaussians are refined along with their corresponding entropy model. Both stages are supervised by a
rate-distortion trade-off, employing simulated quantization and an entropy model to jointly optimize representation and compression.
approach neglects temporal consistency, causing visual ar-
tifacts and model size growth. Some methods [23, 35, 68,
70, 73, 74] model Gaussian attributes over time to represent
dynamic scenes as a unified model, improving quality but
requiring the simultaneous loading of all data, which limits
practical use in long-sequence streaming. Other methods
[23, 40, 61] track Gaussian motion frame by frame, which
is suitable for streaming, but the large size of each frame
hinders transmission efficiency. In contrast, our approach
employs a compact multi-resolution motion grid combined
with per-frame Gaussian compensation, reducing temporal
redundancy and enhancing reconstruction quality.
Dynamic Scene Compression. Recent advances in deep
learning-based image and video compression [1, 3, 4, 14,
31, 36, 38, 41, 42, 44, 59, 71, 72, 75] have demonstrated
strong RD performance. In FVV compression, current ap-
proaches [15, 53, 57, 60, 64, 65, 69, 76, 77] primarily focus
on compressing dynamic NeRF features to improve stor-
age and transmission efficiency. Techniques like ReRF [64]
and TeTriRF [69] apply traditional image/video encoding
methods to dynamic scenes without end-to-end optimiza-
tion, sacrificing dynamic detail and compression efficiency.
Some approaches [76, 77] achieve end-to-end optimization
but struggle with scalability in open scenes and slow ren-
dering. For 3DGS-based methods, most [17, 39, 48, 66]
focus on static scene compression, while dynamic scene
techniques [27, 28] remain limited and typically support
only background-free scenarios without comprehensive op-
timization. Our method achieves both high RD performance
and fast decoding and rendering times in real-world scenar-
ios thanks to our proposed motion-aware dynamic Gaussian
representation and end-to-end joint compression.3. Method
In this section, we introduce the details of the 4DGC frame-
work. Fig. 2 illustrates the overall architecture of 4DGC.
Our approach begins with a motion-aware dynamic Gaus-
sian representation, composed of a compact motion grid and
sparse compensated Gaussians (Sec. 3.1). Subsequently,
we describe a two-stage method combining motion estima-
tion and Gaussian compensation to generate this represen-
tation, effectively capturing both spatial and temporal vari-
ations (Sec. 3.2). Finally, we introduce an end-to-end com-
pression scheme that jointly optimizes representation and
entropy models, ensuring a low-entropy representation and
greatly improving RD performance (Sec. 3.3).
3.1. Motion-aware Dynamic Gaussian Modeling
Recall that 3DGS represents scenes using a collection G
of Gaussian primitives as an explicit representation similar
to point clouds. Each Gaussian G∈Gis defined by a
set of optimizable parameters {µ;R;f;s;α}, where µis
the center location, Ris the rotation matrix, frepresents
SH coefficients for view-dependent color c,sis the scaling
vector, and αis the opacity. For a point xlocated within a
Gaussian primitive, the spatial distribution is determined by
G(x),
G(x) = exp
−1
2(x−µ)TΣ−1(x−µ)
(1)
where Σ=RssTRT. The rendering color cof a pixel is
computed by alpha-blending the NGaussians overlapping
the pixel in depth order:
c=X
i∈Nciα′
ii−1Y
j=1(1−α′
j) (2)
3

---

## Page 4

Figure 3. Illustration of our motion-aware dynamic Gaussian mod-
eling that utilizes a multi-resolution motion grid Mtwith sparse
compensated Gaussians ∆Gtto exploit inter-frame similarities.
where α′
iis the projection from the i-th Gaussian opacity
onto the image plane, ciis the i-th Gaussian color in view-
ing direction.
When extending the representation from static to dy-
namic scenes, a straightforward approach is stacking frame-
wise static Gaussians to form a dynamic sequence. How-
ever, this method neglects temporal coherence, resulting in
significant temporal redundancy, particularly in scenes with
dense Gaussian primitives. Alternative methods [35, 68] ex-
tend Gaussians to 4D space for modeling the entire dynamic
scene. Such approaches suffer from performance degrada-
tion in long sequences and are not suitable for streaming
applications. To overcome these limitations, we propose a
motion-aware dynamic Gaussian representation, which ex-
plicitly models and tracks motion between adjacent frames
to maintain spatial and temporal coherence.
Our modeling approach employs a complete 3DGS rep-
resentation as the initial Gaussians G1for the first frame
(keyframe). For each subsequent frame, we utilize a
multi-resolution motion grid Mtwith two shared global
lightweight MLPs, ΦµandΦR, to estimate the rigid mo-
tion of each Gaussian from the previous frame to the cur-
rent one. This grid captures the multi-scale nature of mo-
tion, enabling precise modeling even for objects that move
at varying speeds or directions. However, rigid transforma-
tion alone is insufficient for accurately representing newly
emerging regions. To address this, we dynamically add
sparse compensated Gaussians ∆Gtto account for newly
observed regions in the current frame. Finally, our 4DGC
sequentially represents the dynamic scene with N frames
asG1,{Mt,∆Gt}N
t=2,Φµ, and ΦRas shown in Fig. 3.
A major benefit of this design is that 4DGC fully exploits
inter-frame similarities, reducing temporal redundancy and
enhancing reconstruction quality.3.2. Sequential Representation Generation
Here, we introduce our two-stage scheme, which combines
motion estimation and Gaussian compensation to generate
a dynamic Gaussian representation that effectively captures
both spatial and temporal variations in the scene. This pro-
cess begins with motion estimation, which tracks and mod-
els frame-by-frame transformations in both translation and
rotation, reducing inter-frame redundancy. To address sce-
narios where newly emerging objects or complex motion
cannot be fully captured through estimation alone, a gaus-
sian compensation step is applied to refine representation
quality in suboptimal areas. Together, these stages form a
flexible and high-fidelity representation that improves com-
pression efficiency and supports high-quality rendering for
streamable applications. We detail each stage below.
Motion Estimation. For each inter-frame, we load the
reconstructed Gaussians of the previous frame ˆGt−1from
the reference buffer, providing a stable reference for track-
ing transformations in the current frame. By combining
these reference Gaussians with the input images of the cur-
rent frame, we employ a motion grid, Mt, along with two
shared lightweight MLPs, ΦµandΦR, to predict the trans-
lation ( ∆µt) and rotation ( ∆Rt) for each Gaussian. Specif-
ically, to achieve accurate motion estimation, we use a
multi-resolution motion grid Mt={Ml
t}L
l=1, where Lde-
notes the number of resolution levels, to capture complex
motions across various scales. For a Gaussian primitive
G∈ˆGt−1in the previous reconstructed frame, its center
location µt−1is mapped to multiple frequency bands Pt−1
via positional encoding:
Pt−1={Pl
t−1}L
l=1={sin(2lπµt−1),cos(2lπµt−1)}L
l=1
(3)
At each level, we use the mapped position to perform trilin-
ear interpolation on Mt, producing motion features across
different scales. These multi-scale features are then con-
catenated and fed into two lightweight MLPs, ΦµandΦR,
to compute translation ∆µtand rotation ∆Rtfor each
Gaussian. Thus, our motion estimation is formalized as fol-
lows:
∆µt= Φµ L[
l=1interp (Pl
t−1,Ml
t)!
∆Rt= ΦR L[
l=1interp (Pl
t−1,Ml
t)! (4)
where interp (·)represents the grid interpolation operation.
With the translation ∆µtand rotation ∆Rt, transfor-
mations are applied to each Gaussian in ˆGt−1, achieving
smooth alignment from the previous to the current frame:
G′
t=ˆGt−1(G⊕Mt(G))
={G(µt−1+ ∆µt; ∆RtRt−1;C)|G∈ˆGt−1}(5)
4

---

## Page 5

where G′
tdenotes the transformed Gaussians for the cur-
rent frame, Crepresents the fixed parameters including
ft−1,st−1, and αt−1.⊕denotes the operation of updat-
ing position µand rotation Rfor each Gaussian GinˆGt−1
according to Mt. This hierarchical approach achieves pre-
cise motion prediction across multiple scales, capturing es-
sential transformations and effectively reducing inter-frame
redundancy.
Gaussian Compensation. Although motion estimation
effectively captures the dynamics of previously observed
objects within a scene, we found that relying solely on mo-
tion estimation is insufficient for achieving high-quality de-
tail, particularly in cases involving newly emerging objects
and subtle motion transformations. To address this limita-
tion, we adopt a Gaussian compensation strategy, refining
representation quality by integrating sparse compensated
Gaussians ∆GtintoG′
tin suboptimal regions.
We first identify the suboptimal areas requiring compen-
sation. These regions are classified into two primary types:
(1) regions with significant gradient changes, typically cor-
responding to newly appearing objects or scene edges, and
(2) larger Gaussian primitives undergoing rapid transforma-
tions, leading to multiview perspective differences. For the
first type, we apply gradient thresholding, cloning the Gaus-
sian primitives at locations where the gradient exceeds a
predefined threshold τgas∆Gg
t, ensuring accurate repre-
sentation of newly observed elements. For the second type,
involving larger Gaussians impacted by rapid transforma-
tions, we clone two additional Gaussian primitives from the
original Gaussian when its motion parameters exceed speci-
fied thresholds: τµfor translation |∆µt|andτRfor rotation
|∆Rt|. The scale of two cloned Gaussians is reduced tos
100
to capture detailed motion dynamics more precisely.
These newly compensated Gaussians are sampled in
N(µ,2Σ)around the original Gaussian and optimized in
the second training stage. This Gaussian splitting pro-
cess yields a fine-grained and adaptive representation, cap-
turing complex motion patterns and enhancing continuity
across frames. Overall, this compensation strategy signif-
icantly improves detail accuracy, reduces artifacts, and en-
sures high-quality reconstruction of dynamic scenes.
3.3. End-to-end Joint Compression
We also propose an end-to-end 4D Gaussian compression
framework that jointly optimizes representation and entropy
models through a two-stage training process. To enable gra-
dient back-propagation, we employ differentiable quantiza-
tion, along with a compact implicit entropy model for ac-
curate bitrate estimation of the motion grid Mtand com-
pensated Gaussians ∆Gt. The first stage focuses on op-
timizing the motion grid alongside its associated entropy
model, while the second stage refines the compensated
Gaussians with their corresponding entropy model. Eachstage is guided by a rate-distortion trade-off, ensuring a
low-entropy 4D Gaussian representation and substantially
improving RD performance.
Simulated Quantization & Rate Estimation. Quanti-
zation and entropy coding effectively reduce the bitrate dur-
ing compression at the expense of some information loss.
However, the rounding operation in quantization prevents
gradient propagation, which is incompatible with end-to-
end training. To address this, we implement a differentiable
quantization strategy using simulated quantization noise.
Specifically, uniform noise u∼U
−1
2q,1
2q
is added to
simulate quantization effects with step size q, enabling ro-
bust training while preserving gradient flow. For rate esti-
mation, we use a tiny and trainable implicit entropy model
[8] to approximate the probability mass function (PMF) of
the quantized values, ˆy. Unlike the learned entropy model in
image compression [46], which is learned from large-scale
training datasets, our implicit entropy model is learned on-
the-fly with the corresponding 4D Gaussian representation
in the training. The PMF is derived using the cumulative
distribution function (CDF) as follows:
PPMF(ˆy) =PCDF(ˆy+1
2)−PCDF(ˆy−1
2).(6)
Incorporating this rate estimation into the loss function en-
ables the network to learn feature distributions with in-
herently lower entropy. This effectively imposes a bitrate
constraint during training while ensuring the compatibility
of gradient-based optimization, balancing compression ef-
ficiency and model accuracy.
Stage 1: Motion Grid Compression. In the first train-
ing stage, we jointly optimize Φµ,ΦR, the multi-resolution
motion grid Mtand its corresponding entropy model. This
process enhances motion prediction accuracy while encour-
aging low-entropy characteristics in Mt. Specifically, we
apply simulated quantization to discretize the motion grid,
ensuring compatibility with entropy encoding. The entropy
model then assigns probabilities to each quantized element
based on a learned probability mass function to estimate the
bitrate of the motion grid more effectively. The loss func-
tionLs1of this stage comprises a photometric term Lcolor
and a rate term LME
rate:
Ls1=Lcolor+λ1LME
rate (7)
Lcolor= (1−λ2)∥cg−ˆc∥1+λ2LSSIM (8)
LME
rate=−1
NX
ˆy∈ˆMtlog2 
P1
PMF(ˆy)
(9)
whereLME
raterepresents the estimated rate derived from ˆMt.
cgandˆcrefer to the ground truth and reconstructed col-
ors , respectively. LSSIM is the D-SSIM[37] metric be-
tween ground truth and the result rendered by 4DGC, and
5

---

## Page 6

Figure 4. Rate-distortion curves across different datasets. Rate-distortion curves not only illustrate the superiority of our method over
ReRF [64], TeTriRF [69], and 3DGStream [61], but also demonstrate the efficiency of various components within our method.
λ2serves as a weight parameter. The parameter λ1balances
the trade-off between rate and distortion, thus controlling
model size and reconstruction quality.
Stage 2: Compensated Gaussians Compression. In
the second training stage, we focus on optimizing the com-
pensated Gaussians ∆Gtand their entropy model to en-
hance detail capture and compression efficiency. While
attributes like position and rotation are crucial for render-
ing, they require little storage. Thus, the main emphasis is
on compressing the SH coefficients, which account for the
largest storage cost.
Leveraging the trained motion grid Mtfrom Stage 1, we
transform Gaussian primitives from the previous frame to
the current frame while preserving fixed attributes like posi-
tion, rotation, and scale. To enhance representation fidelity,
we augment these transformed Gaussians with compensated
Gaussians ∆Gt. The SH coefficients of ∆Gtundergo sim-
ulated quantization and are processed by an implicit entropy
model for accurate rate estimation. The loss function Ls2
optimizes this compression process.
Ls2=Lcolor+λ1LMC
rate (10)
LMC
rate=−1
MX
ˆy∈ˆfC
tlog2 
P2
PMF(ˆy)
(11)
where ˆfC
trepresents the quantized SH coefficients of the
compensated Gaussians.This strategy is similarly applied
to the initial Gaussians G1in the keyframe. The joint op-
timization approach for representation and compression re-
sults in a compact and high-quality 4D Gaussian representa-
tion, facilitating efficient storage and transmission for FVV
applications.
Once the training of the current frame is finished, we re-
construct the complete Gaussian representation for the cur-
rent frame, ˆGt, as follows:
ˆGt=ˆGt−1(G⊕ˆMt(G)) + ∆ ˆGt (12)
where ˆMtand∆ˆGtrepresent the reconstructed motion grid
and compensated Gaussians, respectively. Finally, ˆGtis
stored in the reference buffer to facilitate the reconstruction
of the next frame.Table 1. Quantitative comparison on the N3DV [32] dataset. The
PSNR, SSIM, size, and rendering speed are averaged over the
whole 300 frames for each scene.
MethodPSNR↑
(dB)SSIM↑Size↓
(MB)Render ↑
(FPS)Streamable/
Variable-bitrate
K-Planes [20] 29.91 0.920 1.0 0.15 ✗/✗
HyperReel [2] 31.10 0.938 1.2 2.0 ✗/✗
MixV oxels [62] 30.80 0.931 1.7 16.7 ✗/✗
NeRFPlayer [60] 30.69 0.931 17.7 0.05 ✓/✗
StreamRF [30] 30.61 0.930 7.6 8.3 ✓/✗
ReRF [64] 29.71 0.918 0.77 2.0 ✓/✓
TeTriRF [69] 30.65 0.931 0.76 2.7 ✓/✓
D-3DG [40] 30.67 0.931 9.2 460 ✓/✗
3DGStream [61] 31.54 0.942 8.1 215 ✓/✗
Ours 31.58 0.943 0.5 168 ✓/✓
4. Experiments
4.1. Configurations
Datasets. We validate the effectiveness of 4DGC using
three real-world datasets: N3DV dataset [32], MeetRoom
dataset [30], and Google Immersive dataset [10]. Each
dataset reserves 1 camera view for testing, with the remain-
ing views used for training.
Implementation. Our experimental setup includes an
Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz and an RTX
3090 graphics card. We set the resolution levels of the
multi-resolution motion grid to L= 3 . During train-
ing, the initial settings are as follows: λ1is set to
0.0003,0.0001,0.00005 ,0.00001 to achieve different bi-
trates, while λ2is set to 0.2. The number of iterations for
motion estimation and Gaussian compensation is set to 400
and100, respectively.
Metrics. To evaluate the compression performance of
our method on the experimental datasets, we use Peak
Signal-to-Noise Ratio (PSNR) and Structural Similarity In-
dex (SSIM) [67] as quality metrics, along with bitrate
measured in MB per frame. For comprehensive RD per-
formance analysis, we apply Bjontegaard Delta Bit-Rate
(BDBR) and Bjontegaard Delta PSNR (BD-PSNR) [52].
Rendering efficiency is assessed by measuring the frames
rendered per second (FPS).
6

---

## Page 7

Figure 5. Qualitative comparison on the N3DV [32] and MeetRoom [30] datasets against ReRF [64], TeTriRF [69], and 3DGStream [61].
Table 2. Quantitative comparison on the MeetRoom dataset [30]
and Google Immersive dataset [10].
Dataset MethodPSNR↑
(dB)SSIM↑Size↓
(MB)Render ↑
(FPS)Streamable/
Variable-bitrate
MeetRoom
dataset [30]StreamRF [30] 26.71 0.913 8.23 10 ✓/✗
ReRF [64] 26.43 0.911 0.63 2.9 ✓/✓
TeTriRF [69] 27.37 0.917 0.61 3.8 ✓/✓
3DGStream [61] 28.03 0.921 8.21 288 ✓/✗
Ours 28.08 0.922 0.42 213 ✓/✓
Google
Immersive
dataset [10]StreamRF [30] 28.14 0.929 10.24 8.0 ✓/✗
ReRF [64] 27.75 0.928 0.93 1.4 ✓/✓
TeTriRF [69] 28.53 0.931 0.83 2.1 ✓/✓
3DGStream [61] 29.66 0.935 10.33 199 ✓/✗
Ours 29.71 0.935 0.61 145 ✓/✓
4.2. Comparison
Quantitative comparisons. To validate the effectiveness
of our method, we compare it against several state-of-the-
art approaches including K-Planes [20], HyperReel [2],
MixV oxels [62], NeRFPlayer [60], StreamRF [30], ReRF
[64], TeTriRF [69], D-3DG [40] and 3DGStream [61].
Tab. 1 shows the detailed quantitative results on the N3DV
dataset. It can be seen that our method outperforms other
methods and achieves the best reconstruction quality with
the lowest bitrate. Specifically, 3DGStream [61] requires
8.1 MB to achieve a comparable quality level to our 4DGC,
which only needs 0.5 MB . Although TeTriRF [69] achieves
a similar bitrate, its reconstruction quality is lower due
to separate optimization of representation and compres-
sion. By contrast, our approach jointly optimizes the entire
framework through a rate-distortion trade-off, which sig-
nificantly enhances compression performance. To demon-
strate the generality of our method, we conduct experiments
on the MeetRoom and Google Immersive datasets, provid-
ing a quantitative comparison against StreamRF [30], ReRF
[64], TeTriRF [69], and 3DGStream [61], as illustrated in
Tab. 2. Our method still outperforms others in PSNR,
SSIM, and bitrate.
Fig. 4 illustrates the RD curves of our 4DGC comparedto ReRF [64], TeTriRF [69], and 3DGStream [61] across
various sequences from the three datasets. The RD curves
clearly show that our 4DGC achieves the best RD perfor-
mance across a wide range of bitrates. Furthermore, we cal-
culate the BDBR relative to ReRF [64] and TeTriRF [69],
as presented in Tab. 3. On the N3DV dataset, our 4DGC
achieves an average BDBR reduction of 68.59% compared
to TeTriRF [69]. Similar BDBR savings of 40.71% and
59.99% are observed on the MeetRoom and Google Immer-
sive datasets, respectively. Against ReRF [64], our 4DGC
also demonstrates significantly better RD performance.
Tab. 4 compares the computational complexity of our
4DGC with the state-of-the-art dynamic scene compression
methods, ReRF [64] and TeTriRF [69]. Our 4DGC signif-
icantly improves computational efficiency, with a training
time of 0.83min versus 42.73min for ReRF and 1.04min for
TeTriRF. In rendering, 4DGC requires only 0.006s, vastly
outperforming ReRF (0.502s) and TeTriRF (0.375s). For
encoding and decoding, 4DGC achieves times of 0.72s and
0.09s, respectively, surpassing both ReRF and TeTriRF.
These results highlight 4DGC as a more efficient solution
for FVV compression.
Qualitative comparisons. We present a qualitative
comparison with ReRF [64], TeTriRF [69], and 3DGStream
[61] on the coffee martini sequence from the N3DV dataset
and the trimming sequence from the MeetRoom dataset,
as shown in Fig. 5. Our approach achieves compara-
ble reconstruction quality to 3DGStream [61] at a substan-
tially lower bitrate, achieving a compression rate exceeding
16x. Compared to ReRF [64] and TeTriRF [69], our 4DGC
more effectively preserves finer details such as the head,
window, bottles, and books in coffee martini and the face,
hand, plant, and scissor in trimming , which are lost in the
reconstructions of these two methods. This demonstrates
that our 4DGC captures dynamic scene elements accurately
and maintains high-quality detail in intricate objects while
7

---

## Page 8

Table 3. The BDBR and BD-PSNR results of our 4DGC and ReRF
[64] when compared with TeTriRF [69] on different datasets.
Dataset Method BDBR(%) ↓ BD-PSNR(dB) ↑
N3DV [32]ReRF 371.10 -0.78
Ours -68.59 1.12
MeetRoom [30]ReRF 134.69 -0.99
Ours -40.71 0.55
Google
Immersive [10]ReRF 324.91 -0.93
Ours -59.99 1.03
Table 4. Complexity comparison of our 4DGC method with dy-
namic scene compression methods, ReRF [64] and TeTriRF [69].
Method Train(min) Render(s) Encode(s) Decode(s)
ReRF [64] 42.73 0.502 3.03 0.28
TeTriRF [69] 1.04 0.375 0.79 0.31
4DGC 0.83 0.006 0.72 0.09
achieving a highly compact model size.
4.3. Ablation Studies
We conduct three ablation studies to evaluate the effec-
tiveness of motion estimation, Gaussian compensation, and
joint optimization of representation and compression by
disabling each component individually during training. In
the first study, we apply motion estimation but exclude
Gaussian compensation. In the second, we omit motion es-
timation, training only the compensated Gaussians for each
frame based on the previous frame. In the final study, we
separately train the motion-aware representation and en-
tropy models rather than optimizing them jointly.
The RD curves of the ablation studies are shown in Fig.
4. These curves illustrate that disabling motion estima-
tion, Gaussian compensation or joint optimization results
in reduced RD performance across various bitrates, under-
scoring the importance of these modules. Additionally, the
minus BD-PSNR values observed in the three experiments
compared to 4DGC, as shown in Tab. 5, further confirm the
effectiveness of our 4DGC in compressing dynamic scenes.
Fig. 6 illustrates a qualitative comparison of the com-
plete 4DGC at different bitrates against its variants. When
motion estimation is absent, details are added directly to
the initial frame without tracking object motion, resulting
in overlapping artifacts and increased temporal redundancy.
The variant without Gaussian compensation struggles to
capture newly appearing regions, such as fire eruptions.
Moreover, The variant lacking joint optimization disregards
the distribution characteristics of both the motion grid and
compensated Gaussians, limiting the encoder’s ability to
achieve low entropy. These findings highlight the effective-
ness of our motion estimation, Gaussian compensation, and
joint optimization in the 4DGC.
Furthermore, we analyze the average bit consumption for
keyframe and inter-frame under different λ1configurations,
as shown in Tab. 6. The significantly lower bit consump-
tion for inter-frames demonstrates the effectiveness of our
Figure 6. Qualitative results of 4DGC and its variants. Excluding
any module leads to lower reconstruction quality and increased
bitrate.
Table 5. The BD-PSNR results of the ablation studies when com-
pared with our full method on different datasets.
N3DV MeetRoom Google Immersive
w/o Motion Estimation -1.86 dB -1.13 dB -1.62 dB
w/o Gaussian Compensation -0.23 dB -0.22 dB -0.09 dB
w/o Joint Optimization -0.28 dB -0.34 dB -0.40 dB
Table 6. Analysis of average bit consumption for keyframe and
inter-frame with different λ1on the N3DV dataset.
λ1= 0.00001 λ1= 0.00005 λ1= 0.0001 λ1= 0.0003
keyframe (MB) 17.3 14.4 10.6 7.3
inter-frame (MB) 1.21 0.78 0.48 0.23
dynamic modeling in reducing inter-frame redundancy and
lowering inter-frame bitrates.
5. Discussion
Limitation. As a novel and efficient rate-aware com-
pression framework tailored for 4D Gaussian-based Free-
Viewpoint Video, our method has several limitations. First,
it relies on the reconstruction quality of the first frame,
where poor initialization may degrade overall performance.
Second, our method depends on multi-view video input and
struggles with sparse-view reconstruction.Finally, the de-
coding speed is slower compared to rendering, which could
be improved using advanced entropy decoding techniques.
Conclusion. We propose a novel rate-aware com-
pression framework tailored for 4D Gaussian-based Free-
Viewpoint Video. Leveraging a motion-aware 4D Gaussian
representation, 4DGC effectively captures inter-frame dy-
namics and spatial details while sequentially reducing tem-
poral redundancy. Our end-to-end compression scheme in-
corporates an implicit entropy model combined with rate-
distortion tradeoff parameters, enabling variable bitrates
while jointly optimizing both representation and entropy
model for enhanced performance. Experiments show that
4DGC not only achieves superior rate-distortion perfor-
mance but also adapts to variable bitrates, supporting photo-
realistic FVV applications with reduced storage and band-
width requirements in AR/VR contexts.
8

---

## Page 9

6. Acknowledgements
This work was supported by National Natural Sci-
ence Foundation of China (62271308), STCSM
(24ZR1432000, 24511106902, 24511106900,
22511105700, 22DZ2229005), 111 plan (BP0719010)
and State Key Laboratory of UHD Video and Audio
Production and Presentation.
References
[1] Eirikur Agustsson, Fabian Mentzer, Michael Tschannen,
Lukas Cavigelli, Radu Timofte, Luca Benini, and Luc V
Gool. Soft-to-hard vector quantization for end-to-end learn-
ing compressible representations. Advances in neural infor-
mation processing systems , 30, 2017. 3
[2] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael
Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil
Kim. HyperReel: High-fidelity 6-DoF video with ray-
conditioned sampling. In CVPR , 2023. 6, 7
[3] Johannes Ball ´e, Valero Laparra, and Eero P Simoncelli. End-
to-end optimization of nonlinear transform codes for per-
ceptual quality. In 2016 Picture Coding Symposium (PCS) ,
pages 1–5. IEEE, 2016. 3
[4] Johannes Ball ´e, David Minnen, Saurabh Singh, Sung Jin
Hwang, and Nick Johnston. Variational image compression
with a scale hyperprior. In ICLR , 2018. 3
[5] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.
Mip-nerf: A multiscale representation for anti-aliasing neu-
ral radiance fields. ICCV , 2021. 2
[6] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. CVPR , 2022.
[7] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-
based neural radiance fields. ICCV , 2023. 2
[8] Jean B ´egaint, Fabien Racap ´e, Simon Feltman, and Akshay
Pushparaja. Compressai: a pytorch library and evalua-
tion platform for end-to-end compression research. arXiv
preprint arXiv:2011.03029 , 2020. 5
[9] Jill M Boyce, Renaud Dor ´e, Adrian Dziembowski, Julien
Fleureau, Joel Jung, Bart Kroon, Basel Salahieh, Vinod Ku-
mar Malamal Vadakital, and Lu Yu. Mpeg immersive video
coding standard. Proceedings of the IEEE , 109(9):1521–
1536, 2021. 2
[10] Michael Broxton, John Flynn, Ryan Overbeck, Daniel Er-
ickson, Peter Hedman, Matthew DuVall, Jason Dourgarian,
Jay Busch, Matt Whalen, and Paul Debevec. Immersive light
field video with a layered mesh representation. 39(4):86:1–
86:15, 2020. 6, 7, 8
[11] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In CVPR , pages 130–141, 2023.
2
[12] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent
Sitzmann. pixelsplat: 3d gaussian splats from image pairs
for scalable generalizable 3d reconstruction. In arXiv , 2023.
2[13] Yu Chen and Gim Hee Lee. Dbarf: Deep bundle-adjusting
generalizable neural radiance fields. In CVPR , pages 24–34,
2023. 2
[14] Zhibo Chen, Tianyu He, Xin Jin, and Feng Wu. Learning
for video compression. IEEE Transactions on Circuits and
Systems for Video Technology , 30(2):566–576, 2020. 3
[15] Chenxi Lola Deng and Enzo Tartaglione. Compressing ex-
plicit voxel grid representations: fast nerfs become also
small. In Proceedings of the IEEE/CVF Winter Confer-
ence on Applications of Computer Vision , pages 1236–1245,
2023. 3
[16] Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B. Tenen-
baum, and Jiajun Wu. Neural radiance flow for 4d view
synthesis and video processing. In Proceedings of the
IEEE/CVF International Conference on Computer Vision ,
2021. 2
[17] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, De-
jia Xu, and Zhangyang Wang. Lightgaussian: Unbounded
3d gaussian compression with 15x reduction and 200+ fps,
2023. 3
[18] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance fields with time-aware neural voxels.
InSIGGRAPH Asia 2022 Conference Papers . ACM, 2022. 2
[19] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang,
Tao Liu, Zhilin Pei, Hengjie Li, Xingcheng Zhang, and Bo
Dai. Flashgs: Efficient 3d gaussian splatting for large-scale
and high-resolution rendering, 2024. 2
[20] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
CVPR , pages 12479–12488, 2023. 2, 6, 7
[21] Zhongpai Gao, Benjamin Planche, Meng Zheng, Anwesa
Choudhuri, Terrence Chen, and Ziyan Wu. 6dgs: Enhanced
direction-aware gaussian splatting for volumetric rendering,
2024. 2
[22] Danillo Graziosi, Ohji Nakagami, Satoru Kuma, Alexandre
Zaghetto, Teruhiko Suzuki, and Ali Tabatabai. An overview
of ongoing point cloud compression standardization activi-
ties: Video-based (v-pcc) and geometry-based (g-pcc). AP-
SIPA Transactions on Signal and Information Processing , 9:
e13, 2020. 2
[23] Zhiyang Guo, Wen gang Zhou, Li Li, Min Wang, and
Houqiang Li. Motion-aware 3d gaussian splatting for effi-
cient dynamic scene reconstruction. ArXiv , abs/2403.11447,
2024. 3
[24] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In SIGGRAPH 2024 Conference Papers .
Association for Computing Machinery, 2024. 2
[25] Lukas H ¨ollein, Alja ˇz Boˇziˇc, Michael Zollh ¨ofer, and Matthias
Nießner. 3dgs-lm: Faster gaussian-splatting optimization
with levenberg-marquardt, 2024. 2
[26] Mustafa Is ¸ık, Martin R ¨unz, Markos Georgopoulos, Taras
Khakhulin, Jonathan Starck, Lourdes Agapito, and Matthias
Nießner. Humanrf: High-fidelity neural radiance fields for
humans in motion. ACM Transactions on Graphics (TOG) ,
42(4), 2023. 2
9

---

## Page 10

[27] Yuheng Jiang, Zhehao Shen, Yu Hong, Chengcheng Guo,
Yize Wu, Yingliang Zhang, Jingyi Yu, and Lan Xu. Robust
dual gaussian splatting for immersive human-centric volu-
metric videos. arXiv preprint arXiv:2409.08353 , 2024. 2,
3
[28] Yuheng Jiang, Zhehao Shen, Penghao Wang, Zhuo Su, Yu
Hong, Yingliang Zhang, Jingyi Yu, and Lan Xu. Hifi4g:
High-fidelity human performance rendering via compact
gaussian splatting. In CVPR , pages 19734–19745, 2024. 2,
3
[29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics , 42
(4), 2023. 2
[30] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and
Ping Tan. Streaming radiance fields for 3d video synthe-
sis.Advances in Neural Information Processing Systems , 35:
13485–13498, 2022. 2, 6, 7, 8
[31] Mu Li, Wangmeng Zuo, Shuhang Gu, Debin Zhao, and
David Zhang. Learning convolutional networks for content-
weighted image compression. In CVPR , pages 3214–3223,
2018. 3
[32] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
CVPR , pages 5521–5531, 2022. 2, 6, 7, 8
[33] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang.
Neural scene flow fields for space-time view synthesis of dy-
namic scenes. In CVPR , pages 6494–6504, 2021. 2
[34] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker,
and Noah Snavely. Dynibar: Neural dynamic image-based
rendering. In CVPR , pages 4273–4284, 2023. 2
[35] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
InCVPR , pages 8508–8520, 2024. 2, 3, 4
[36] Kai Lin, Chuanmin Jia, Xinfeng Zhang, Shanshe Wang, Si-
wei Ma, and Wen Gao. Dmvc: Decomposed motion mod-
eling for learned video compression. IEEE Transactions
on Circuits and Systems for Video Technology , 33(7):3502–
3515, 2023. 3
[37] Artur Loza, Lyudmila Mihaylova, Nishan Canagarajah, and
David Bull. Structural similarity-based object tracking in
video sequences. In 2006 9th International Conference on
Information Fusion , pages 1–6, 2006. 5
[38] Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei
Cai, and Zhiyong Gao. Dvc: An end-to-end deep video com-
pression framework. In CVPR , pages 10998–11007, 2019. 3
[39] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured
3d gaussians for view-adaptive rendering. In CVPR , pages
20654–20664, 2024. 3
[40] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 3DV, 2024. 3, 6, 7
[41] Haichuan Ma, Dong Liu, Ning Yan, Houqiang Li, and Feng
Wu. End-to-end optimized versatile image compression withwavelet-like transform. IEEE Transactions on Pattern Anal-
ysis and Machine Intelligence , 44(3):1247–1263, 2022. 3
[42] Jue Mao and Lu Yu. Convolutional neural network based
bi-prediction utilizing spatial and temporal information in
video coding. IEEE Transactions on Circuits and Systems
for Video Technology , 30(7):1856–1870, 2020. 3
[43] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi,
Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duck-
worth. NeRF in the Wild: Neural Radiance Fields for Un-
constrained Photo Collections. In CVPR , 2021. 2
[44] Xiandong Meng, Chen Chen, Shuyuan Zhu, and Bing Zeng.
A new hevc in-loop filter based on multi-channel long-short-
term dependency residual networks. In 2018 Data Compres-
sion Conference , pages 187–196, 2018. 3
[45] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM , 65(1):99–106, 2021.
2
[46] David Minnen, Johannes Ball ´e, and George D Toderici.
Joint autoregressive and hierarchical priors for learned im-
age compression. Advances in neural information processing
systems , 31, 2018. 5
[47] Thomas M ¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. ACM Trans. Graph. , 41(4):102:1–
102:15, 2022. 2
[48] KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi
Koohpayegani, and Hamed Pirsiavash. Compgs: Smaller and
faster gaussian splatting with vector quantization. ECCV ,
2024. 3
[49] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien
Bouaziz, Dan B Goldman, Steven M. Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
InICCV (ICCV) , pages 5865–5874, 2021. 2
[50] Keunhong Park, Philipp Henzler, Ben Mildenhall, and Ri-
cardo Barron, Jonathan T.and Martin-Brualla. Camp: Cam-
era preconditioning for neural radiance fields. ACM Trans.
Graph. , 2023. 2
[51] Sungheon Park, Minjung Son, Seokhwan Jang, Young Chun
Ahn, Ji-Yeon Kim, and Nahyup Kang. Temporal interpo-
lation is all you need for dynamic neural radiance fields.
CVPR , pages 4212–4221, 2023. 2
[52] By S Pateux and J. Jung. An excel add-in for computing
bjontegaard metric and its evolution,” in vceg meeting. 2007.
6
[53] Sida Peng, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xi-
aowei Zhou. Representing volumetric videos as dynamic
mlp maps. In CVPR , pages 4252–4262, 2023. 3
[54] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer. D-NeRF: Neural Radiance Fields
for Dynamic Scenes. In CVPR , 2020. 2
[55] Saskia Rabich, Patrick Stotko, and Reinhard Klein. Fpo++:
Efficient encoding and rendering of dynamic neural radiance
fields by analyzing and enhancing fourier plenoctrees. arXiv
preprint arXiv:2310.20710 , 2023. 2
10

---

## Page 11

[56] Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srini-
vasan, Ben Mildenhall, Andreas Geiger, Jon Barron, and Pe-
ter Hedman. Merf: Memory-efficient radiance fields for real-
time view synthesis in unbounded scenes. ACM Transactions
on Graphics (TOG) , 42(4):1–12, 2023. 2
[57] Daniel Rho, Byeonghyeon Lee, Seungtae Nam, Joo Chan
Lee, Jong Hwan Ko, and Eunbyung Park. Masked wavelet
representation for compact neural radiance fields. In CVPR ,
pages 20680–20690, 2023. 3
[58] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu,
Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural
4d decomposition for high-fidelity dynamic reconstruction
and rendering. In CVPR , pages 16632–16642, 2023. 2
[59] Xihua Sheng, Jiahao Li, Bin Li, Li Li, Dong Liu, and Yan
Lu. Temporal context mining for learned video compression.
IEEE Transactions on Multimedia , 25:7311–7322, 2023. 3
[60] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele
Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerf-
player: A streamable dynamic scene representation with de-
composed neural radiance fields. IEEE Transactions on Visu-
alization and Computer Graphics , 29(5):2732–2742, 2023.
2, 3, 6, 7
[61] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei
Zhao, and Wei Xing. 3dgstream: On-the-fly training of
3d gaussians for efficient streaming of photo-realistic free-
viewpoint videos. In CVPR , pages 20675–20685, 2024. 1,
2, 3, 6, 7
[62] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei
Song, and Huaping Liu. Mixed neural voxels for fast multi-
view video synthesis. In ICCV , pages 19649–19659, 2023.
2, 6, 7
[63] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yan-
shun Zhang, Yingliang Zhang, Minve Wu, Jingyi Yu, and
Lan Xu. Fourier plenoctrees for dynamic radiance field ren-
dering in real-time. In CVPR , pages 13514–13524, 2022. 2
[64] Liao Wang, Qiang Hu, Qihan He, Ziyu Wang, Jingyi Yu,
Tinne Tuytelaars, Lan Xu, and Minye Wu. Neural resid-
ual radiance fields for streamably free-viewpoint videos. In
CVPR , pages 76–87, 2023. 1, 2, 3, 6, 7, 8
[65] Liao Wang, Kaixin Yao, Chengcheng Guo, Zhirui Zhang,
Qiang Hu, Jingyi Yu, Lan Xu, and Minye Wu. Videorf: Ren-
dering dynamic radiance fields as 2d feature video streams,
2023. 2, 3
[66] Yufei Wang, Zhihao Li, Lanqing Guo, Wenhan Yang, Alex C
Kot, and Bihan Wen. Contextgs: Compact 3d gaussian
splatting with anchor level context model. arXiv preprint
arXiv:2405.20721 , 2024. 3
[67] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing , 13(4):
600–612, 2004. 6
[68] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
InCVPR , pages 20310–20320, 2024. 2, 3, 4
[69] Minye Wu, Zehao Wang, Georgios Kouros, and Tinne Tuyte-
laars. Tetrirf: Temporal tri-plane radiance fields for efficientfree-viewpoint video. In CVPR , pages 6487–6496, 2024. 1,
2, 3, 6, 7, 8
[70] Jinbo Yan, Rui Peng, Luyang Tang, and Ronggang Wang. 4d
gaussian splatting with scale-aware residual field and adap-
tive optimization for real-time rendering of temporally com-
plex dynamic scenes. In ACM MM , pages 7871–7880, 2024.
3
[71] Ning Yan, Dong Liu, Houqiang Li, Bin Li, Li Li, and Feng
Wu. Invertibility-driven interpolation filter for video coding.
IEEE Transactions on Image Processing , 28(10):4912–4925,
2019. 3
[72] R. Yang, M. Xu, Z. Wang, and T. Li. Multi-frame quality
enhancement for compressed video. In CVPR , pages 6664–
6673, Los Alamitos, CA, USA, 2018. 3
[73] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for
high-fidelity monocular dynamic scene reconstruction. arXiv
preprint arXiv:2309.13101 , 2023. 3
[74] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-
time photorealistic dynamic scene representation and render-
ing with 4d gaussian splatting. 2024. 3
[75] Zhenghui Zhao, Shiqi Wang, Shanshe Wang, Xinfeng
Zhang, Siwei Ma, and Jiansheng Yang. Enhanced bi-
prediction with convolutional neural network for high-
efficiency video coding. IEEE Transactions on Circuits and
Systems for Video Technology , 29(11):3291–3301, 2019. 3
[76] Zihan Zheng, Houqiang Zhong, Qiang Hu, Xiaoyun Zhang,
Li Song, Ya Zhang, and Yanfeng Wang. Hpc: Hierarchical
progressive coding framework for volumetric video. In ACM
MM, page 7937–7946, New York, NY , USA, 2024. Associa-
tion for Computing Machinery. 2, 3
[77] Zihan Zheng, Houqiang Zhong, Qiang Hu, Xiaoyun Zhang,
Li Song, Ya Zhang, and Yanfeng Wang. Jointrf: End-to-
end joint optimization for dynamic neural radiance field rep-
resentation and compression. In 2024 IEEE International
Conference on Image Processing (ICIP) , pages 3292–3298,
2024. 2, 3
11

---

## Page 12

4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable
Free-Viewpoint Video
Supplementary Material
7. More Implementation Details
7.1. Entropy Model
Here, we provide a detailed introduction to our implicit en-
tropy model. This model estimates entropy by learning a
cumulative distribution function (CDF) that represents the
probability distribution of the input data. The model com-
prises multiple layers, each parameterized by weight matri-
ces, biases, and scaling factors. These parameters transform
the input tensor through a series of operations, such as ma-
trix multiplication and bias addition, combined with non-
linear activations like softplus. Each layer progressively re-
fines the input values to approximate a CDF, capturing the
cumulative probability distribution of the data.
To maintain precision, we avoid assuming any prede-
fined distribution for the data. Instead, we construct a novel
distribution within the entropy model to closely approxi-
mate the actual data distribution. Specifically, the entropy
model computes cumulative logits for values just below and
above the actual data point. This approach enables the
model to capture the probability interval that contains the
data point, thereby improving estimation accuracy. The
likelihood, which represents the probability of the input
within this interval, is calculated as the difference between
the sigmoid activations of these cumulative logits. This
likelihood is then incorporated into the overall loss function
during training.
When the training process is complete, we apply quanti-
zation and a range coder for entropy coding to further com-
press the data volume and generate the bitstream. The en-
tropy model itself occupies about 100 KB per frame, which
is relatively large compared to the compressed motion grid
Mtand the compensated Gaussians ∆Gt. Therefore, we
analyze the actual distribution of the data prior ωtbefore
entropy coding and transmit ωtinstead of the entire en-
tropy model. This approach further reduces the size of each
frame.
The process of entropy encoding can be represented as
follows:
Q(x) =⌊q·x+ 0.5⌋,
Bt=E(Q(x)−Q(min( x)) ;ωt).(13)
Here, xrepresents the data to be compressed, which in-
cludes Mtand∆Gt. The bitstream after entropy encoding,
denoted as Bt, consists of BM
tandB∆G
t, corresponding
toMtand∆Gt, respectively. The range encoder is repre-
sented by E.Table 7. The average size of each component of inter-frames on
the N3DV dataset.
BM
t B∆G
t ωM
t ω∆G
t Total
Size (KB) 164.72 65.88 0.25 0.17 231.02
To enable compression into the int8 format, we convert
the compressed data into non-negative values and subse-
quently restore it to its original range during decompression.
The variable qdenotes the quantization parameter. During
quantization, the data is multiplied by q, which effectively
expands its range and subtly enhances the precision of the
quantization process. On the decoding side, the entropy de-
coding process can be expressed as follows:
ˆx=D(Bt;ωt) +Q(min( x))
q, (14)
where Dis the range decoder. Therefore, for each inter-
frame, the data to be transmitted includes the bitstream BM
t
andB∆G
t, and the data distributions ωt={ωM
t, ω∆G
t}.
The size of each of these components is detailed in Tab. 7.
7.2. Hyperparameters Settings
In this section, we provide a more detailed explanation of
the hyperparameter settings for the two aspects.
Model Parameter Settings : We use two shared global
lightweight MLPs ΦµandΦR, both with an input dimen-
sion of 20, a hidden layer size of 64, and output dimensions
of 3 and 4, respectively. For the multi-resolution motion
gridMt, we use feature grid channels of 4, 4, and 2, with
dimensions 323,643, and 1283.
Gaussian Compensation Parameter Settings : For
Gaussian compensation, we choose τg= 0.0001 ,τµ=
0.08, and τR=π
4. We only apply the second type of Gaus-
sian compensation to the Gaussians whose scale s>−0.01.
(The actual scale is activated using the exponential func-
tion.) To prevent an excessive increase in the number of
Gaussians, we filter out Gaussians with opacity α < 0.01
after stage 2.
8. More Results
8.1. Quantitative Results
We provide a quantitative comparison of image quality,
measured by PSNR, and model size across all scenes in the
N3DV dataset in Tab. 8. To further demonstrate the vari-
able bitrate characteristic and superior RD performance of
1

---

## Page 13

GT ReRF 3DGStream Ours TeTriRF0.78MB 0.82MB 8.07MB 0.51MB
0.84MB 0.69MB 8.05MB 0.44MB
0.81MB 0.47MB 0.85MB 8.19MB
0.91MB 0.44MB 0.87MB 8.19MBFigure 7. More Qualitative comparison on more sequences of the N3DV dataset against ReRF, TeTriRF, and 3DGStream.
(a) Cook Spinach (b) Cut Beef (c) Flame Steak (d) Sear SteakOurs Full ReRF TeTriRF 3DGStream
Bitrate (MB/Frame) Bitrate (MB/Frame) Bitrate (MB/Frame) Bitrate (MB/Frame)PSNR
Figure 8. More Rate-distortion curves across different sequences of N3DV datasets.
our method, we present additional RD curves for more se-
quences in Fig. 8.8.2. Qualitative Results
We have prepared more qualitative comparison results in the
Fig. 7.
2

---

## Page 14

Table 8. Quantitative comparison of average PSNR values(dB) and model size(MB) across all sequences in the N3DV dataset.
MethodCoffee
MartiniCook
SpinachCut
BeefFlame
SalmonFlame
SteakSear
SteakMean
StreamRF 27.77/9.34 31.54/7.48 31.74/7.17 28.19/7.93 32.18/7.02 32.29/6.88 30.61/7.64
ReRF 26.24/0.79 31.23/0.84 31.82/0.81 26.80/0.78 32.08/0.91 30.03/0.51 29.71/0.77
TeTriRF 27.10/0.73 31.97/0.69 32.45/0.85 27.61/0.82 32.74/0.87 32.03/0.60 30.65/0.76
3DGStream 27.96/8.00 32.88 /8.05 32.99/8.19 28.52 /8.07 33.41/8.19 33.58/8.16 31.54/8.11
Ours 27.98 /0.58 32.81/ 0.44 33.03 /0.47 28.49/ 0.51 33.58 /0.44 33.60 /0.50 31.58 /0.49
3