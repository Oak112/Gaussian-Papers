

---

## Page 1

arXiv:2505.16533v1  [cs.CV]  22 May 2025Motion Matters: Compact Gaussian Streaming for
Free-Viewpoint Video Reconstruction
Jiacong Chen1,2Qingyu Mao3Youneng Bao4Xiandong Meng5Fanyang Meng5
Ronggang Wang5,6Yongsheng Liang1,2B
1College of Applied Technology, Shenzhen University, Shenzhen, China
2College of Big Data and Internet, Shenzhen Technology University, Shenzhen, China
3College of Electronics and Information Engineering, Shenzhen University, Shenzhen, China
4Department of Computer Science, City University of Hong Kong, Hong Kong, China
5Pengcheng Laboratory, Shenzhen, China
6School of Electronic and Computer Engineering, Peking University, Shenzhen, China
(c) Comual uir ed Storage
30.430.630.83131.231.431.631.83232.2
0.04 0.4 40PSNR (dB)
 
Storage per frame (MB)
NeRFPlayer
K-Plane
HyperReel
4DGS
StreamRF
3DGStream
QUEEN-s
QUEEN-l  
ComGS-sComGS-s (Ours)
4DGS
K-Plane
Better
3DGStream
NeRFPlayerStreamRFHyperReelQUEEN-sQUEEN-l
Offline
Online
3DGStream
PSNR: 33.01dB
Storage: 7.8MB/frameOurs
PSNR: 33.61dB
Storage:  0.049MB/frame
159× smaller,  
improved 
fidelity
4 PerformanceComGS-l (Ours)
ComGS-l
Figure 1: Left: Experimental results on N3DV dataset [ 14] showcase the effectiveness of our method,
which reduces the storage requirement of 3DGStream [ 13] by 159 ×, with enhanced visual quality.
Right : Comparison with existing methods in storage and reconstruction fidelity. Hollow circles
denote offline methods, while solid circles represent online methods.
Abstract
3D Gaussian Splatting (3DGS) has emerged as a high-fidelity and efficient
paradigm for online free-viewpoint video (FVV) reconstruction, offering viewers
rapid responsiveness and immersive experiences. However, existing online meth-
ods face challenge in prohibitive storage requirements primarily due to point-wise
modeling that fails to exploit the motion properties. To address this limitation, we
propose a novel Compact Gaussian Streaming (ComGS) framework, leveraging the
locality and consistency of motion in dynamic scene, that models object-consistent
Gaussian point motion through keypoint-driven motion representation. By transmit-
ting only the keypoint attributes, this framework provides a more storage-efficient
solution. Specifically, we first identify a sparse set of motion-sensitive keypoints
localized within motion regions using a viewspace gradient difference strategy.
Equipped with these keypoints, we propose an adaptive motion-driven mechanism
that predicts a spatial influence field for propagating keypoint motion to neighbor-
ing Gaussian points with similar motion. Moreover, ComGS adopts an error-aware
correction strategy for key frame reconstruction that selectively refines erroneous
regions and mitigates error accumulation without unnecessary overhead. Over-
all, ComGS achieves a remarkable storage reduction of over 159 ×compared to
3DGStream and 14 ×compared to the SOTA method QUEEN, while maintaining
competitive visual fidelity and rendering speed. Our code will be released.
Preprint. Under review.

---

## Page 2

1 Introduction
Reconstructing free-viewpoint video (FVV) from multi-view videos captured by cameras with known
poses has attracted growing interest in the field of computer vision and graphics. FVV exhibits great
potential as a next-generation visual medium that enables immersive and interactive experiences,
with broad application in virtual and augmented reality (VR/AR) applications [26].
Recently, 3D Gaussian Splatting (3DGS) has become a promising method for FVV reconstruction,
due to its significant advancements in real-time rendering and high-fidelity view synthesis. These
approaches typically fall into two categories: 1) incorporating temporal function into Gaussian
primitives and optimizing directly [ 19,34,12], and 2) applying a deformation field to capture
the spatio-temporal transformations of canonical Gaussians [ 18,11,37–39]. While these FVV
reconstructions accurately represent dynamic scenes, they are trained in an offline manner and require
transmitting the full set of reconstructed parameters prior to rendering.
In contrast, by enabling per-frame training and progressive transmission, online FVV reconstruction
allows immediate playback without the overhead of full-scene preloading. As a pioneer work,
3DGStream [ 13] extends 3DGS to online FVV reconstruction using InstantNGP [ 4] to model
the geometric transformation frame-by-frame. While achieving impressive rendering speed, its
structural constraint hinders the volumetric representation performance and degrades the visual
quality. Building on this paradigm, subsequent works [ 32,33] enhance model expressiveness through
explicitly optimizing Gaussian attribute residuals, achieving competitive synthesis quality and higher
robustness. However, the storage demands of these methods remain prohibitively high for real-time
transmission, with reconstructed data typically exceeding 20 MB per second.
In this paper, we aim to design a storage-efficient solution for FVV streaming that minimizes
bandwidth requirements and enables real-time transmission. In online FVV reconstruction, since
dynamic scenes contain a large proportion of static regions, the key to efficient reconstruction lies
in motion modeling. Our first insight, therefore, is to only model the Gaussian attribute residuals in
the motion regions, which eliminates the unnecessary updates in static regions. Building on motion
modeling, we note that scene motion tends to be consistent, where Gaussian points associated with
the same object typically exhibit the same or similar motion in dynamic scene representation. Our
second insight, based on this observation, is to use a shared motion representation to model the
attribute residuals with similar motion. This contrasts with existing online methods [ 13,32] that
utilize point-wise strategy to update the attribute residuals in motion regions, and the result is motion
redundancy elimination and more compact storage. Lastly, we exploit a key frame fine-tune strategy
to handle the error accumulation brought by non-rigid motion and novel objects emergence.
Specifically, to accomplish this, we propose a Compact Gaussian Streaming (ComGS) framework
that leverages a set of keypoints ( = 200 ), significantly fewer than the full set of Gaussian points
(≈200K), to holistically model motion regions at each timestep. ComGS begins with a motion-
sensitive keypoints selection through a viewspace gradient difference strategy. This ensures that
the selected keypoints are accurately positioned within motion regions and prevents redundant or
incorrect modeling of static areas. Subsequently, we design an adaptive motion-driven mechanism
that defines a keypoint-specific spatial influence field, with which neighboring Gaussian points can
share the motion of the keypoint. Unlike conventional k-nearest neighbor (KNN) methods [ 43,44],
the spatial influence field can accommodate the complexity and variability of motion structure in
dynamic scenes, so that keypoints can more accurately drive the motion of the surrounding region.
Finally, to mitigate error accumulation in a compact and effective manner, we propose an error-aware
correction strategy for key frame reconstruction that selectively updates only those Gaussians with
reconstruction errors.
Our major contributions can be summarized as follow:
•We introduce a motion-sensitive keypoint selection to accurately identify keypoints within
motion regions, and an adaptive motion-driven mechanism that effectively propagates motion
to neighboring points. These leverage the locality and consistency of motion and achieve a
more storage-efficient solution for online FVV reconstruction.
•We propose an error-aware correction strategy for key frame reconstruction that mitigates
error accumulation over time by selectively updating Gaussian points with reconstruction
errors, which ensures long-term consistency and minimizes redundant correction.
2

---

## Page 3

•Experiments on two benchmark datasets show that the effectiveness of our method and
its individual components. Our method achieves a compression ratio of 159×over the
3DGStream and 14×over state-of-the-art model QUEEN, enabling real-time transmission
while preserving competitive reconstruction quality and rendering speed.
2 Related work
2.1 Dynamic Gaussian Splatting
Recently, 3D Gaussian Splatting (3DGS) [ 9,10,20–23] has attracted great attention in Free-viewpoint
Video (FVV) reconstruction for its high photorealistic performance and real-time rendering speed.
Several works [ 19,34,12] expand temporal variation as a function and optimize directly for modeling
Gaussian attributes across frames. For instance, 4D Gaussian Splatting [ 19] incorporates time-
conditioned 3D Gaussians and auxiliary components into 4D Gaussians, while ST-GS [ 12] models the
transformation of structural attributes and opacity as a temporal function to represent scene motions.
These time variant-based methods achieve superior rendering efficiency, but suffer from prohibitive
storage requirements. Other works [ 11,39–42] employ vanilla 3D Gaussians as a canonical space
and a deformation field to represent the dynamic scene. In this category, 4D-GS [ 11] utilizes
hexplanes [ 15], six orthogonal planes, as latent embeddings and deliver them into a small MLP to
deform temporal transformation of Gaussian points, achieving efficient computational complexity
and lightweight storage requirement. Building upon this, GD-GS [ 39] further improves scene
modeling accuracy by incorporating geometric priors, which provides a more structured and precise
representation of dynamic scene. Among them, both SC-GS [ 43] and SP-GS [ 44] adopt sparse control
points to control scene motion using a k-nearest neighbor (KNN) [ 45] strategy for motion modeling.
While these methods achieve notable improvements in computational efficiency and rendering speed,
they are designed for offline FVV reconstruction and do not support frame-by-frame delivering.
Additionally, motion-insensitive control point selection and scale-agnostic KNN motion modeling
lead to redundant representation of static regions and reduced deformation accuracy in dynamic
scenes. Our online method addresses these limitations by selecting keypoints from motion regions
at each timestep and modeling motion with awareness of local motion scales, which enables more
accurate and efficient modeling of online FVV .
2.2 Online Free-Viewpoint Video Reconstruction
Compared to the offline methods, online reconstruction enables FVV to be incrementally trained
and transmitted in a per-frame manner, which allows users to preview or interact immediately with
the video content. Leveraging the high-fidelity view synthesis capabilities of Neural Radiance Field
(NeRF) [ 1–8], a set of studies have explored NeRF-based methods [ 25,46,47,36,29] for online
FVV reconstruction, such as StreamRF [ 25], VideoRF [ 46] and TeTriRF [ 36]. Despite advanced
visual quality, NeRF-based methods are hindered by their limited rendering speeding of implicit
structure, which limits their practical applications.
With the utilization of 3DGS [ 9], 3DGStream [ 13] introduces a hash-based MLP to encode the
position and rotation transformation of Gaussian points at each frame, and designs an adaptive
Gaussians addition strategy for novel objects across frames. Based on this paradigm, QUEEN [ 32]
proposes a Gaussian residual-based framework for model expressiveness enhancement and a learned
quantization-sparsity framework for residuals compression. HiCoM [ 33] designs a hierarchical
coherent motion mechanism to effectively capture and represent scene motion for fast and accurate
training. To deploy into mobile device, V3[28] presents a novel approach that compresses Gaussian
attributes as a 2D video to facilitate hardware video codecs. IGS [ 48] proposes a generalized anchor-
driven Gaussian motion network that learns residuals with a singe step, achieving a significant
improvement of training speed. Nevertheless, these methods face challenge in real-time transmission,
due to their substantial storage requirements. This overhead mainly stems from redundant updates of
both static Gaussian points across frames, as well as repeated modeling of Gaussian points with similar
motion. Our study exploits the locality and consistency of motion by leveraging motion-sensitive
keypoints to adaptively drive motion regions, and this avoids redundant storage and transmission.
3

---

## Page 4

Key frame
(a) Group of FramesNon key frame
Non key frame
Key frameViewspace 
GradientTop k 
valuesDynamic scores
KeypointsFrame t-1
Frame t
t-1 GaussiansFrame t-1
Frame t
t-1 Gaussians
(b)Motion-Sensitive Keypoint Selection
 (c)Adaptive Motion-Driven MechanismKeypoint
Neighboring
point
kpq
kp
npq
npDriven
wkp→np
Influence 
field
wkp→np1.0
adap
Scene
(d)Error-Aware Correctiont-1 Gaussians
t t t t t SH sR  , ,,,  
t t t t t SH sR  , ,,,   t Gaussianserror Gaussiansnon-error
Gaussians
1→tw
0→tw wt
aware gatewt
Error--
aware gateNon-key frame reconstructio n
Key frame reconstruction
Spatial
DifferenceFigure 2: The overall pipeline of ComGS framework. (a) The reconstruction process starts from
the first frame initialized using vanilla 3DGS [ 9]. Subsequent frames are organized into groups
of frames (GoFs). For non-key frames, (b) we begins with a motion-sensitive keypoint selection
using a viewspace gradient difference strategy, (c) and utilizes an adaptive motion-driven mechanism
to control neighboring points motion. For key frames, (d) an error-aware correction strategy is
introduced to mitigate the error accumulation across frames.
3 Methods
Our goal is to reconstruct and transmit FVV in a storage-efficient and streaming manner. To achieve
it, we propose a Compact Gaussian Streaming (ComGS) framework for online FVV reconstruction,
as illustrated in Fig. 2. First, ComGS begins with a motion-sensitivity keypoint selection using a
viewspace gradient difference, ensuring subsequent motion control learning (Sec. 3.2). Second,
we develop an adaptive motion-driven mechanism that applies a spatial influence field to control
neighboring point motion (Sec. 3.3). Third, we devise an error-aware correction strategy for key
frame reconstruction to mitigate error accumulation brought by non-rigid motion and novel objects
emergence in online reconstruction (Sec. 3.4). Finally, we introduce our compression techniques and
optimization process in Sec. 3.5.
3.1 Preliminary
3DGS [ 9] models a 3D scene as a large amount of anisotropic 3D Gaussian points in world space as
an explicit representation. The central position and geometric shape of each Gaussian point in world
space are defined by a mean vector µand covariance matrix Σ, mathematically represented as:
G(x) =exp(−1
2(x−µ)TΣ−1(x−µ)) (1)
For differentiable optimization, the covariance matrix Σis decoupled into a scaling matrix Sand a
rotation matrix R. Each Gaussian point is characterized by its color ciand opacity α.
During rendering, the Gaussian points are initially projected into viewing plane, and the color of each
pixel Ccan be obtained by α-blending:
C=X
i∈nciαii−1Y
j=1(1−αj) (2)
where nis the number of Gaussian points contributing to this pixel.
3.2 Motion-Sensitive Keypoint Selection
Establishing an effective keypoint-driven motion representation necessitates to select appropriate
keypoints. Considering motion locality, keypoints should be located in motion regions, which avoids
redundant modeling in static areas and enables accurate modeling of complex motions
4

---

## Page 5

Thus, inspired by [ 32], we propose a motion-sensitive keypoint selection based on viewspace gradient
difference (Fig. 2 (b)). The core idea is to identify the dynamic Gaussian by the gradient change of
rendering loss in inter-frames, and based on the gradient values, the kGaussian points with the largest
gradient are selected as keypoints. Specifically, following the gradient computation in 3DGS [ 9],
we compute gradients using the previous Gaussian positions pt−1, the rendered images Rt−1, the
reconstruction loss Lrecon , and the ground-truth images GTt−1andGTt:
Gt−1=∂Lt−1
recon
∂pt−1,Lt−1
recon =Lrecon(Rt−1, GTt−1) (3)
Gt=∂Lt
recon
∂pt−1,Lt
recon =Lrecon(Rt−1, GTt) (4)
Dynamic significance scores ∆Gt∈RN(Nis the number of Gaussians) were calculated by means
of absolute values of gradient differences:
∆Gt=1
VVX
v=1|G(v)
t− G(v)
t−1| (5)
where Vis the number of the training viewpoints. Finally, we select the top khigh dynamic
significance scores from all Gaussian points as keypoints Ktat timestamp t. Selecting the top- k
Gaussian points with the highest dynamic scores not only identifies those located in motion regions,
but also naturally allocates more keypoints to the areas with complex motion, facilitating more
accurate modeling of such regions.
In this paper, for a balance of training efficiency and reconstructed quality, we set k= 200 .
3.3 Adaptive Motion-Driven Mechanism
Equipped with the selected keypoints Ktat current timestep, the next step is to determine which
neighboring points are controlled by these keypoints, and apply their transformations to drive the
motion of the controlled neighboring points. Previous works [ 43,44] employ k-nearest neighbor
(KNN) [ 45] search to predict the motion of each Gaussian points, showing advanced results in
monocular synthetic video reconstruction, but they do not fully consider unnecessary modeling in
static region and motion scale difference, which leads to computational redundancy and inaccurate
representation.
In contrast, we propose an adaptive motion-driven mechanism that enables each keypoint to drive
neighboring points through a spatial influence field, as illustrated Fig. 2 (c). Specifically, motivated
by [9], for each keypoints Ki
tatttimestep, we initialize a quaternion qi
adap∈R4and a scaling vector
si
adap∈R3to compute the spatial influence field Σi
adap∈R3×3. For a neighboring Gaussian point
Gjwith position µj, its distance to keypoint Ki
tis given by µij=µj−µKi
t. The influence weight is
then computed as:
wij= exp
−1
2µ⊤
ij(Σi
adap)−1µij
(6)
Ifwijexceeds a predefined threshold τadap, the Gaussian Gjis considered to be controlled by
keypoint Ki
t.
Ifwij≥τadap, G j∈ Ci
t (7)
where Ci
tdenotes the set of Gaussians controlled by keypoint Ki
t.
To model motion, each keypoint Ki
tis further assigned a learnable translation offset ∆µi
K∈R3and a
rotation represented by a quaternion ∆qi
K∈R4. For a Gaussian Gjcontrolled by multiple keypoints
{Ki
t}i∈Ij, its overall motion is computed by aggregating the motions of its associated keypoints,
weighted by their influence scores wij:
∆µt
j=X
i∈Ijwij·∆µi
K,∆qt
j=X
i∈Ijwij·∆qi
K (8)
where ∆µt
jand∆qt
jindicate the motion of Gaussian jatttimestep.
5

---

## Page 6

By leveraging a compact set of keypoints with spatial influence fields, our method enables accurate
and efficient control of Gaussian motions at each frame. Since Gaussians share motion attributes
through keypoints, only 14 parameters per keypoint are required, significantly reducing storage
demands and mitigating data redundancy.
3.4 Error-Aware Corrector
By using keypoints to drive scene motion, we model the transformation of Gaussian points from the
previous frame to the current frame with an extremely compact parameters. Nevertheless, keypoints-
based motion controlling only supports to represent rigid motion effectively and faces challenge to
handle non-rigid motion and novel objects emergence, which results in error accumulation across
frames.
A straightforward solution to mitigate error accumulation and ensure accurate long-term FVV
representations is to separate the video into frame groups and update the attributes of all Gaussians at
key frames. However, this strategy would lead to a substantial of unnecessary parameters updating,
since most of parameters are already correctly representing the scene and do not require modification.
To mitigate error accumulation in a compact and efficient manner, we propose an error-aware
corrector strategy that only finetunes the Gaussians with detected errors, significantly decreasing
storage demands and promoting more accurate scene reconstruction, as illustrated in Fig. 2 (d).
Specifically, given a video sequence, we select a key frame every sframes, forming the key frame
sequence {fs, f2s, . . . , f ns}, as shown in Fig. 2 (a). The remaining frames are reconstructed by
keypoints driven. During key frame reconstruction, given the attributes of a Gaussian point at previous
timestep θt−1
i: (µt−1
i, qt−1
i, st−1
i, σt−1
i, sht−1
i), we introduce a set of learnable parameters ∆θt
ito
model the attribute residuals. To identify which Gaussian points require correction, we predict a
learnable mask mifor each point. A sigmoid function is used to map mito the range (0,1), which
refers as a soft mask:
msoft
i=Sigmoid (mi), msoft
i∈(0,1) (9)
Similar to [ 21,22], the soft mask is binarized into a hard mask using a predefined threshold ϕthres ,
where the non-differentiable binarization is handled with the straight-through estimator (STE) to
enable gradient flow, represented as:
mhard
i=sg( 1(msoft
i> ϕthres)−msoft
i) +msoft
i, mhard
i∈ {0,1} (10)
where 1is the indicator function and sgindicates the stop gradient operation. Then, the mhard
i is
applied to the attribute residuals before rendering, followed as:
θt
i=θt−1
i⊕mhard
i∗∆θt
i (11)
where⊕denotes the attribute-specific update operation. Meanwhile, we define a optimized function
to regulate the perceptual error while encouraging sparse residual updates:
Lerror =1
NX
imsoft
i (12)
After optimization for the current key frame, only the attribute residuals ˆ∆θt={∆θt|Mhard= 1}
and the hard mask set Mhardneed to be stored and transmitted, minimizing the required data
redundancy and transmission overhead.
3.5 Optimization and Compression
For the first frame optimization, we employ COLMAP [ 27] to generate the initial point cloud and
follow the pipeline of 3DGStream [ 13]. The optimization for both the first frame and non-key frames
is supervised by the reconstruction loss Lrecon , which is composed by an L1-norm loss L1and a
D-SSIM loss LD−SSIM [24]:
Lrecon = (1−λD−SSIM )L1+λD−SSIMLD−SSIM (13)
For key frame optimization, we minimize a combined loss consisting of Lrecon andLerror :
Ltotal=Lrecon +λerrorLerror (14)
6

---

## Page 7

Table 1: Quantitative comparisons on Neural 3D Video (N3DV) [14] and MeetRoom datasets [25].
Dataset Category Method PSNR (dB) ↑Storage (MB) ↓Training (sec) ↓Rendering (FPS) ↑
N3DVOfflineNeRFPlayer [29] 30.69 17.10 72 0.05
HyperReel [30] 31.10 1.20 104 2.00
4DGS [11] 31.15 0.13 8 34
SpaceTime [12] 32.05 0.67 20 140
OnlineStreamRF [25] 30.68 31.4 15 8.3
TeTriRF [36] 30.43 0.06 39 4
3DGStream [13] 31.67 7.80 8.5 261
QUEEN-s [32] 31.89 0.68 4.65 345
QUEEN-l [32] 32.19 0.75 7.9 248
ComGS-s (ours) 31.87 0.049 37 91
ComGS-l (ours) 32.12 0.106 43 147
MeetRoomStaticI-NGP [35] 28.10 48.2 66 4.1
3DG-S [9] 31.31 21.1 156 571
OnlineStreamRF [25] 26.72 9.0 10.2 10
3DGStream [13] 30.79 4.1 4.9 350
ComGS-s (ours) 31.49 0.028 28.3 98
StreamRF 3DGStream Ours GT
3DGStream Ours GT
Figure 3: Quantitative comparison. We visualize our method and other online FVV methods on
N3DV [14] and MeetRoom [25] dataset.
where λerror controls the degree of error awareness, thereby balancing reconstruction quality and
memory efficiency. We set λD−SSIM = 0.2andλerror = 0.001in this paper.
After optimization, the initialized Gaussians θ0and the residuals ˆ∆θtfor key frame error correction
are further compressed through quantization and entropy coding, enabling compact storage without
performance degradation. More details are provided in the Appendix .
Table 2: Ablation study on proposed components. Flame Steak and Flame Salmon are from the
N3DV dataset.
Experiments Selection Adaptive CorrectionFlame Steak Flame Salmon
PSNR (dB) ↑Storage (KB) ↓PSNR (dB) ↑Storage (KB) ↓
1 × ✓ ✓ 33.27 46.7 29.22 56.7
2 ✓ × ✓ 32.82 36.4 28.96 45.7
3 × × ✓ 31.26 37.9 27.75 46.4
4 ✓ ✓ × 31.67 26.9 28.74 26.9
5 ✓ ✓ ✓ 33.49 46.5 29.32 53.4
4 Experiments
4.1 Experimental Setup
We evaluate our method on two widely-used public benchmark datasets. (1) Neural 3D Video
(N3DV) [14] consists of six indoor video sequences captured by 18 to 21 viewpoints. (2) Meet
7

---

## Page 8

Figure 4: Visualization of our keypoint-driven motion representation. Top: selected keypoints are
concentrated in motion regions. Bottom : adaptive control of neighboring points also focuses on
motion-intensive areas, enabling accurate and efficient motion modeling.
(a) Farthest keypoint selection (b) Random keypoint selection (c) Ours
Figure 5: Visualization of different selection methods and corresponding updated regions.
Room [25] comprises four indoor scenes recorded with a 13 cameras multi-view system. In both
of two datasets, we employ the first view for testing. Our method is implemented on an NVIDIA
A100 GPU. We train 150 epochs for non-key frames reconstruction and 1000 epochs for key frames
fine-tuning. We measure the visual quality of rendered images by average PSNR, required storage,
rendering FPS and training time. More implement details are provided in the Appendix .
4.2 Quantitative Comparisons
We conduct quantitative comparisons on existing online methods including StreamRF [ 25],
TeTriRF [ 36], 3DGStream [ 13] and QUEEN [ 32], as well as the SOTA offline FVV ap-
proaches [ 12,11,29,30] on N3DV and Meetroom (Tab. 1). Our method is evaluated in two variants:
ComGS-s (small) and ComGS-l (large), using key frame intervals of s=10 ands=2, respectively.
Tab. 1 shows that our ComGS achieves competitive results among existing online FVV methods on
N3DV dataset. Notably, ComGS-s achieves a substantial reduction in storage by 159×compared
to 3DGStream and 14×compared to QUEEN. This advantage enables real-time transmission in
limited bandwidth and enhances the overall user viewing experience. On MeetRoom dataset, our
method outperforms 3DGStream, obtaining +0.7dB PSNR and 146 ×smaller size. Our advantages
are mainly due to two factor: 1) using keypoint as a shared representation requires transmitting
only a small number of keypoint attributes; and 2) the error-aware correction module effectively
rectifies regions with scene inaccuracies using minimal additional parameters. In the Appendix , the
quantitative results are provided for each scene to offer a more detailed comparison.
4.3 Qualitative Comparisons
As shown in Fig. 3, we compare our reconstructed results to other online FVV methods on N3DV
and MeetRoom. ComGS effectively reconstructs both motion and static regions and provides more
closer results to the ground truth. Fig. 3 shows that 3DGStream introduces noticeable artifacts due to
its global update of Gaussian points across the entire scene, which often leads to incorrect updates
in static regions. In contrast, our method restricts modeling to motion regions and applies targeted
corrections in error-prone areas, resulting in more accurate and robust scene reconstruction. More
qualitative results are offered in Appendix .
4.4 Ablation Study
To validate the effectiveness of our proposed methods, we ablate three components of ComGS
framework in Tab. 2. In the Experiment 1, we adopt a random chosen techniques for keypoint
8

---

## Page 9

/uni00000013 /uni00000018/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000015/uni00000018/uni00000013 /uni00000016/uni00000013/uni00000013/uni00000033/uni00000036/uni00000031/uni00000035
/uni00000015/uni0000001b/uni00000015/uni0000001c/uni00000016/uni00000013/uni00000016/uni00000014/uni00000016/uni00000015/uni00000016/uni00000016/uni00000016/uni00000017
/uni00000014/uni00000018/uni00000013w/o correction  
Ours-s
Ours-l
(b) (a) (c) (d)Frame IndexFigure 6: (a) PSNR comparison over time. Visualizations of (b) w/o key frames correction. (c)
ComGS-s. (d) ComGS-l.
selection. However, since the selection is not guided by region regions, it may result in ineffective
modeling of static areas and inadequate representation of scenes, which leads to a slight degradation
in PSNR. Experiment 2 model scene motion only using the selected keypoints, without applying
the adaptive motion-driven mechanism for neighboring points. The resulting drop in reconstruction
quality demonstrates the importance of collaborative modeling in keypoint-driven regions for accurate
motion representation. In the Experiment 3, we reconstruct the FVV only relying on the key frames
correction, which indicates that keypoint-driven motion representation is crucial for our method.
Experiment 4 disable the error-aware correction in key frames reconstruction. This leads to a
significant performance drop, demonstrating that error-aware correction in key frames would solve
the error accumulation across frames.
Table 3: Ablation study on comparing control
strategies for neighboring points.
Control tech PSNR (dB) Storage (KB)
KNN 31.39 44.1
Adaptive 31.87 49.0Table 4: Ablation study of the error-aware correc-
tion strategy.
Configuration PSNR (dB) Storage (KB)
w/o error-aware 31.65 373
with error-aware 31.87 49.0
To further investigate the role of keypoint-driven motion representation, we visualize the selection
and driven process in Fig. 4. The top row shows that keypoints are predominantly selected in motion
regions, such as the human body and moving objects. The bottom row highlights the adaptively
controlled areas for neighboring points, which similarly focus on regions with significant motion
(e.g., the person and the dog). Fig. 5 visualizes Gaussians updated region using farthest keypoint
selection [ 44], random keypoint selection and our method, respectively, which demonstrates that our
method accurately captures motion-intensive areas. These results indicate that ComGS can effectively
leverage the locality and consistency of scene motion.
We also evaluate a KNN-based method [ 45] for selecting neighboring points around keypoints
(Tab. 3). This approach shows inferior performance, as it does not distinguish between static and
motion regions, leading to redundant modeling and poor adaptation to varying motion scales.
Fig. 6 evaluates the effect of key frame correction. The visual results in Fig. 6 (b–d) further highlight
that key frame correction significantly reduces artifacts in motion regions such as flames, helping to
maintain finer temporal consistency throughout the sequence. Tab. 4 shows that correction without
error-aware leads to significantly higher storage due to redundant Gaussians updating. Moreover,
without focusing on high-error regions, updates may affect error-free areas and result in suboptimal
performance. Therefore, enabling error-awareness improves both accuracy and efficiency.
5 Conclusion
In this paper, we proposed ComGS, a storage-efficient framework for online FVV real-time transmis-
sion. We utilized a keypoint-driven motion representation to models scene motion by leveraging the
locality and consistency of motion. This approach significantly reduces storage requirements through
motion-sensitive keypoint selection and an adaptive motion driven mechanism. To address error
accumulation over time, we further introduce an error-aware correction strategy that mitagates these
9

---

## Page 10

error in an efficient manner. Experiments demonstrate the surpassing storage efficiency, competitive
visual fidelity and rendering speed of our method. In future work, we aim to design a practical
solution on novel applications, such as 3D video conference and volumetric live streaming, providing
viewers with immersive and interactive experiences.
References
[1]B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf:
Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM ,
vol. 65, no. 1, pp. 99–106, 2021.
[2]A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance fields,” in European
conference on computer vision . Springer, 2022, pp. 333–350.
[3]S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, “Plenoxels: Radiance
fields without neural networks,” in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , 2022, pp. 5501–5510.
[4]T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a
multiresolution hash encoding,” ACM transactions on graphics (TOG) , vol. 41, no. 4, pp. 1–15, 2022.
[5]A. Chen, Z. Xu, X. Wei, S. Tang, H. Su, and A. Geiger, “Dictionary fields: Learning a neural
basis decomposition,” ACM Transactions on Graphics (TOG) , vol. 42, no. 4, pp. 1–12, 2023.
[6]C. Sun, M. Sun, and H.-T. Chen, “Direct voxel grid optimization: Super-fast convergence for
radiance fields reconstruction,” in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , 2022, pp. 5459–5469.
[7]J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan,
“Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields,” in Proceedings of the
IEEE/CVF international conference on computer vision , 2021, pp. 5855–5864.
[8]D. Verbin, P. Hedman, B. Mildenhall, T. Zickler, J. T. Barron, and P. P. Srinivasan, “Ref-nerf:
Structured view-dependent appearance for neural radiance fields,” in 2022 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) . IEEE, 2022, pp. 5481–5490.
[9]B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3d gaussian splatting for real-time
radiance field rendering.” ACM Trans. Graph. , vol. 42, no. 4, pp. 139–1, 2023.
[10] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian: Unbounded 3d gaussian
compression with 15x reduction and 200+ fps,” arXiv preprint arXiv:2311.17245 , 2023.
[11] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian
splatting for real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2024, pp. 20 310–20 320.
[12] Z. Li, Z. Chen, Z. Li, and Y . Xu, “Spacetime gaussian feature splatting for real-time dynamic
view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , 2024, pp. 8508–8520.
[13] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-the-fly training of
3d gaussians for efficient streaming of photo-realistic free-viewpoint videos,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 20 675–20 685.
[14] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove,
M. Goesele, R. Newcombe et al. , “Neural 3d video synthesis from multi-view video,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp. 5521–5531.
[15] A. Cao and J. Johnson, “Hexplane: A fast representation for dynamic scenes,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 130–141.
[16] Z. Li, S. Niklaus, N. Snavely, and O. Wang, “Neural scene flow fields for space-time view
synthesis of dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , 2021, pp. 6498–6508.
10

---

## Page 11

[17] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, “K-planes: Explicit
radiance fields in space, time, and appearance,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2023, pp. 12 479–12 488.
[18] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y . Zhang, and X. Jin, “Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 2024, pp. 20 331–20 341.
[19] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic dynamic scene representation
and rendering with 4d gaussian splatting,” arXiv preprint arXiv:2310.10642 , 2023.
[20] K. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash, “Compact3d: Compressing
gaussian splat radiance field models with vector quantization,” arXiv preprint arXiv:2311.18159 ,
2023.
[21] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian representation
for radiance field,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , 2024, pp. 21 719–21 728.
[22] H. Wang, H. Zhu, T. He, R. Feng, J. Deng, J. Bian, and Z. Chen, “End-to-end rate-distortion
optimized 3d gaussian representation,” in European Conference on Computer Vision . Springer,
2025, pp. 76–92.
[23] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis, “Reducing the memory
footprint of 3d gaussian splatting,” Proceedings of the ACM on Computer Graphics and Interactive
Techniques , vol. 7, no. 1, pp. 1–17, 2024.
[24] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from
error visibility to structural similarity,” IEEE transactions on image processing , vol. 13, no. 4, pp.
600–612, 2004.
[25] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, “Streaming radiance fields for 3d video synthesis,”
Advances in Neural Information Processing Systems , vol. 35, pp. 13 485–13 498, 2022.
[26] Y . Chen, Q. Wang, H. Chen, X. Song, H. Tang, and M. Tian, “An overview of augmented reality
technology,” in Journal of Physics: Conference Series , vol. 1237, no. 2. IOP Publishing, 2019, p.
022082.
[27] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proceedings of the
IEEE conference on computer vision and pattern recognition , 2016, pp. 4104–4113.
[28] P. Wang, Z. Zhang, L. Wang, K. Yao, S. Xie, J. Yu, M. Wu, and L. Xu, “Vˆ 3: Viewing
volumetric videos on mobiles via streamable 2d dynamic gaussians,” ACM Transactions on Graphics
(TOG) , vol. 43, no. 6, pp. 1–13, 2024.
[29] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y . Xu, and A. Geiger, “Nerfplayer: A
streamable dynamic scene representation with decomposed neural radiance fields,” IEEE Transactions
on Visualization and Computer Graphics , vol. 29, no. 5, pp. 2732–2742, 2023.
[30] B. Attal, J.-B. Huang, C. Richardt, M. Zollhoefer, J. Kopf, M. O’Toole, and C. Kim, “Hyper-
reel: High-fidelity 6-dof video with ray-conditioned sampling,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , 2023, pp. 16 610–16 620.
[31] D. A. Huffman, “A method for the construction of minimum-redundancy codes,” Proceedings
of the IRE , vol. 40, no. 9, pp. 1098–1101, 1952.
[32] S. Girish, T. Li, A. Mazumdar, A. Shrivastava, S. De Mello et al. , “Queen: Quantized efficient
encoding of dynamic gaussians for streaming free-viewpoint videos,” Advances in Neural Information
Processing Systems , vol. 37, pp. 43 435–43 467, 2025.
[33] Q. Gao, J. Meng, C. Wen, J. Chen, and J. Zhang, “Hicom: Hierarchical coherent motion for
dynamic streamable scenes with 3d gaussian splatting,” in The Thirty-eighth Annual Conference on
Neural Information Processing Systems .
11

---

## Page 12

[34] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d gaussians: Tracking by
persistent dynamic view synthesis,” in 2024 International Conference on 3D Vision (3DV) . IEEE,
2024, pp. 800–809.
[35] Y . Jiang, K. Yao, Z. Su, Z. Shen, H. Luo, and L. Xu, “Instant-nvr: Instant neural volumetric ren-
dering for human-object interactions from monocular rgbd stream,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , 2023, pp. 595–605.
[36] M. Wu, Z. Wang, G. Kouros, and T. Tuytelaars, “Tetrirf: Temporal tri-plane radiance fields for
efficient free-viewpoint video,” in Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition , 2024, pp. 6487–6496.
[37] Z. Lu, X. Guo, L. Hui, T. Chen, M. Yang, X. Tang, F. Zhu, and Y . Dai, “3d geometry-
aware deformable gaussian splatting for dynamic view synthesis,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , 2024, pp. 8900–8910.
[38] M. Liu, Q. Yang, H. Huang, W. Huang, Z. Yuan, Z. Li, and Y . Xu, “Light4gs: Lightweight
compact 4d gaussian splatting generation via context model,” arXiv preprint arXiv:2503.13948 , 2025.
[39] J. Bae, S. Kim, Y . Yun, H. Lee, G. Bang, and Y . Uh, “Per-gaussian embedding-based deformation
for deformable 3d gaussian splatting,” in European Conference on Computer Vision . Springer, 2024,
pp. 321–335.
[40] W. O. Cho, I. Cho, S. Kim, J. Bae, Y . Uh, and S. J. Kim, “4d scaffold gaussian splatting for
memory efficient dynamic scene reconstruction,” arXiv preprint arXiv:2411.17044 , 2024.
[41] D. Sun, H. Guan, K. Zhang, X. Xie, and S. K. Zhou, “Sdd-4dgs: Static-dynamic aware
decoupling in gaussian splatting for 4d scene reconstruction,” arXiv preprint arXiv:2503.09332 ,
2025.
[42] J. Yan, R. Peng, L. Tang, and R. Wang, “4d gaussian splatting with scale-aware residual field and
adaptive optimization for real-time rendering of temporally complex dynamic scenes,” in Proceedings
of the 32nd ACM International Conference on Multimedia , 2024, pp. 7871–7880.
[43] Y .-H. Huang, Y .-T. Sun, Z. Yang, X. Lyu, Y .-P. Cao, and X. Qi, “Sc-gs: Sparse-controlled
gaussian splatting for editable dynamic scenes,” in Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition , 2024, pp. 4220–4230.
[44] D. Wan, R. Lu, and G. Zeng, “Superpoint gaussian splatting for real-time high-fidelity dynamic
scene reconstruction,” in Proceedings of the 41st International Conference on Machine Learning ,
2024, pp. 49 957–49 972.
[45] L. E. Peterson, “K-nearest neighbor,” Scholarpedia , vol. 4, no. 2, p. 1883, 2009.
[46] L. Wang, K. Yao, C. Guo, Z. Zhang, Q. Hu, J. Yu, L. Xu, and M. Wu, “Videorf: Rendering
dynamic radiance fields as 2d feature video streams,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , 2024, pp. 470–481.
[47] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and M. Wu, “Neural residual
radiance fields for streamably free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , 2023, pp. 76–87.
[48] J. Yan, R. Peng, Z. Wang, L. Tang, J. Yang, J. Liang, J. Wu, and R. Wang, “Instant gaussian
stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting,”
arXiv preprint arXiv:2503.16979 , 2025.
12

---

## Page 13

Appendix
We provide more material to supplement our main paper. This appendix first introduces more
implementation details in Sec. A. Then, we provide additional experimental results in Sec. B,
discussion on limitations and future works in Sec. C and broader impact in Sec. D.
A More Implementation details
Training: Our code is based on the open-source code of 3DGStream [ 13]. On both N3DV and
MeetRoom dataset, we utilize COLMAP [ 27] to generate the initial point cloud and vanilla 3DGS [ 9]
to initialize the Gaussians for 3000 epochs at first frame. Subsequently, our ComGS reconstructs the
non key frames for 150 epochs and key frames for 1000 epochs. For the balance of visual quality and
storage requirements, we set spherical harmonics (SH) degree to 1. During training, the learning rate
for Gaussian attributes is set to 0.002, for the attributes of the adaptive influence region to 0.02, and
for the learnable mask mito 0.01. Other learning rates follow the setting of 3DGStream [13].
Compression: For the reconstruction process, the uncompressed Gaussian attributes and their
residuals have substantial memory requirements. We employ quantization and entropy coding to
further compress them. Specifically, for the first frame reconstruction, we apply 16-bit quantization
to the position attributes due to their higher sensitivity, while the other attributes are quantized to 8
bits. For the correction in key frame reconstruction, we quantize all attribute residuals using 8 bits.
Notably, the attributes of a keypoint play a crucial role in guiding the motion of nearby non-keypoints.
As a result, even minor quantization errors in keypoints may be amplified throughout the scene. To
preserve modeling accuracy, we thus refrain from quantizing keypoint attributes. Finally, we deliver
these quantized values to entropy coding [31].
Datasets: (1) Neural 3D Video (N3DV) dataset [14] comprises of six indoor scenes captured by
a multi-view system of 18 to 21 cameras at a resolution of 2704 ×2028 and 30 FPS. Following the
previous works [ 13,14,11], we downsample the videos by a factor of 2 for training and testing and
employ the central view for testing view. (2) MeetRoom dataset [25] is captured by a 13-camera
multi-view system, including four dynamic scenes at 1280 ×720 resolution and 30 FPS. The center
reference camera is also used for testing. We perform distortion for this dataset following the settings
of the 3DGS [9] to improve the reconstruction quality.
Table 5: Ablation study on Number of keypoints.
#Keypoints 50 100 200 300 400 500
PSNR (dB ↑) 31.77 31.85 31.87 31.84 31.86 31.80
Storage (KB ↓) 44.4 46.2 50.1 50.2 54.4 57.3
Table 6: Ablation study on Group of Frames.
#Frames 2 5 10 15 20
PSNR (dB ↑) 32.12 32.01 31.87 31.78 31.66
Storage (KB ↓) 108.3 66.6 50.1 43.2 40.0
B Additional Experimental Results
B.1 More Ablation Study
In this section, we further investigate the hyperparameters and analyze the impact of the proposed
components on N3DV [14] dataset, to achieve a balance between performance and efficiency.
Effect of the keypoint numbers : To investigate the impact of the number of keypoints on recon-
struction quality and compression efficiency, we conduct an ablation study by varying the number of
keypoints from 50 to 500. As shown in Tab. 5, the reconstruction performance peaks when using
200 keypoints. This observation aligns with the nature of dynamic scenes, where motion typically
13

---

## Page 14

Table 7: Effect of λerror on reconstructed quality and storage.
λerror 0 0.0001 0.001 0.01
PSNR (dB ↑) 31.91 31.91 31.87 31.79
Storage (KB ↓) 183.0 96.3 50.1 29.2
(a) Error -aware correction (b) Error -aware region (c) Rendered image (d) GT
Figure 7: (a) Visualization of error-aware Gaussians. (b) Visualization of error regions between key
frame and previous frame. (c)(d) Comparison on rendered images and original images.
occurs in a limited spatial region. Using 200 keypoints is sufficient to capture these areas for effective
reconstruction. Increasing the number of keypoints beyond this leads to redundant or incorrect
representation in static regions. Therefore, using 200 keypoints strikes a good balance between
performance and storage, and is adopted as the default configuration in our method.
Effect of group of frames : We evaluate how the size of the Group of Frames (GoF) affects
reconstruction quality and storage, as shown in Tab. 6. These results indicate that shorter GoFs
can better handle non-rigid motions and novel objects, which are difficult to be reconstructed by
keypoint-driven motion. Larger GoFs exploit temporal redundancy for better compression, but may
accumulate errors in the presence of motion and scene changes. In our setting, we use GoF = 2
as our large model for high-fidelity reconstruction, and GoF = 10 as our small model for compact
representation.
Effect of error-aware correction : We explore the effect of the parameter λerroron reconstruction
quality and storage, as shown in Tab. 7. While a larger λerrorimproves compression by focusing only
on perceptually salient errors, it may overlook subtle regions, which leads to degraded reconstruction.
In contrast, smaller values retain more points, which helps suppress error accumulation across frames,
albeit with higher storage costs.
Fig. 7 (a) visualizes the error-aware Gaussian points identified by error-aware correction, while (b)
shows a heatmap of differences between the key frame and the previous frame, which highlights
the error regions. We observe that the error-aware points in (a) align well with the high-error
regions in (b), which indicates that our method effectively captures areas likely to suffer from error
accumulation. Fig. 7 (c) and (d) compare our rendered images with the ground truth. The results
show that our method significantly reduces artifacts in dynamic regions, confirming the effectiveness
of our error-aware correction.
B.2 More Results
To offer a more comprehensive comparison, the per-scene quantitative results are presented on
N3DV [ 14] and MeetRoom [ 25] in Tab. 8 and Tab. 9, respectively. Moreover, we also provide the
experimental results of existing offline and online methods in Tab. 8 as a reference. Further qualitative
results with StreamRF [25] and 3DGStream [13] are indicated in Fig. 8 and Fig. 9.
14

---

## Page 15

Table 8: Per-scene quantitative results on the N3DV dataset . Offline and online methods are
separated for clarity.
Method Coffee Martini Cook Spinach Cut Beef
PSNR Storage PSNR Storage PSNR Storage
(dB↑) (MB ↓) (dB ↑) (MB ↓) (dB ↑) (MB ↓)
KPlanes [17] 29.99 1.0 32.60 1.0 31.82 1.0
NeRFPlayer [29] 31.53 18.4 30.56 18.4 29.35 18.4
HyperReel [30] 28.37 1.2 32.30 1.2 32.92 1.2
4DGS [19] 28.33 29.0 32.93 29.0 33.85 29.0
4D-GS [11] 27.34 0.3 32.46 0.3 32.49 0.3
Spacetime-GS [12] 28.61 0.7 33.18 0.7 33.52 0.7
E-D3DGS [39] 29.33 0.5 33.19 0.5 33.25 0.5
StreamRF [25] 27.84 31.84 31.59 31.84 31.81 31.84
3DGStream [13] 27.75 7.80 33.31 7.80 33.21 7.80
QUEEN-l [32] 28.38 1.17 33.40 0.59 34.01 0.57
ComGS-s (ours) 28.63 0.058 32.94 0.047 33.30 0.051
ComGS-l (ours) 28.76 0.154 33.26 0.094 33.53 0.104
Flame Salmon Flame Steak Sear Steak
PSNR Storage PSNR Storage PSNR Storage
(dB↑) (MB ↓) (dB ↑) (MB ↓) (dB ↑) (MB ↓)
KPlanes [17] 30.44 1.0 32.38 1.0 32.52 1.0
NeRFPlayer [29] 31.65 18.4 31.93 18.4 29.12 18.4
HyperReel [30] 28.26 1.2 32.20 1.2 32.57 1.2
4DGS [19] 29.38 29.0 34.03 29.0 33.51 29.0
4D-GS [11] 29.20 0.3 32.51 0.3 32.49 0.3
Spacetime-GS [12] 29.48 0.7 33.40 0.7 33.46 0.7
E-D3DGS [39] 29.72 0.5 33.55 0.5 33.55 0.5
StreamRF [25] 28.26 31.84 32.24 31.84 32.36 31.84
3DGStream [13] 28.42 7.80 34.30 7.80 33.01 7.80
QUEEN-l [32] 29.25 1.00 34.17 0.59 33.93 0.56
ComGS-s (ours) 29.31 0.052 33.42 0.045 33.59 0.040
ComGS-l (ours) 29.58 0.129 33.84 0.083 33.74 0.0704
Table 9: Per-scene quantitative results on the MeetRoom dataset .
Method Discussion Stepin Trimming VrHeadset
PSNR (dB ↑) 31.72 30.17 32.12 31.95
Storage (KB ↓) 37.5 24.2 27.0 24.5
C Limitations and Future Works
Despite achieving significant improvements, there are several limitations that constrains the perfor-
mance of our ComGS, which would be addressed in future works. First, as the first frame serves as
the foundation for subsequent frame updates, poor initialization would lead to error propagation and
degraded performance. Developing a robust and efficient initialized strategy for first frame could
further improve the visual quality and storage efficiency of online FVV . Second, while this paper
mainly focuses on designing a compact representation for online FVV , it does not fully consider the
training time of the encoding stage, leaving room for further improvements in training efficiency.
Additionally, our method relies on the dense view videos as inputs, which is expensive for practical
applications. Future work will explore extending the framework to sparse-view or monocular inputs
for real-world scenarios.
D Broader Impact
Our work is a positive technology. This method reconstructs free-viewpoint videos from multi-view
2D videos in a streaming manner, which can improve the immersive and interactive experience of
15

---

## Page 16

viewers. As discussed in the introduction, this technology has potential to benefit various aspects of
daily life, including applications in remote diagnosis and 3D video conferencing.
(a) Flame Steak
(b) Cook Spinach
(c) Cut Roasted Beef
(d) Flame Salmon3DGStream Ours GT StreamRF
Figure 8: Comparison on N3DV [14] dataset.
16

---

## Page 17

(a) Discussion
(b) Stepin
(c) Trimming3DGStream Ours GTFigure 9: Comparison on MeetRoom [25] dataset.
17