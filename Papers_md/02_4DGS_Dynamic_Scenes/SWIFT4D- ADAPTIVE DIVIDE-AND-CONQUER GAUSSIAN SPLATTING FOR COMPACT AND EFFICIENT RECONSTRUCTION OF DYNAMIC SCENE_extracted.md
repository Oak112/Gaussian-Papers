

---

## Page 1

Published as a conference paper at ICLR 2025
SWIFT 4D: A DAPTIVE DIVIDE -AND -CONQUER
GAUSSIAN SPLATTING FOR COMPACT AND EFFICIENT
RECONSTRUCTION OF DYNAMIC SCENE
Jiahao Wu, Rui Peng, Zhiyan Wang, Lu Xiao,
Luyang Tang ,Jinbo Yan ,Kaiqiang Xiong ,Ronggang Wang∗
Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology
Shenzhen Graduate School, Peking University
2301212750@stu.pku.edu.cn ,rgwang@pkusz.edu.cn
32.23
32.05 32.01
31.67
31.1531.79
31.6331.7
30.6830.6930.8
30.630.83131.231.431.631.83232.232.4
0 2 4 6 8 10 12 14PSNR(dB)
Time(hours)FPS:125
Swift4D (Ours) 
Ours(lite)
FPS:215
3DGstreamFPS:0.15
K-planes
FPS:30
4DGS
FPS:5
Mix-voxel
FPS:8
StreamRFFPS:0.05
NeRFPlayerFPS:114
Real-time 4dgsFPS:110
SpaceTimeGS
FPS:0.12
Hex-plane
The size of circle 
represents the size 
of model
Figure 1: Our method demonstrates high-quality rendering, rapid convergence, and compact storage
characteristics. It can achieve competitive result with just 5 minutes of training. Additionally, with
increased training iterations, our method excels in handling finer details.
ABSTRACT
Novel view synthesis has long been a practical but challenging task, although
the introduction of numerous methods to solve this problem, even combining
advanced representations like 3D Gaussian Splatting, they still struggle to recover
high-quality results and often consume too much storage memory and training
time. In this paper we propose Swift4D, a divide-and-conquer 3D Gaussian
Splatting method that can handle static and dynamic primitives separately,
achieving a good trade-off between rendering quality and efficiency, motivated by
the fact that most of the scene is the static primitive and does not require additional
dynamic properties. Concretely, we focus on modeling dynamic transformations
only for the dynamic primitives which benefits both efficiency and quality. We
first employ a learnable decomposition strategy to separate the primitives, which
relies on an additional parameter to classify primitives as static or dynamic. For
the dynamic primitives, we employ a compact multi-resolution 4D Hash mapper
to transform these primitives from canonical space into deformation space at each
timestamp, and then mix the static and dynamic primitives to produce the final
output. This divide-and-conquer method facilitates efficient training and reduces
storage redundancy. Our method not only achieves state-of-the-art rendering
quality while being 20× faster in training than previous SOTA methods with a
minimum storage requirement of only 30MB on real-world datasets. Code is
available at https://github.com/WuJH2001/swift4d .
1arXiv:2503.12307v1  [cs.CV]  16 Mar 2025

---

## Page 2

Published as a conference paper at ICLR 2025
(a) Dynamic NeRF method (b) Dynamic 3DGS method (c) Our decomposition method Deformed point
Canonical point
Static point
Dynamic point
Deformation path
Figure 2: Illustration of different dynamic scene rendering methods. (a) Pumarola et al. (2021);
Park et al. (2021) proposes mapping deformation field points to canonical space, a widely adopted
practice in NeRF-based methods; (b) Wu et al. (2024); Yang et al. (2024) propose mapping canonical
space points to the deformation field; (c) We propose dividing the points in canonical space into
dynamic and static, and then mapping only the dynamic points to the deformation space.
1 I NTRODUCTION
Novel view synthesis (NVS) is a crucial task in computer vision and graphics, with significant
applications in areas such as augmented reality (AR), virtual reality (VR), and content production.
The goal of NVS is to render photorealistic images from arbitrary viewpoints using 2D images
or video inputs. While recent advancements have achieved considerable success in static scenes,
this task becomes particularly challenging when applied to dynamic scenes, where complexities
introduced by object motion and temporal changes make accurate rendering significantly difficult.
Current NVS techniques can be broadly classified into two predominant approaches: neural
rendering methods, exemplified by Neural Radiance Fields (NeRF) Mildenhall et al. (2021), and
point cloud-based rendering techniques, such as 3D Gaussian Splatting Kerbl et al. (2023). NeRFs
have recently made significant strides in achieving photorealistic rendering of static scenes, with
subsequent works Barron et al. (2021; 2022; 2023); Reiser et al. (2023) further enhancing both
quality and speed. Despite these advancements in static scene rendering, NeRFs face significant
challenges when extended to dynamic scenes, primarily due to the substantial training time and
storage requirements. To overcome these obstacles, various approaches have been proposed. As
shown in Fig.2(a), Pumarola et al. (2021) and Park et al. (2021) leverage deformation fields to map
deformation space at arbitrary timestamps to canonical space, effectively capturing dynamic scene
changes. Li et al. (2021) and Gao et al. (2021) employ scene flow to model the motion trajectories
within dynamic environments. Cao & Johnson (2023) and Fridovich-Keil et al. (2023), decompose
the 4D spacetime domain into multiple compact planes, thereby improving training and rendering
speeds. Although these methods have achieved some degree of success, achieving high-quality
real-time rendering remains challenging.
Compared to NeRF, 3DGS offers significant advantages, including real-time rendering and
substantially reduced training time. Within the scope of dynamic modeling, several notable methods
have emerged. As shown in Fig.2(b), 4DGS Wu et al. (2024), inspired by HexPlane, introduces a
neural voxel encoder to model deformation relationships over time. 3DGStream Sun et al. (2024)
utilizes a compact Neural Transformation Cache (NTC) to efficiently model the translation and
rotation of 3D Gaussians between two adjacent frames. RTGS Yang et al. (2023) treats spacetime as
an integrated whole by optimizing a set of 4D primitives, parameterized as anisotropic ellipses that
capture both geometry and appearance. Additionally, STGS Li et al. (2024) enhances standard 3D
Gaussians with temporal opacity and motion/rotation parameters, effectively capturing both static
and dynamic elements to model dynamic deformation.
While these approaches achieve higher-quality results with faster rendering times, they still face
challenges related to long training time and heavy storage requirements. One potential limitation
in their approach is the uniform treatment of all Gaussian points during the modeling process.
However, we observe that static points, such as those in background regions, constitute the majority
of the scene. These points exhibit minimal or no deformation and therefore do not require complex
dynamic modeling. It is more efficient to partition the scene into static and dynamic points and
model each separately. This strategy has the potential to significantly reduce computational overhead
and storage requirements. Moreover, as demonstrated by Wang et al. (2023), applying the same
modeling technique to both dynamic and static points can cause blurring in dynamic regions due to
the influence of static areas, ultimately compromising rendering quality.
∗Corresponding author
2

---

## Page 3

Published as a conference paper at ICLR 2025
In this paper, we introduce Swift4D, a method that simultaneously achieves fast convergence,
compact storage, and real-time high-quality rendering. Our approach starts by decomposing
Gaussian points into dynamic and static groups based on 2D multi-view images, incorporating an
additional parameter dfor differentiation. For temporal modeling, we employ a deformation field
approach using a compact multi-resolution 4DHash and MLPs as the deformer, which maps dynamic
Gaussian points from canonical space to deformation space at arbitrary timestamps. Notably, as
shown in Fig. 2(c), temporal modeling is applied exclusively to the dynamic points, while static
points are treated as temporally invariant, significantly reducing computational demands. This
reduction in the number of dynamic points enables the 4DHash to concentrate on deformation,
leading to faster convergence and improved rendering quality. Finally, we combine the static and
dynamic Gaussian points to render the final output. This approach also addresses the issue of
blurring caused by static elements interfering with the time-aware multi-resolution 4DHash.
Our method achieves SOTA performance in terms of training and rendering speed, storage efficiency,
and rendering quality. Furthermore, our supplement videos ( basketball 1 and 2) demonstrate that
our approach remains effective even in scenarios involving large movements. We will release our
code and pre-trained models upon acceptance. In summary, the key contributions of our work are:
1) We propose a novel method for decomposing dynamic 3D scenes into dynamic and static
components based on 2D images, effectively reducing computational complexity. This
method can be seamlessly integrated into existing dynamic approaches as a plug-and-play
module to enhance quality.
2) We introduce a compact multi-resolution 4DHash, with a footprint as small as 8MB, to
effectively model the spatio-temporal domain. This approach not only enhances rendering
quality and accelerates training but also ensures efficient and compact storage.
3) Our method achieves state-of-the-art performance in training and rendering speed, storage,
and high-quality output.
2 R ELATED WORK
Novel View Synthesis. In recent years, novel view synthesis has garnered significant attention,
leading to numerous breakthroughs. NeRFMildenhall et al. (2021) pioneered this domain
by leveraging multi-layer perceptrons (MLPs) combined with volume rendering to model 3D
radiance fields, enabling image rendering from arbitrary viewpoints. Subsequent works aimed to
enhance efficiency and quality. Methods such as TensorF Chen et al. (2022), DVGO Sun et al.
(2022), PlenoxelFridovich-Keil et al. (2022), and Plenoctree Yu et al. (2021) adopt grid-based
representations for faster training and rendering. Instant NGP M ¨uller et al. (2022) further accelerates
this process with a hash encoder, significantly reducing computation time. Meanwhile, MipNeRF
Barron et al. (2021) and MipNeRF360 Barron et al. (2022) propose integrated positional encoding
(IPE) to model conical frustums, effectively mitigating aliasing issues. More recently, 3DGS Kerbl
et al. (2023) introduced a novel point-based rendering paradigm for novel view synthesis, achieving
real-time rendering with high quality. This has spurred additional advancements, including
Mipsplatting Yu et al. (2024) for anti-aliasing, 2DGS Huang et al. (2024a) for improved mesh
extraction, and ScaffoldGS Lu et al. (2024) for large-scale scene rendering.
Novel View Synthesis for dynamic scene. Li et al. (2021); Lin et al. (2024); Kratimenos et al.
(2023) attempt to directly model the trajectories of moving points across the scene, but they continue
to encounter challenges related to storage. Pumarola et al. (2021); Park et al. (2021); Wu et al.
(2024); Yang et al. (2024) try to build a consistent canonical space across each time step and
then employ a deformer, mainly MLP-based and Muti-plane-based, to map this canonical space
to deformation spaces at each timestamp. Huang et al. (2024b) focuses on monocular dynamic
inputs, leveraging sparse control points to reconstruct scene dynamics with exceptionally high FPS.
Lin et al. (2024) employs Fourier series and polynomial fitting to model the motion of Gaussian
points, enabling dynamic reconstruction. K-planes Fridovich-Keil et al. (2023) and Hexplane Cao &
Johnson (2023) employ an explicit structural representation of the 6D light field rather than modeling
underlying motions. Representing the deformer using MLPs or low-rank planes can reduce storage
requirements, but it often results in slower training and limited capacity for capturing complex
deformations.
3

---

## Page 4

Published as a conference paper at ICLR 2025
Static GS
𝑮(𝝁,𝒔,𝒓,𝝈,𝒄,𝒅) Mixed GS𝑡
Rendered IMG𝑑>𝜁?Density controlDecomp. Init.
4D hash ℎ4𝑑 MLPs
Spatial -Temporal Structure   Sec.3.3Deform
d𝑟
d𝜇...
First Frame Images
Sparse points Init.
2D Dynamic -static masksDecomp.Dynamic GS
Train view 
Test viewFloaterDensity control
Sec.3.4 Sec.3.2Canonical. points
Figure 3: Pipeline of our Swift4D. First, we use the first frame images to obtain a well-initialized
canonical point cloud. Then, we train the dynamic parameter daccording to the method described
in Sec.3.2. Based on d, the point cloud is divided into dynamic and static categories. Dynamic
points undergo deformation using a spatio-temporal structure, as discussed in Sec.3.3. Finally, the
deformed dynamic points are mixed with static points for rendering.
Recently, He et al. (2024); Yan et al. attempt to separate dynamic and static Gaussian points to
improve rendering quality and introduced external models to segment foreground and background
areas. While these efforts have explored this direction, the resulting output quality remains
suboptimal. Liang et al. (2023) employs adaptive dynamic-static separation, which differs from
our explicit separation approach.
3 M ETHOD
Our main approach aims to achieve faster training speeds and higher quality rendering results
through the decomposition of dynamic and static elements. Based on this insight, we designed our
pipeline, as illustrated in Fig. 3. In this section, we will provide a detailed analysis of each module
in the pipeline. The preliminary concepts of 3D Gaussian Splatting are briefly introduced in Sec.
3.1. We initially train the canonical space Gaussians using the first-frame images and then optimize
the dynamic parameter dof each Gaussian point based on the 2D dynamic-static pixel masks from
different viewpoints, as discussed in Sec.3.2. In the following stage, as outlined in Sec. 3.3, we
freeze the training of dynamic parameter dand proceed to jointly optimize the remaining parameter
of the Gaussian points alongside the swift spatio-temporal structure. Furthermore, our pruning
strategies are thoroughly described in Sec. 3.4, while Sec. 3.5 provides an in-depth discussion
of the optimization process.
3.1 G AUSSIAN SPLATTING PRELIMINARY
3DGSKerbl et al. (2023) uses 3D Gaussian points as its rendering primitives. These 3D Gaussian
points have the following parameter: mean µ, covariance matrix Σ, opacity σ, and view-dependent
color c. A 3D Gaussian point is mathematically defined as:
G(x) =e−1
2(x−µ)TΣ−1(x−µ)(1)
In the next rendering phase, the 3D mean µis directly projected onto the plane as a 2D mean µ2D,
while the 3D covariance matrix is transformed into a 2D covariance matrix using the following
formula: Σ′= (JWΣWTJT), where WandJdenote the viewing transformation and the Jacobian
of the affine approximation of the perspective projection transformation, respectively. Finally, the
color of each pixel can be calculated using the following formula:
C(x) =X
i∈N(x)ciαi(x)i−1Y
j=1(1−αj(x))where αi(x) =σiexp
−1
2(x−µ2D
i)TΣ′−1(x−µ2D
i)
.
(2)
4

---

## Page 5

Published as a conference paper at ICLR 2025
Input view 2
Input view 1Input view 3Dynamic
Static
Rendering flow
Gradient flow
Occlusion
(a)
 (b)
Figure 4: (a) Diagram illustrating dynamic parameter doptimization. Even when static points (blue)
are occluded by dynamic points (orange) from View 1, they can still be correctly optimized from
View 2 and 3. (b) shows the result of decomposition. From top left to bottom right, the order is GT
mask, dynamic parameter rendered image, dynamic point and static point rendering results.
Where Nis the number of Gaussian points that intersect with the pixel x∈R2. In the actual
implementation, the covariance matrix Σis typically decomposed into rotation qand scaling s. The
color cis represented by a spherical harmonics (SH) function. Therefore, a Gaussian point can be
represented as G{µ, q, s, σ, c }.
3.2 E FFICIENT DYNAMIC AND STATIC DECOMPOSITION
In this section, we introduce our dynamic-static decomposition method for eliminating redundant
computations for static Gaussian points. The time-varying motion model is applied solely to
the dynamic components, leaving the static elements unchanged. This approach leads to faster
convergence and enhanced rendering quality. Specifically, we introduce a learnable dynamic
parameter d(Initialized to 0.) within the Gaussian points to quantify the the dynamic level of
each points. A higher dcorresponds to more pronounced motion, indicating that the point is likely
dynamic. We first compute a 2D dynamic-static pixel mask D(x)from the training videos to
distinguish dynamic and static pixels, as shown in Eq. 3, which serves as the supervision signal.
D(x) =
1 S(x)>=γ
0 S(x)< γwhere S(x) =vuut1
TTX
t=1(C(x, t)−1
TTX
t=1C(x, t))2(3)
where C(x, t) represents the pixel intensity of in tth frame at location x∈R2.S(x)is the temporal
standard deviation (std) for each pixel xacross the entire time duration T. Subsequently, a threshold
ofγ= 0.02is applied to binarize S(x), generating a pixel mask D(x). Pixels with S(x)greater
than or equal to γare classified as dynamic, while those below are considered static.
Based on the concept: During backpropagation, Gaussian points intersecting dynamic pixels should
receive a positive gradient, while those intersecting static pixels should receive a negative gradient,
with the gradient gradually weakening with distance and occlusion. Our decomposition design is
illustrated in Fig. 4(a). We use the αcomposition for parameter dwith Sigmoid function to render
a dynamic value ˆD(x)at location x, as Eq. 4 shows.
ˆD(x) =Sigmoid (X
i=1diαi(x)i−1Y
j=1(1−αj(x))) (4)
By applying the sigmoid function, we optimize the dynamic parameter dto span (−∞,+∞),
enabling finer differentiation of dynamic degrees. This approach converts our dynamic-static
decomposition into a binary classification problem. Consequently, optimizing the dynamic value
for each Gaussian point can be accomplished by minimizing the binary cross-entropy loss:
Ld=Ex[−D(x)log( ˆD(x))−(1−D(x))log(1 −ˆD(x))]. (5)
From the equation above, we effectively optimize the dynamic parameter dfor the Gaussian points.
The entire optimization process is highly efficient, typically concluding within 1 minute. Ultimately,
5

---

## Page 6

Published as a conference paper at ICLR 2025
Gaussian points with dynamic parameter greater than the dynamic threshold ζ= 7.0are classified
as dynamic; otherwise, they are classified as static. An ablation study on ζis presented in Fig. 7 and
Sec. 5, demonstrating that our decomposition method is robust to the choice of ζ.
Notably, our method adapts well to occlusion. In Fig. 4(a), while dynamic pixels (orange) from
View 1 incorrectly assign positive dynamic values to static Gaussian points (blue), other views like
View 2 and 3 assign larger negative values, ensuring correct classification.
3.3 S PATIO -TEMPORAL STRUCTURE
We introduce our proposed efficient spatio-temporal structure encoder, the 4D multi-resolution hash
h4d, and the deformation decoder MLPs, used to predict the deformation of each dynamic Gaussian.
4D Multi-resolution hash encoder. Inspired by INGP M ¨uller et al. (2022), we propose utilizing a
4D multi-resolution hash h4dfor encoding to effectively model the temporal information of dynamic
Gaussians by normalizing the point cloud into the hash grid range. As described in INGP, voxel grid
at each resolution is mapped to a hash table that stores F-dimensional learnable feature vectors. For
a given 4D dynamic Gaussian (µ, t)∈R4, its hash encoding at resolution l, denoted as h4d(µ, t;l)∈
RF, is computed through linear interpolation of the feature vectors associated with corners of the
surrounding grid. Consequently, its multi-resolution hash encoding features are as follows:
fh= [h4d(µ, t; 0), h4d(µ, t; 1)...h4d(µ, t;L−1)]∈RLF, (6)
where Ldenotes the number of resolution levels, typically set to 16. Following this, a small MLP ϕd
combines all features to produce fd=ϕd(fh). Using the 4D hash h4das an encoder offers several
advantages: compactness, O(1)query complexity, and the multi-resolution approach effectively
integrates global and local information.
However, while 4D Hash offers O(1)query complexity, its hashing characteristics make encoding
the temporal information of an entire scene both challenging and storage-intensive. Fortunately, our
proposed decomposition method focuses on encoding the temporal information of dynamic points
only, reducing the need for a larger hashing space and simplifying the modeling of the scene’s
temporal domain. This approach enables us to retain the fast access speed of 4D Hash while
minimizing storage requirements.
Multi-head Gaussian Deformation Decoder. Once all features of dynamic Gaussian points are
encoded, we can compute any required variables using a multi-head Gaussian deformation decoder
MLPs = {ϕµ, ϕs, ϕq, ϕσ, ϕsh}:
dµ,ds,dq,dσ,dsh=MLPs (fd) (7)
Here, d µ,ds,dq,dσ,dshrepresent the deformation intensity of the mean, scaling, rotation, opacity,
and color of the Gaussian point at time t. Therefore, the deformed parameters of dynamic Gaussian
Gdcan be expressed as:
(µ′, r′, q′, σ′, sh′) = (µ+dµ, r+dr, q+dq, σ+dσ, sh +dsh) (8)
where ( µ′, r′, q′, σ′, sh′) represent the new parameters of the dynamic Gaussian at time t. For static
Gaussian elements Gs, they are directly combined with the deformed dynamic Gaussian elements
Gdto render the final rendered image It.
3.4 D ENSITY CONTROL
In the original 3DGS, the opacity of all points is regularly reduced, and Gaussian points with low
transparency are clipped during the pruning stage. However, this approach is not appropriate for our
method as it results in excessive coupling between the canonical space and the deformation space.
Therefore, we eliminate the reset opacity operation. Inspired by previous works Niemeyer et al.
(2024); Deng et al. (2024); Fan et al. (2023), which focus on the compact representation of static
scenes by pruning redundant Gaussians based on spatial attributes such as transparency and volume.
We adopt a novel approach to pruning floaters across canonical and deformation space: Temporal
Importance Pruning, as shown in Fig. 3. This involves calculating the importance of each Gaussian
point to each training viewpoint at every timestamp. Gaussians with importance below a certain
6

---

## Page 7

Published as a conference paper at ICLR 2025
Table 1: Quality comparison on the N3DV dataset. The best and the second best results are
denoted by red and blue.1online method.
Method PSNR ↑DSSIM ↓LPIPS ↓ Time↓ Size(MB) ↓FPS↑
DyNeRF Li et al. (2022b) 29.58 0.020 0.099 1300.0 hours 30 0.02
NeRFPlayer Song et al. (2023) 30.69 - 0.111 6.0 hours 5100 0.05
HexPlane Cao & Johnson (2023) 31.70 0.014 0.075 12.0 hours 240 0.21
K-Planes Fridovich-Keil et al. (2023) 31.63 0.018 - 5.0 hours 300 0.15
4DGS Wu et al. (2024) 31.02 0.030 0.150 50 mins 90 30
3DGStream1Sun et al. (2024) 31.67 - - 60 mins 2340 215
SpaceTimeGS Li et al. (2024) 32.05 0.014 0.044 10.0 hours 200 110
Real-Time4DGS Yang et al. (2023) 32.01 0.014 0.055 9.0 hours >1000 114
Swift4DLite(Ours) 31.79 0.017 0.072 20 mins 30 128
Swift4D(Ours) 32.23 0.014 0.043 25 mins 120 125
Table 2: Quantitative comparison on the MeetRoom dataset. PSNR is averaged across all frames,
while training time and storage requirements accumulate over the entire sequence.1online method.
Method PSNR ↑Time(hours) ↓Size(MB) ↓
PlenoxelFridovich-Keil et al. (2022) 27.15 70 304500
I-NGPM ¨uller et al. (2022) 28.10 5.5 14460
3DGSKerbl et al. (2023) 31.31 13 6330
StreamRF1Li et al. (2022a) 26.72 0.85 2700
3DGStream1Sun et al. (2024) 30.79 0.6 1230
Swift4D(Ours) 32.05 0.3 40
threshold can be clipped, effectively reducing floater issues. For a Gaussian point gi, the importance
wiis calculated as follows:
wi= max
I∈I,x∈I,t∈T(αi(x|t)i−1Y
j=1(1−αj(x|t))) (9)
Here,Irepresents the images from all training views, αi(x|t)is the value of αi(x)at time tin Eq. 2,
Trepresents the set of query times. We prune Gaussians when their importance satisfies wi<0.02.
As illustrated in Fig. 9, this method effectively eliminates artifacts that are suspended in the air and
were not captured by the training views. For the cloning and splitting of Gaussians, we adhere to
the procedures of 3DGS, with the child Gaussians inheriting the dynamic properties of their parent
Gaussians.
3.5 O PTIMIZATION PIPELINE
We start by initializing the SfM Schonberger & Frahm (2016) point cloud using the first frames,
then train on the first-frame images for 5000 iterations to establish a well-defined canonical space.
Next, training the dynamic attributes of each Gaussian point within the canonical space takes about 1
minute, followed by training the spatio-temporal structure. Consistent with the principles of 3DGS,
our loss function remains simple, without additional terms:
Lrec= (1−λ1)L1+λ1LSSIM
4 E XPERIMENT
In this section, we provide details of our implementation and datasets in Sec. 4.1 and 4.2,
respectively. A thorough analysis of the experimental results is presented in Sec. 4.3, while Sec. 4.4
covers the ablation experiments for our method. The results show that our approach achieves sota
performance in terms of training speed, storage efficiency, and rendering quality.
4.1 I MPLEMENTATION DETAILS
We initialize with point clouds generated by Colmap, followed by constructing our canonical space
using the first frames from all training viewpoints, trained for 5000 epochs. Next, we train the
7

---

## Page 8

Published as a conference paper at ICLR 2025
(a) GT
 (b) Ours
 (c) 3DGStream
 (d) 3DGS
Figure 5: Qualitative result on the discussion .
dynamic parameter dof Gaussian points using the Adam optimizer Kingma & Ba (2014) with a
learning rate of 0.05. This training spans 3000 epochs and completes in under 1 minute. Finally,
we train our spatio-temporal structure for approximately 14000 epochs, utilizing settings for the 4D
Hash table similar to those in InstantNGP M ¨uller et al. (2022). We use the Adam optimizer with
an initial learning rate of 0.002, which exponentially decays to 0.00002 over the course of training.
Lite refers to a lite-version model with a hash table size set to 215andλ1= 0. All experiments were
conducted on an NVIDIA RTX 3090 GPU.
4.2 D ATASET
Table 3: Quantitative results comparison for sear
steak andflame steak includes average PSNR and
time metrics over 300 frames of the test view.
Method PSNR ↑Time(mins) ↓
Lite 33.31 20
Muti-planes 33.48 28
W/o decomp. 32.68 35
Full 33.83 25The N3DV dataset Li et al. (2022b) is
captured using a multi-view system with
18-21 cameras, recording dynamic scenes at
a resolution of 2704×2028 and 30 FPS. It
includes various complex scenarios such as fire,
reflections, and new objects. Following prior
works Li et al. (2022b); Cao & Johnson (2023);
Wu et al. (2024); Yang et al. (2023); Li et al.
(2024), we downsampled the videos by a factor
of two and used the same training and testing
data splits as established by them.
The Meet Room dataset Li et al. (2022a) is
captured using a multi-view system with 13
cameras, recording dynamic scenes at a resolution of 1280×720 and 30 FPS. Following prior
works Sun et al. (2024); Li et al. (2022a), we used 12 views for training and reserved 1 view for
testing.
The Basketball court dataset VRU (2024) is captured using a multi-view system with 34 cameras,
recording dynamic scenes at a resolution of 1920×1080 and 25 FPS. This dataset encompasses a
large scene with many complex situations, including bouncing, fast motion, occlusion, and transient
objects, making it highly challenging.
4.3 E VALUATION
For the Meetroom and Basketball court dataset, we follow the processing approach from Sun et al.
(2024), using COLMAP Schonberger & Frahm (2016) to estimate the camera pose of the first frame
as the global pose. For the N3DV dataset, we adopt the processing approach from Wu et al. (2024).
We evaluate the methods using three metrics across all 300 frames: 1) Average PSNR, DSSIM, and
LPIPS Zhang et al. (2018) scores for the test views; 2) Total training time and FPS; 3) Model size .
Tab. 1 and Fig. 6 respectively present the quantitative and qualitative evaluations of various methods
on the N3DV video dataset. As shown in Tab. 1, our approach not only significantly surpasses
previous methods in rendering quality but also achieves speeds at least 20 times faster compared to
methods achieving similar rendering quality Yang et al. (2023); Li et al. (2024). It can be seen in
Fig. 6 that our method not only achieves higher-quality modeling of static regions, such as the plate
in the bottom right corner and the background outside the window in the coffee martini , but also
provides more detailed modeling of moving regions, such as the arms. As for the MeetRoom dataset
results, shown in Fig. 5 and Tab. 2, our method achieves state-of-the-art performance in rendering
quality, training time, and storage efficiency. Particularly noteworthy is the storage efficiency, as
3DGStream requires 30 times more storage compared to our approach. The results of training on
the basketball court dataset are presented in Fig. 10 and the supplementary videos (basketball 1 and
2), showcasing our method’s ability to handle highly complex dynamic scenes. Our decomposition
8

---

## Page 9

Published as a conference paper at ICLR 2025
(a) GT
 (b) Ours
 (c) RTGS
 (d) STGS
Figure 6: Qualitative result on coffee martini andcut beef . It can be observed that our method
achieves higher-quality modeling in both dynamic and static regions.
technique effectively separates all athletes from the scene, illustrating the model’s strong adaptability
to occlusion, as discussed in Sec. 3.2.
4.4 ABLATION AND ANALYSIS
3232.232.432.632.83333.233.433.6
00.020.040.060.080.10.120.140.160.180.2
3 4 5 6 7 8 9The percentage of dynamic points PSNR
Figure 7: Distribution of dynamic points counts
and PSNR at different thresholds.Dynamic and static decomposition. To
validate the effectiveness of our dynamic-static
decomposition method, we conducted
experiments on the sear steak and flame
steak . As illustrated in Fig. 8(c) and Tab. 3,
treating all points as dynamic led to increased
computation time, significantly reduced
rendering quality, and introduced blurring in
areas with large motion amplitudes.
Muti-plane encoder. There are three
commonly used choices for encoder selection:
an implicit MLP Gao et al. (2021), multi-planes
Cao & Johnson (2023), and a hash table M ¨uller
et al. (2022). In this study, we delve into
employing the multi-plane approach to replace
the 4D Hash as the encoder in our method. The subjective and objective experimental results, shown
in Fig. 8(b) and Tab. 3, indicate that while it slightly lags behind the hash table in terms of rendering
speed and quality, it still outperforms methods that do not apply dynamic-static decomposition Wu
et al. (2024).
Temporal importance pruning. As shown in Fig. 9, (a) and (b) exhibit severe artifacts. In contrast,
images rendered with our pruning strategy, (c), appear much cleaner.
5 D ISCUSSION
Incomplete decomposition of dynamic and static points. Although we employ the pixel level
supervisor, it fails to fully decouple dynamic and static points. This can lead to two issues: Gaussian
9

---

## Page 10

Published as a conference paper at ICLR 2025
(a) Lite
 (b) Muti-plane encoder
 (c) W/O decomp
 (d) Full(Ours)
Figure 8: Some ablation experiments results on sear steak .
(a) With opacity reset (b) W/O opacity reset (c) Ours (d) GT
Figure 9: Importance Pruning Ablation Experiments: (a), (b), (c), and (d) show the rendered
results of the our model with opacity reset every 3000 iterations, without opacity reset, with our
importance pruning method, and the ground truth, respectively.
points in textureless regions of small moving objects may be mistaken for static, while static objects
may be identified as dynamic due to interference from nearby moving objects.
In the first situation, as shown in Fig. 4(b), certain textureless areas, like clothing and the table, are
mistakenly identified as static, despite being dynamic. In fact, this proves beneficial in Tab. 3. If the
entire table were labeled as dynamic, the increased dynamic points would lower rendering quality
(W/o decomp). By recognizing only the edges as dynamic, where pixel changes are significant, the
method reduces the number of dynamic points and enhances rendering quality (Ours).
In the second situation, as illustrated in Fig. 10, some Gaussians in static areas are classified as
dynamic due to the shadows or movements of the basketball players passing through these regions.
Therefore, it is reasonable to identify these areas as dynamic. Classifying these points as static will
impact the visual experience, as static points can hardly model dynamic areas.
The selection of the dynamic threshold. As shown in Fig. 7, experiments with the ”cook spinach”
scene revealed that varying the dynamic threshold ζfrom 3 to 9 did not significantly affect PSNR
or the percentage of dynamic points, demonstrating the robustness of our method. To ensure
consistency, we set the threshold to 7.
The training time. Due to the large dataset (nearly 6000 images), we load images during training,
which empirically wastes around 40% of the training time on disk I/O. Eliminating this overhead
could reduce training time to 10 minutes while maintaining high-quality 4D scene reconstruction.
6 C ONCLUSION
In this paper, we introduce Swift4D, which achieves fast convergence, compact storage, and
high-quality real-time rendering capabilities within the field of 4D reconstruction. The core
innovation of our method lies in the introduction of a dynamic-static decomposition technique,
which can be applied to most existing dynamic scene reconstruction methods, enhancing quality
and accelerating convergence. Additionally, we introduce a 4D Hash encoder and a multi-head
decoder as our spatio-temporal structure, allowing for faster and more efficient temporal modeling
of dynamic points. Finally, to prevent severe coupling between the canonical and deformation fields,
we propose a novel temporal pruning method that effectively removes floaters in the scene. Our
proposed method delivers competitive results in just 5 minutes, and we hope it can offer new insights
for applications struggling with training efficiency.
Limitation: Similar to previous work Sun et al. (2024); Li et al. (2024), our mrthod focuses
on multi-view scenes and currently does not support monocular datasets for dynamic scene
10

---

## Page 11

Published as a conference paper at ICLR 2025
(a) (d) (c) (b)
Figure 10: Basketball court dataset experiment. (a) and (c) are dynamic point renderings, while
(b) and (d) are GT. The black floaters are actually Gaussian points from the dynamic background.
reconstruction. Additionally, our method focuses on scene reconstruction and does not include
human reconstruction Wu et al. (2020); Cheng et al. (2023).
7 ACKNOWLEDGEMENTS
This work is financially supported by Guangdong Provincial Key Laboratory of Ultra High
Definition Immersive Media Technology(Grant No. 2024B1212010006), National Natural
Science Foundation of China U21B2012, Shenzhen Science and Technology Program-Shenzhen
Cultivation of Excellent Scientific and Technological Innovation Talents project(Grant No.
RCJC20200714114435057), this work is also financially supported for Outstanding Talents Training
Fund in Shenzhen.
REFERENCES
Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and
Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields.
InProceedings of the IEEE/CVF international conference on computer vision , pp. 5855–5864,
2021.
Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf
360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition , pp. 5470–5479, 2022.
Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf:
Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International
Conference on Computer Vision , pp. 19697–19705, 2023.
Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 130–141, 2023.
Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance
fields. In European conference on computer vision , pp. 333–350. Springer, 2022.
Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang
Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-rendering: A diverse neural actor repository for
high-fidelity human-centric rendering. In Proceedings of the IEEE/CVF International Conference
on Computer Vision , pp. 19982–19993, 2023.
Tianchen Deng, Yaohui Chen, Leyan Zhang, Jianfei Yang, Shenghai Yuan, Jiuming Liu, Danwei
Wang, Hesheng Wang, and Weidong Chen. Compact 3d gaussian splatting for dense visual slam.
arXiv preprint arXiv:2403.11247 , 2024.
Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps. arXiv preprint
arXiv:2311.17245 , 2023.
Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo
Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition , pp. 5501–5510, 2022.
Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo
Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 12479–12488, 2023.
11

---

## Page 12

Published as a conference paper at ICLR 2025
Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic
monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision ,
pp. 5712–5721, 2021.
Bing He, Yunuo Chen, Guo Lu, Li Song, and Wenjun Zhang. S4d: Streaming 4d real-world
reconstruction with gaussians and 3d control points. arXiv preprint arXiv:2408.13036 , 2024.
Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting
for geometrically accurate radiance fields. arXiv preprint arXiv:2403.17888 , 2024a.
Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi.
Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 4220–4230, 2024b.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimk ¨uhler, and George Drettakis. 3d gaussian
splatting for real-time radiance field rendering. ACM Trans. Graph. , 42(4):139–1, 2023.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980 , 2014.
Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization for
real-time dynamic view synthesis with 3d gaussian splatting. arXiv preprint arXiv:2312.00112 ,
2023.
Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and Ping Tan. Streaming radiance fields for 3d
video synthesis. Advances in Neural Information Processing Systems , 35:13485–13498, 2022a.
Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim,
Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video
synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 5521–5531, 2022b.
Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time
dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pp. 8508–8520, 2024.
Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time
view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 6498–6508, 2021.
Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-Phuoc, Douglas Lanman, James Tompkin,
and Lei Xiao. Gaufre: Gaussian deformation fields for real-time dynamic novel view synthesis.
arXiv preprint arXiv:2312.11458 , 2023.
Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. Gaussian-flow: 4d reconstruction with dynamic
3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pp. 21136–21145, 2024.
Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pp. 20654–20664, 2024.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications
of the ACM , 65(1):99–106, 2021.
Thomas M ¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics
primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) , 41(4):
1–15, 2022.
Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel
Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari.
Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+
fps.arXiv preprint arXiv:2403.13806 , 2024.
12

---

## Page 13

Published as a conference paper at ICLR 2025
Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M
Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of
the IEEE/CVF International Conference on Computer Vision , pp. 5865–5874, 2021.
Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural
radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pp. 10318–10327, 2021.
Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srinivasan, Ben Mildenhall, Andreas Geiger, Jon
Barron, and Peter Hedman. Merf: Memory-efficient radiance fields for real-time view synthesis
in unbounded scenes. ACM Transactions on Graphics (TOG) , 42(4):1–12, 2023.
Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings
of the IEEE conference on computer vision and pattern recognition , pp. 4104–4113, 2016.
Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and
Andreas Geiger. Nerfplayer: A streamable dynamic scene representation with decomposed neural
radiance fields. IEEE Transactions on Visualization and Computer Graphics , 29(5):2732–2742,
2023.
Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast
convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition , pp. 5459–5469, 2022.
Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei Zhao, and Wei Xing. 3dgstream: On-the-fly
training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp.
20675–20685, 2024.
VRU. https://anonymous.4open.science/r/vru-sequence/. 2024.
Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei Song, and Huaping Liu. Mixed neural
voxels for fast multi-view video synthesis. In Proceedings of the IEEE/CVF International
Conference on Computer Vision , pp. 19706–19716, 2023.
Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian,
and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 20310–20320,
2024.
Minye Wu, Yuehao Wang, Qiang Hu, and Jingyi Yu. Multi-view neural human rendering. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp.
1682–1691, 2020.
Jinbo Yan, Rui Peng, Luyang Tang, and Ronggang Wang. 4d gaussian splatting with scale-aware
residual field and adaptive optimization for real-time rendering of temporally complex dynamic
scenes. In ACM Multimedia 2024 .
Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic
scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642 ,
2023.
Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 20331–20341, 2024.
Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for
real-time rendering of neural radiance fields. In Proceedings of the IEEE/CVF International
Conference on Computer Vision , pp. 5752–5761, 2021.
Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting:
Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pp. 19447–19456, 2024.
13

---

## Page 14

Published as a conference paper at ICLR 2025
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable
effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on
computer vision and pattern recognition , pp. 586–595, 2018.
14