

---

## Page 1

4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing for
Efficient and High-Fidelity Dynamic Scene Reconstruction
Woong Oh Cho, In Cho, Seoha Kim, Jeongmin Bae, Youngjung Uh, Seon Joo Kim
Yonsei University
{wocho, join, hailey07, jaymin.bae, yj.uh, seonjookim }@yonsei.ac.kr
Abstract
Modeling dynamic scenes through 4D Gaussians offers high
visual fidelity and fast rendering speeds, but comes with sig-
nificant storage overhead. Recent approaches mitigate this
cost by aggressively reducing the number of Gaussians. How-
ever, this inevitably removes Gaussians essential for high-
quality rendering, leading to severe degradation in dynamic
regions. In this paper, we introduce a novel 4D anchor-based
framework that tackles the storage cost in different perspec-
tive. Rather than reducing the number of Gaussians, our
method retains a sufficient quantity to accurately model dy-
namic contents, while compressing them into compact, grid-
aligned 4D anchor features. Each anchor is processed by an
MLP to spawn a set of neural 4D Gaussians, which repre-
sent a local spatiotemporal region. We design these neural 4D
Gaussians to capture temporal changes with minimal param-
eters, making them well-suited for the MLP-based spawn-
ing. Moreover, we introduce a dynamic-aware anchor grow-
ing strategy to effectively assign additional anchors to under-
reconstructed dynamic regions. Our method adjusts the accu-
mulated gradients with Gaussians’ temporal coverage, signif-
icantly improving reconstruction quality in dynamic regions.
Experimental results highlight that our method achieves state-
of-the-art visual quality in dynamic regions, outperforming
all baselines by a large margin with practical storage costs.
Introduction
Reconstructing dynamic scenes from multi-view videos has
received significant attention due to its wide applications.
Beyond methods based on neural radiance fields (NeRFs)
(Li et al. 2022; Wang et al. 2022; Attal et al. 2023), 3D
Gaussian Splatting (3DGS) (Kerbl et al. 2023) has become a
major approach for dynamic scene reconstruction, due to its
ability to render high-quality novel views in real-time. The
two main categories of this approach are 1) modeling tem-
poral changes as deformations of canonical 3D Gaussians
(Yang et al. 2023; Bae et al. 2024; Jiawei et al. 2024) and 2)
employing 4D Gaussians to approximate a scene’s 4D vol-
umes (Yang et al. 2024; Li et al. 2023; Lee et al. 2024c,a).
Directly optimizing 4D Gaussians offers higher visual
quality and faster rendering than deformation-based meth-
ods. By modeling temporal changes through multiple 4D
Gaussians, each covering a certain time range, they effec-
tively capture complex dynamic regions without the heavy
computation cost of deformation fields. However, this ap-proach accompanies a large number of 4D Gaussians, which
subsequently leads to substantial storage overhead–often ex-
ceeding 6GB for a 10-second video (see Figure 1, left).
Several approaches have attempted to address the storage
cost by reducing the number of Gaussians. They achieve this
by more complicated motion modeling (Lee et al. 2024b),
aggressive pruning (Li et al. 2023), or motion interpolation
(Lee et al. 2024a). While these methods effectively reduce
storage, they also inevitably remove Gaussians in dynamic
regions, sacrificing the expressiveness needed to represent
complex temporal changes. As a result, these methods often
suffer from quality degradation in dynamic components.
In this paper, we introduce a 4D anchor-based framework
that addresses the storage overhead from a different perspec-
tive. Instead of reducing the number of Gaussians–which
is crucial for maintaining the expressiveness–our method
maintains a sufficient number to render dynamic regions
with high visual quality. This is achieved by representing
dynamic scenes through structured anchor features (Lu et al.
2024a), aligned with a sparse 4D grid. Each anchor holds a
compact feature vector to model a local spatiotemporal re-
gion. This feature is processed by shared multi-layer percep-
trons (MLP) to output properties of neural Gaussians.
Considering the limited capacity of the shared shallow
MLP, we propose a compact parametrization of the neural
Gaussians that effectively captures temporal changes. This
includes two key modelings: linear motion to model the
time-varying 3D position, and a generalized Gaussian func-
tion to model temporal opacity, which is better suited for
capturing sudden appearance changes. This design enables
our framework to effectively represent complex dynamic tra-
jectories through a sequence of linear segments.
While the anchor-based scheme effectively reduces stor-
age, a na ¨ıve extension of 3D scaffolding (Lu et al. 2024a)
fails to accurately capture dynamic regions (see Figure 1,
Scaff-naive). The limitation arises from the anchor grow-
ing strategy used in (Lu et al. 2024a), which is originally
designed for static scenes. In our 4D framework, each an-
chor and its neural Gaussians are active only within a spe-
cific time range, while the anchor growing strategy accu-
mulates gradients across all frames without considering this
coverage. As a result, dynamic regions appearing in only a
few frames receive less gradient signals, leading to under-
reconstructed dynamic components in the final scene.arXiv:2411.17044v2  [cs.CV]  5 Aug 2025

---

## Page 2

Ours
Ex4DGS
4DGS
 Scaff -
Naive
Ground
Truth
STG
Figure 1: Dynamic region quality comparison. (left) Comparison of dynamic region quality ( y-axis) versus storage cost ( x-
axis) on N3DV . (right) Results on the flame salmon scene from N3DV . Scaff-naive refers to the naive anchor-based model
without our neural Gaussian design and dynamic-aware anchor growing. Our method significantly outperforms all baselines
with efficient storage usage, while other methods either exhibit degraded quality in dynamic regions or excessive storage costs.
We present a dynamic-aware anchor growing strategy to
properly allocate new anchors to under-reconstructed dy-
namic regions. Our approach adjusts the accumulated gra-
dients based on each Gaussian’s temporal coverage. We ad-
just dynamic Gaussians with shorter coverage to have larger
gradient, compensating for the penalty caused by their short
appearance. This modification encourages new anchors to
be allocated more in under-reconstructed dynamic regions,
achieving substantial improvement in reconstruction quality.
Our anchor-based framework, coupled with the dynamic-
aware anchor growing strategy, represents complex dynamic
components in high-quality while addressing storage costs.
We validate our approach through extensive experiments
conducted on the N3DV (Li et al. 2022) and Technicolor
(Sabater et al. 2017) datasets. Notably, our method outper-
forms all baselines by a large margin with practical storage
overhead, as illustrated in Figure 1.
Related Work
Efficient 3D Gaussians. In recent years, 3D Gaussian
Splatting (3DGS) has attracted significant attention for
achieving real-time rendering by representing scenes with
3D Gaussian primitives and introducing a tile-based raster-
izer. For photorealistic rendering quality, 3DGS requires a
significant number of 3D Gaussians, which increases storage
costs. To address this problem, previous works remove un-
necessary Gaussians that do not harm rendering quality (Fan
et al. 2023; Girish, Gupta, and Shrivastava 2023), replace
Gaussians or their parameters with efficient representations
(Niedermayr, Stumpfegger, and Westermann 2024; Papan-
tonakis et al. 2024), and compress Gaussian parameters us-
ing existing compression techniques such as entropy cod-
ing and image compression (Niedermayr, Stumpfegger, and
Westermann 2024; Chen et al. 2025; Xie et al. 2025). In this
paper, we address compressing Gaussians in spatiotemporal
space, which have been relatively unexplored compared to
3D Gaussians.Dynamic 3D Gaussians. Two main approaches are pro-
posed to extend 3DGS (Kerbl et al. 2023) into dynamic
scene reconstruction. The first approach involves deform-
ing 3D Gaussians along with temporal changes (Yang et al.
2023; Wu et al. 2023; Bae et al. 2024; Shaw et al. 2024; Ji-
awei et al. 2024; Kwak et al. 2025). These deformable 3D
Gaussians offer the advantage of compact storage require-
ments but exhibit relatively slow rendering speeds and low
visual quality. In contrast, the other approach directly em-
ploys 4D Gaussians in the spatio-temporal domain. 4DGS
(Yang et al. 2024) demonstrate superior visual quality and
faster rendering speeds, but suffer from higher storage re-
quirements. Although some works (Li et al. 2023; Lee et al.
2024a,c) improve storage efficiency using fewer Gaussians,
they tend to focus on the holistic scene, thereby neglect-
ing the quality of dynamic areas. To address this, we take
an alternative approach: reducing storage while maintaining
quality by preserving the Gaussian count.
Feature-based neural rendering. Recently, a growing
trend in scene reconstruction has been to integrate neural
features as additional inputs to enhance model performance.
For instance, there have been attempts to extract features
from source views and utilize them for novel view synthesis,
enabling few-view reconstruction (Yu et al. 2021; Chen et al.
2021) or view interpolation through transformers (Wang
et al. 2021; Reizenstein et al. 2021; T et al. 2023). In 3DGS
studies, Compact3DGS (Deng et al. 2024) uses a hash grid
instead of per-Gaussian SH coefficients and STG (Li et al.
2023) renders features into RGB via shallow MLPs. Some
works generate Gaussian attributes from a multi-level tri-
plane (Wu and Tuytelaars 2024) and predict the attributes of
local 3D Gaussians from anchor features (Lu et al. 2024a).
We introduce a 4D anchor-based framework that includes
dynamic linear motion and temporal opacity derived from
a generalized Gaussian distribution, considering real-world
dynamics.

---

## Page 3

4D anchorlearnable 4D offsets3D Gaussians
shared MLPsd
view directionfeature vector
sparse 4D grid𝑥𝑥𝑦𝑦
𝑡𝑡𝑡𝑡=𝑡𝑡1𝑡𝑡=𝑡𝑡0
neural 4D Gaussians𝑥𝑥𝑦𝑦
𝑡𝑡Splatting
rendered image4D positionp = (𝑥𝑥,𝑦𝑦,𝑧𝑧,𝑡𝑡)
𝐪𝐪,𝒔𝒔,𝜎𝜎𝜌𝜌
𝐯𝐯
𝐜𝐜
𝐾𝐾 Gaussians
Velocity 
MLPOpacity
MLP
Shape 
MLP
Color 
MLP4D to 3D𝐱𝐱
+
𝑡𝑡=𝑡𝑡2
𝜇𝜇,𝛼𝛼 | 𝑡𝑡𝑟𝑟Figure 2: Proposed 4D anchor-based framework. We begin with a sparse 4D grid of anchors, each defined by a unique 4D
spatiotemporal position pand learnable offsets. Shared MLPs utilize these anchors to generate neural 4D Gaussians, capturing
dynamic appearance changes with view direction d. These 4D Gaussians are then projected to 3D Gaussians at specific time tr
for rendering, using a 3D Gaussian splatting pipeline to produce the final rendered image. We omit the z-axis for simplicity.
Method
We reconstruct dynamic 3D scenes through a sparse, grid-
aligned 4D anchor grid. Each anchor holds a compressed
feature vector that specifies nearby 4D Gaussians. The
dynamic-aware anchor growing strategy encourages new an-
chors to be closely placed at under-reconstructed dynamic
regions. Furthermore, we design each 4D Gaussian to move
along a line segment, and define the temporal opacity func-
tion which fits a piecewise persistent period. Through these
components, our framework employs a sufficient number of
Gaussians to achieve high-quality reconstruction of dynamic
regions, while the storage overhead is significantly reduced
via compressed anchor features.
4D Anchor-Based Framework
The overview of our method is illustrated in Figure 2. Our
method begins with a set of sparse 4D anchor points, each of
which has a unique, grid-aligned spatiotemporal 4D position
p∈R4with a feature vector f∈RC. We leverage shared
MLPs to produce Kneural 4D Gaussians from these anchor
features. Corresponding 3D Gaussians are computed from
these 4D Gaussians to render a frame at timestep t.
Initializing 4D anchors. Similar to 3D Scaffold-GS (Lu
et al. 2024b), we utilize static point clouds to initialize 4D
anchor points. We first obtain the static point cloud from
multi-view frames at a certain timestep t0using Structure-
from-Motion (SfM) (Sch ¨onberger and Frahm 2016). The
4D positions of anchors are initialized from a set of voxels
V∈RN×3obtained from this point cloud:
p= (xv, yv, zv, t0),∀v∈V, (1)
xk=p+ ∆xk, k∈ {1,2, ..., K}, (2)
where ( xv,yv,zv) is the center coordinates of the voxel v.
Each anchor point also accompanies two learnable parame-
ters: a feature vector f, and 4D offsets ∆xk∈R4for deter-
mining properties of Kneural 4D Gaussians. 4D Gaussian
position xis determined by anchor position and offset.
Neural 4D Gaussians. We leverage shared MLPs and
learnable parameters of the 4D anchors to generate Kneural4D Gaussians from each anchor. Specifically, shared MLPs
take the anchor feature fand yield properties of Kneural
Gaussians, which include base opacity ρ∈R, quaternion
q∈R4and scaling s∈R3for the covariance matrix, view-
dependent color c∈R3. Our shared MLPs also produce
a temporal scale σ∈Rto compute our temporal opacity,
and the neural velocity u∈R3. Note that the color MLP
additionally takes the view direction d∈R3as inputs for
view-dependent modeling.
Rendering neural Gaussians. To render the neural 4D
Gaussians, we compute 3D Gaussian parameters at time tr
from our time-varying properties. For the k-th Gaussian of
the anchor p, the center µk∈R3and the opacity αk∈Rof
the corresponding 3D Gaussian at time trare derived as
µk=xxyz
k+h(tr,xt
k,u), (3)
αk=ρk·g(tr,xt
k, σk), (4)
where xxyz
k∈R3andxt
k∈Rare the spatial and the tem-
poral position of the 4D Gaussian. h(·),g(·)model time-
varying values of the positions and the opacity with the neu-
ral velocity and temporal opacity function, which will be fur-
ther described in the following section.
After deriving 3D Gaussians at time tr, we utilize the
existing 3D Gaussian splatting pipeline (Kerbl et al. 2023)
to render the frames. Same as 3D Scaffold-GS (Lu et al.
2024b), only Gaussians within the view frustum and having
opacity higher than the threshold are passed to the renderer.
Compact Parametrization of 4D Gaussians
The properties of neural 4D Gaussians are predicted by a
shallow shared MLP. Increasing the number of Gaussian
properties complicates the optimization process. To this end,
we propose a compact parametrization of 4D Gaussians that
captures temporal changes as a set of linear segments. Our
neural Gaussian design involves two key modelings. First,
we represent time-varying spatial positions as linear mo-
tions. Second, we employ a generalized Gaussian function
to model temporal opacity, which effectively captures sud-
den appearance changes with a single parameter.

---

## Page 4

tunivariat e Gaussian
α(t)tgeneralized Gaussian
α(t)ΣFigure 3: Illustration of the modified temporal opacity.
To reconstruct the sudden and continuous appearance of an
object, the univariate Gaussian function requires the sum of
multiple Gaussians, while the generalized Gaussian achieves
the same expressiveness with only a single Gaussian.
Linear motion. As described in Eq. (3), the time-varying
3D position of the 3D Gaussian µk∈R3is determined by
the time-varying function h(·). We formulate the h(·)as a
linear motion based on the neural velocity u:
h(t,xt
k,u) = (t−xt
k)u. (5)
This formulation simplifies the temporal slicing of Gaussian
positions proposed in 4DGS (Lee et al. 2024a), using only
3 parameters per Gaussian. Despite its simplicity, a set of
Gaussians with linear motion effectively captures the tempo-
ral dynamics of scene elements, particularly when combined
with our modified temporal opacity.
Modified temporal opacity. Previous 4D Gaussian meth-
ods (Yang et al. 2024; Li et al. 2023) typically adopt the uni-
variate Gaussian function to make Gaussians appear within
a specific time range. However, this formulation lacks the
capacity to model abrupt changes in real-world scene ele-
ments, limiting the expressiveness of individual Gaussians.
We reformulate the temporal opacity function based on
a generalized Gaussian distribution. This offers steeper
derivatives at the beginning and the end, making it better
suited to fit piecewise persistent periods. Our time-varying
opacity g(·)described in Eq. (4) is formulated as
g(t,xt
k, σk) =exp(−(|t−xt
k|/σk)β), (6)
where βis a tunable hyperparameter. In practice, we set the
inverse sigma as a learnable parameter, as it leads to more
stable training. We also model β= 2β′andβ′as a hyperpa-
rameter, which removes the | · |in the above equation.
While the issue of temporal opacity is also explored in
(Lee et al. 2024a), we show that efficiently representing tem-
poral coverage with fewer parameters is crucial for the per-
formance of the anchor-based scheme.
Dynamic-Aware Anchor Growing
Growing new anchors to under-reconstructed dynamic re-
gions is a critical factor for high-quality results. The direct
application of previous anchor growing to the dynamic re-
construction fails to identify such regions, as it neglects the
temporal coverage of the Gaussians.The previous anchor growing strategy (Lu et al. 2024b)
designed for static scenes gathers the gradients as a mean
overNiterations:
∇g=PN∥∇ 2D∥
N. (7)
With this strategy, dynamic regions appearing in a short pe-
riod will have lower ∇g, as these regions will be penalized
by the denominator Nregardless of their actual errors.
To address this, we propose a dynamic-aware anchor
growing operation. The core of our anchor growing is the
computation of the accumulated gradients of each 4D Gaus-
sian∇g. We formulate ∇gas a weighted sum of the gradient
overNiterations:
∇g=PNw(α′, σ)∥∇ 2D∥PNw(α′, σ), (8)
where ∇2Dis the 2D position gradient of the Gaussian, and
α′=g(tr,xt, σ)is the time-variant component of the Gaus-
sian opacity. We define the weight term was a function of α
andσ, with γas a hyperparameter:
w(α′, σ) =α′(1/σ)γ. (9)
We collect the accumulated gradients of all Gaussians and
then voxelize them. New anchors are placed at the centers of
the voxels having ∇2Dhigher than each threshold.
The dynamic-aware anchor growing strategy differs from
the original anchor growing operation (Lu et al. 2024a) in
two key aspects. First, gradients are accumulated only when
the Gaussian’s temporal opacity is greater than zero ( i.e.,
when it is activated). It allows our method to accurately
gather the gradients of 4D Gaussians placed at the regions
where dynamic scene elements only appear in a short period.
Second, the temporal coverage σdirectly influences to the
anchor growing operation. This encourages regions with a
short temporal coverage to receive stronger gradients, mak-
ing it easier for anchors to grow in those areas. By accumu-
lating gradients as a weighted sum, our method effectively
places new anchors to under-reconstructed dynamic regions.
Training and Implementation Details
Training objective. Both learnable anchor parameters and
MLP weights are jointly optimized through the rendering
loss. We employ L1with SSIM loss LSSIM and volume reg-
ularization Lvolfollowing the 3D Scaffold-GS (Lu et al.
2024a). The full training objective with weighting coeffi-
cients λSSIM= 0.2andλvol= 0.01is
L= (1−λSSIM)L1+λSSIMLSSIM+λvolLvol. (10)
Implementation details. Similar to 4DGS, our model is
trained for 120K iterations with a single batch, which takes
approximately 3 hours on a single NVIDIA A6000 GPU.
We set β= 2 andγ= 1 for our modified temporal opac-
ity model and dynamic-aware densification. For 4D voxel
grid size, we use 0.001 as the spatial grid size and use the
frame interval time as the temporal grid size for all scenes.
We set K= 10 following (Lu et al. 2024a). After training,

---

## Page 5

Dynamic region Full region Computational cost
Model PSNR ↑SSIM↑LPIPS ↓#Gaussians PSNR ↑SSIM↑LPIPS ↓#Gaussians FPS ↑Storage ↓Training time ↓
4DGaussian (2023) 25.33 0.833 0.166 - 30.71 0.935 0.056 192 K 51.9 57 MB 50 m
E-D3DGS (2024) 26.92 0.884 0.112 - 32.04 0.951 0.034 180 K 74.5 66 MB 1 h 52 m
Grid4D (2024) 26.65 0.877 0.129 - 31.92 0.949 0.039 202 K 127.1 48 MB 1 h 20 m
STG (2023) 25.84 0.860 0.127 44 K 31.24 0.941 0.051 434 K 125.9 63 MB 1 h 8 m
4DGS (2024) 27.65 0.907 0.075 3306 K 32.14 0.947 0.047 3333 K 61.4 6194 MB 9 h 30 m
4DGS†(1GB) 26.70 0.877 0.123 557 K 31.59 0.943 0.052 581 K 152.7 1080 MB 3 h 37 m
Ex4DGS (2024a) 26.33 0.874 0.121 52 K 32.01 0.947 0.048 268 K 91.9 115 MB 1 h 6 m
Scaff-naive 24.79 0.811 0.199 206 K 31.21 0.942 0.053 732 K 159.7 83 MB 2 h 57 m
Ours-light 27.50 0.902 0.076 181 K 31.54 0.944 0.045 314 K 148.2 90 MB 2 h 51 m
Ours 28.86 0.927 0.054 533 K 32.03 0.947 0.041 775 K 129.9 149 MB 3 h 6 m
Table 1: Quantitative results on the N3DV dataset. 4DGS†refers to the result with 1GB storage, for fair comparisons.
we prune the invalid anchor points of which all Gaussians
have negative opacity. To improve inference speed, we cache
the time-invariant and view-invariant outputs of MLPs. We
provide more implementation details in the supplements.
Experiments
In this section, we first evaluate our method through the
comparisons with several state-of-the-art baselines for dy-
namic scene reconstruction. Then we conduct ablations and
analysis to explore the effectiveness of our main features.
Our code will be made publicly available.
Datasets. We evaluate our method on two representative
real-world datasets, which are Neural 3D Video (N3DV) (Li
et al. 2022) and Technicolor (Sabater et al. 2017).
•Neural 3D Video dataset (N3DV) comprises 17 to 21
synchronized videos of six scenes, with each video con-
taining 300 frames. Following previous works, we down-
sample the resolution to 1352×1014 and use the first
camera as the test view. We exclude cam13 for the cof-
feemartini scene due to synchronization issues.
•Technicolor dataset contains 16 synchronized videos
across five scenes, with each video comprising 50
frames. We retain the original resolution of 2048×1088
and use cam10 as the test view.
Baselines. We choose the following state-of-the-art com-
petitors: 4DGaussian (Wu et al. 2023), E-D3DGS (Bae
et al. 2024) and Grid4D (Jiawei et al. 2024) represent-
ing deformation-based methods, and 4DGS (Yang et al.
2024), STG (Li et al. 2023), C3DGS (Lee et al. 2024c) and
Ex4DGS (Lee et al. 2024a) representing 4D Gaussians. We
reproduce them using the official code to report their perfor-
mance. We also introduce Scaff-naive as an additional base-
line. This is an anchor-based model without our neural Gaus-
sian design and the anchor growing, which instead adopts
linear motion and the temporal opacity modeling used in
(Yang et al. 2024). We present a variant of our model, Ours-
light, which employs a larger voxel size to achieve a more
reduced storage footprint. For fair comparisons, we report
STG trained using 300 frames and initial SfM points derived
from the first frames, same as other methods.Metrics. To assess the reconstruction quality, we measure
peak signal-to-noise ratio (PSNR), structural similarity in-
dex (SSIM), and LPIPS (Zhang et al. 2018) of the rendered
images. To compute LPIPS, we follow (Bae et al. 2024) and
employ AlexNet (Krizhevsky, Sutskever, and Hinton 2012).
To measure the visual quality in dynamic regions, we use a
combined mask of Global-Median and Temporal-Difference
(Li et al. 2022), binarized with a threshold of 50. We as-
sess storage efficiency by calculating the total size of the
output files, including MLP weights, as well as Gaussian
and anchor parameters. In addition to storage size, we also
report the number of Gaussians in both dynamic and full
regions. FPS and training time are measured on a single
NVIDIA A6000 GPU, using the flame salmon andFabien
scenes from N3DV and Technicolor, respectively.
Experimental Results
N3DV . Table 1 presents the quantitative results on the
N3DV dataset. Our model outperforms all baselines in terms
of visual quality in dynamic regions, while also deliver-
ing competitive results in full regions. While our method
employs more Gaussians to achieve high-quality results,
it maintains efficient storage overhead and FPS, thanks to
the anchor-based compression scheme. Our storage-efficient
variant (Ours-light) achieves the second-best storage effi-
ciency among 4D Gaussian methods, while also surpass-
ing other efficient baselines in visual quality. Scaff-naive
yields degraded results, indicating that the na ¨ıve extension
of the scaffolding is insufficient to accurately reconstruct
dynamic 3D scenes. 4DGS often models static components
through multiple dynamic Gaussians, resulting in an exces-
sive number of Gaussians and storage costs. Although our
method does not leverage additional techniques to model
far-background areas (Li et al. 2023; Yang et al. 2024), it
still exhibits plausible reconstruction quality in both static
and full regions.
The effectiveness of our method is also demonstrated in
qualitative results. As shown in Figure 4, our model effec-
tively represents dynamic regions with a sufficient number
of Gaussians, thereby achieving high-quality dynamic re-
construction. In contrast, other efficient baselines struggle

---

## Page 6

4DGaussians E-D3DGS Grid4D 4DGS STG Ex4DGS Ours GT
flame_steak cook_spinach
Figure 4: Qualitative comparisons on the N3DV dataset. Our method achieves high-fidelity in both static and dynamic
regions.
/uni00000015/uni00000013 /uni00000017/uni00000013 /uni00000019/uni00000013 /uni0000001b/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000014/uni00000015/uni00000013 /uni00000014/uni00000017/uni00000013/uni00000014/uni00000019/uni00000013/uni00000014/uni0000001b/uni00000013
/uni00000036/uni00000057/uni00000052/uni00000055/uni00000044/uni0000004a/uni00000048/uni00000003/uni0000000b/uni00000030/uni00000025/uni0000000c
/uni00000015/uni00000018/uni00000011/uni00000018/uni00000015/uni00000019/uni00000011/uni00000013/uni00000015/uni00000019/uni00000011/uni00000018/uni00000015/uni0000001a/uni00000011/uni00000013/uni00000015/uni0000001a/uni00000011/uni00000018/uni00000015/uni0000001b/uni00000011/uni00000013/uni00000033/uni00000036/uni00000031/uni00000035/uni00000003/uni0000000b/uni00000047/uni00000025/uni0000000c
/uni00000036/uni00000046/uni00000044/uni00000049/uni00000049/uni00000052/uni0000004f/uni00000047/uni00000010/uni00000051/uni00000044/uni0000004c/uni00000059/uni00000048
/uni00000015/uni00000013/uni00000013/uni0000002e/uni00000028/uni0000005b/uni00000017/uni00000027/uni0000002a/uni00000036
/uni00000017/uni0000001a/uni0000002e
/uni00000036/uni00000037/uni0000002a
/uni00000016/uni00000013/uni0000002e
/uni00000016/uni00000014/uni0000002e/uni00000018/uni0000001b/uni0000002e/uni00000018/uni00000017/uni0000002e/uni00000014/uni00000013/uni00000014/uni0000002e/uni00000014/uni0000001b/uni00000015/uni0000002e/uni00000015/uni00000016/uni00000014/uni0000002e/uni00000016/uni00000018/uni0000001a/uni0000002e/uni00000032/uni00000058/uni00000055/uni00000056
Figure 5: Quality-storage trade-off . The results on the
flame steak scene of N3DV , along with the number of Gaus-
sians, are reported. We increase the number of employed
Gaussians of our model by adjusting the anchor grid size.
to capture complex scene dynamics (see the boxed regions).
Please refer to the supplements for more visual comparisons,
including uncropped results and videos.
Technicolor. We deliver the quantitative comparisons on
the Technicolor dataset in Table 2. Our model achieves
state-of-the-art performance across all visual quality eval-
uation metrics. This is further exemplified by our storage-efficient model, Ours-light. Despite its compact size of ap-
proximately 100 MB, Ours-light delivers competitive visual
quality on par with 4DGS, highlighting the effectiveness of
the proposed anchor-based scheme. Qualitative comparisons
on the Technicolor dataset are provided in the supplements.
Analysis and Ablation Study
To provide deeper insights of our main features, we conduct
the ablations and analysis in dynamic regions on N3DV .
Quality-storage trade-off. We first analyze the quality-
storage trade-off of our model by gradually increasing the
number of employed Gaussians. This is controlled by ad-
justing the voxel size of the anchor grid. As presented in Fig-
ure 5, the performance of our model is improved with more
Gaussians, confirming the necessity of using a sufficient
quantity. Notably, our method outperforms other storage-
efficient baselines when using similar or smaller storage
costs. Thanks to the anchor-based scheme, our method em-
ploys 3.37×or7.0×more Gaussians and achieves higher
visual quality, yet still maintains a lower storage overhead.
Understanding the anchor growing. We then ablate the
proposed anchor growing strategy. As reported at the bot-
tom of Table 3, the dynamic-aware anchor growing leads to
noticeable improvement in visual quality. By properly allo-
cating new anchors to under-reconstructed dynamic regions,
our model accurately represents dynamic scenes with prac-
tical storage overhead.

---

## Page 7

Dynamic region Full region Computational cost
Model PSNR ↑SSIM↑LPIPS ↓#Gaussians PSNR ↑SSIM↑LPIPS ↓#Gaussians FPS ↑Storage ↓Training time ↓
4DGaussian (2023) 23.99 0.709 0.280 - 29.62 0.843 0.176 268 K 23.4 72 MB 31 m
E-D3DGS (2024) 29.39 0.886 0.148 - 33.38 0.907 0.100 212 K 60.8 77 MB 2 h 34 m
STG (2023) 28.35 0.869 0.174 71 K 33.33 0.912 0.096 132 K 149.0 30 MB 1 h 24 m
4DGS (2024) 31.91 0.930 0.093 5597 K 33.30 0.910 0.095 5759 K 141.5 10699 MB 6 h 11 m
4DGS†(1GB) 29.18 0.890 0.144 538 K 28.72 0.856 0.166 591 K 156.6 1098 MB 3 h 36 m
C3DGS (2024c) 27.21 0.843 0.209 87 K 32.52 0.901 0.110 122 K 178.7 18 MB 1 h 9 m
Ex4DGS (2024a) 30.58 0.918 0.109 120 K 33.40 0.914 0.088 426 K 78.4 140 MB 1 h 53 m
Scaff-naive 28.00 0.853 0.195 721 K 33.34 0.915 0.081 2056 K 127.0 129 MB 1 h 59 m
Ours-light 30.95 0.916 0.117 186 K 33.97 0.923 0.074 480 K 131.4 108 MB 1 h 40 m
Ours 31.94 0.932 0.093 1165 K 34.11 0.925 0.072 1272 K 110.8 278 MB 2 h 30 m
Table 2: Quantitative results on Technicolor dataset. For fair comparison, 4DGS†result of using 1GB storage is reported.
DA Motion Opacity PSNR ↑LPIPS ↓Storage ↓
✓ Polynomial Ours 27.82 0.070 257 MB
✓ Linear 4DGS 27.93 0.065 195 MB
✓ Linear Ex4DGS 28.34 0.056 413 MB
Linear Ours 25.77 0.153 180 MB
✓ Linear Ours 29.57 0.050 149 MB
Table 3: Ablation results on the main components. “DA”
refers to the dynamic-aware anchor growing, “Motion” and
“Opacity” for each modeling of the neural Gaussian design.
To further explore the effects of the anchor growing strat-
egy, we provide visual comparisons in Figure 6. The pre-
vious method fails to accurately collect gradients in dy-
namic regions due to their short appearing periods. This
leads to a lack of anchors in dynamic regions, resulting in
degraded visual quality. In contrast, our method success-
fully accumulates higher gradients in these dynamic regions
(see the red-boxed parts in Figure 6). Consequently, the an-
chors generated by our method more accurately represent
dynamic regions at each timestep. These results demon-
strate that our anchor-growing strategy effectively captures
under-reconstructed dynamic regions, playing a critical role
in achieving high-quality reconstruction.
Effects of neural Gaussian design. We first evaluate our
neural design by replacing each modeling with the formu-
lation in previous works. Specifically, we replace the linear
motion with the polynomial trajectory (Li et al. 2023), and
replace our modified temporal opacity with the one used in
4DGS (Yang et al. 2024) and Ex4DGS (Lee et al. 2024a).
As shown in Table 3, our model with the proposed compo-
nents achieves the best results. Our compact parametrization
enhances the expressiveness of each Gaussian with minimal
parameters, improving both visual quality and efficiency.
Conclusion
Our framework employs a sufficient number of Gaussians to
capture complex dynamic regions, while addressing the stor-
age overhead through the anchor-based compression. The
dynamic-aware anchor growing and the neural Gaussian de-
Gradient  Rendered Image 4D AnchorsBaseline Baseline  + TA
Figure 6: Anchor growing analysis . Top-to-bottom: ac-
cumulated gradients at 5000th training iteration, final 4D
anchors at 106 frame, and the rendering results. Our an-
chor growing effectively accumulates gradients in under-
reconstructed dynamic regions.
sign lead to a substantial improvement. Experimental results
support the validity of our method, achieving state-of-the-art
visual quality and practical storage costs.
Limitations and future work. Since our method is cur-
rently designed for multi-view video datasets, applying it to
monocular videos can introduce additional challenges. Like
other methods, our method still suffers from reconstructing
elements that appear very shortly (1 or 2 frames). Resolving
this challenge can be an interesting future work.

---

## Page 8

References
Attal, B.; Huang, J.-B.; Richardt, C.; Zollhoefer, M.; Kopf,
J.; O’Toole, M.; and Kim, C. 2023. HyperReel: High-fidelity
6-DoF video with ray-conditioned sampling. In CVPR ,
16610–16620.
Bae, J.; Kim, S.; Yun, Y .; Lee, H.; Bang, G.; and Uh, Y .
2024. Per-Gaussian Embedding-Based Deformation for De-
formable 3D Gaussian Splatting. In ECCV .
Chen, A.; Xu, Z.; Zhao, F.; Zhang, X.; Xiang, F.; Yu, J.;
and Su, H. 2021. Mvsnerf: Fast generalizable radiance field
reconstruction from multi-view stereo. In ICCV , 14124–
14133.
Chen, Y .; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2025.
Hac: Hash-grid assisted context for 3d gaussian splatting
compression. In ECCV , 422–438. Springer.
Deng, T.; Chen, Y .; Zhang, L.; Yang, J.; Yuan, S.; Liu, J.;
Wang, D.; Wang, H.; and Chen, W. 2024. Compact 3d
gaussian splatting for dense visual slam. arXiv preprint
arXiv:2403.11247 .
Fan, Z.; Wang, K.; Wen, K.; Zhu, Z.; Xu, D.; and Wang,
Z. 2023. Lightgaussian: Unbounded 3d gaussian compres-
sion with 15x reduction and 200+ fps. arXiv preprint
arXiv:2311.17245 .
Girish, S.; Gupta, K.; and Shrivastava, A. 2023. Eagles: Ef-
ficient accelerated 3d gaussians with lightweight encodings.
arXiv preprint arXiv:2312.04564 .
Jiawei, X.; Zexin, F.; Jian, Y .; and Jin, X. 2024. Grid4D:
4D Decomposed Hash Encoding for High-Fidelity Dynamic
Scene Rendering. The Thirty-eighth Annual Conference on
Neural Information Processing Systems .
Kerbl, B.; Kopanas, G.; Leimk ¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian Splatting for Real-Time Radiance Field
Rendering. ACM TOG , 42(4).
Krizhevsky, A.; Sutskever, I.; and Hinton, G. E. 2012. Im-
agenet classification with deep convolutional neural net-
works. NeurIPS , 25.
Kwak, S.; Kim, J.; Jeong, J. Y .; Cheong, W.-S.; Oh, J.; and
Kim, M. 2025. MoDec-GS: Global-to-Local Motion De-
composition and Temporal Interval Adjustment for Compact
Dynamic 3D Gaussian Splatting. In CVPR .
Lee, J.; Won, C.; Jung, H.; Bae, I.; and Jeon, H.-G. 2024a.
Fully Explicit Dynamic Guassian Splatting. In NeurIPS .
Lee, J. C.; Rho, D.; Sun, X.; Ko, J. H.; and Park, E. 2024b.
Compact 3D Gaussian Representation for Radiance Field.
InCVPR , 21719–21728.
Lee, J. C.; Rho, D.; Sun, X.; Ko, J. H.; and Park, E. 2024c.
Compact 3D Gaussian Splatting for Static and Dynamic Ra-
diance Fields. arXiv preprint arXiv:2408.03822 .
Li, T.; Slavcheva, M.; Zollhoefer, M.; Green, S.; Lassner,
C.; Kim, C.; Schmidt, T.; Lovegrove, S.; Goesele, M.; New-
combe, R.; et al. 2022. Neural 3d video synthesis from
multi-view video. In CVPR .
Li, Z.; Chen, Z.; Li, Z.; and Xu, Y . 2023. Spacetime Gaus-
sian Feature Splatting for Real-Time Dynamic View Synthe-
sis.arXiv preprint arXiv:2312.16812 .Lu, T.; Yu, M.; Xu, L.; Xiangli, Y .; Wang, L.; Lin, D.; and
Dai, B. 2024a. Scaffold-gs: Structured 3d gaussians for
view-adaptive rendering. In CVPR , 20654–20664.
Lu, T.; Yu, M.; Xu, L.; Xiangli, Y .; Wang, L.; Lin, D.; and
Dai, B. 2024b. Scaffold-gs: Structured 3d gaussians for
view-adaptive rendering. In CVPR , 20654–20664.
Niedermayr, S.; Stumpfegger, J.; and Westermann, R. 2024.
Compressed 3d gaussian splatting for accelerated novel view
synthesis. In CVPR , 10349–10358.
Papantonakis, P.; Kopanas, G.; Kerbl, B.; Lanvin, A.; and
Drettakis, G. 2024. Reducing the Memory Footprint of 3D
Gaussian Splatting. Proceedings of the ACM on Computer
Graphics and Interactive Techniques , 7(1): 1–17.
Reizenstein, J.; Shapovalov, R.; Henzler, P.; Sbordone, L.;
Labatut, P.; and Novotny, D. 2021. Common Objects in 3D:
Large-Scale Learning and Evaluation of Real-life 3D Cate-
gory Reconstruction. arXiv:2109.00512.
Sabater, N.; Boisson, G.; Vandame, B.; Kerbiriou, P.; Babon,
F.; Hog, M.; Gendrot, R.; Langlois, T.; Bureller, O.; Schu-
bert, A.; et al. 2017. Dataset and pipeline for multi-view
light-field video. In Proceedings of the IEEE conference on
computer vision and pattern recognition Workshops , 30–40.
Sch¨onberger, J. L.; and Frahm, J.-M. 2016. Structure-from-
Motion Revisited. In CVPR .
Shaw, R.; Nazarczuk, M.; Song, J.; Moreau, A.; Catley-
Chandar, S.; Dhamo, H.; and P ´erez-Pellitero, E. 2024.
Swings: sliding windows for dynamic 3D gaussian splatting.
InECCV . ECCV .
T, M. V .; Wang, P.; Chen, X.; Chen, T.; Venugopalan, S.; and
Wang, Z. 2023. Is Attention All That NeRF Needs? In The
Eleventh International Conference on Learning Representa-
tions .
Wang, F.; Tan, S.; Li, X.; Tian, Z.; and Liu, H. 2022. Mixed
neural voxels for fast multi-view video synthesis. arXiv
preprint arXiv:2212.00190 .
Wang, Q.; Wang, Z.; Genova, K.; Srinivasan, P.; Zhou,
H.; Barron, J. T.; Martin-Brualla, R.; Snavely, N.; and
Funkhouser, T. 2021. IBRNet: Learning Multi-View Image-
Based Rendering. In CVPR .
Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu,
W.; Tian, Q.; and Xinggang, W. 2023. 4D Gaussian Splatting
for Real-Time Dynamic Scene Rendering. arXiv preprint
arXiv:2310.08528 .
Wu, M.; and Tuytelaars, T. 2024. Implicit gaussian splat-
ting with efficient multi-level tri-plane representation. arXiv
preprint arXiv:2408.10041 .
Xie, S.; Zhang, W.; Tang, C.; Bai, Y .; Lu, R.; Ge, S.; and
Wang, Z. 2025. MesonGS: Post-training Compression of 3D
Gaussians via Efficient Attribute Transformation. In ECCV ,
434–452. Springer.
Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y .; and
Jin, X. 2023. Deformable 3D Gaussians for High-Fidelity
Monocular Dynamic Scene Reconstruction. arXiv preprint
arXiv:2309.13101 .

---

## Page 9

Yang, Z.; Yang, H.; Pan, Z.; and Zhang, L. 2024. Real-time
Photorealistic Dynamic Scene Representation and Render-
ing with 4D Gaussian Splatting. In ICLR .
Yu, A.; Ye, V .; Tancik, M.; and Kanazawa, A. 2021. pixel-
NeRF: Neural Radiance Fields from One or Few Images. In
CVPR .
Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang,
O. 2018. The unreasonable effectiveness of deep features as
a perceptual metric. In CVPR , 586–595.

---

## Page 10

Appendix
Additional Comparison with STG
Unlike our approach, which operates from sparse initial
points, STG’s (Li et al. 2023) original setting utilizes dense
SfM point clouds from all timesteps and train the scene in
50-frame sequences. This original setting requires signifi-
cant preprocessing for COLMAP (Sch ¨onberger and Frahm
2016) calculation and less densification process during train-
ing. Table S1 and Figure S1 show a qualitative and quan-
titative comparison between STG and our method, using
the same number of iterations. While STG achieves higher
scores when operating in its original dense setting, our
method still demonstrates a superior score in visual qual-
ity. This indicates that our method successfully reconstructs
fine-grained details that were omitted by the initial SfM
points.
OursGround truthSTG (300 frames)STG (50 frames)cook_spinach
flame_salmon
Figure S1: Comparison between 50-frame STG and
Ours. The single STG model trained on 300 frames of
the N3DV dataset shows lower quality in the dynamic re-
gions compared to Ours. The white boxes highlight under-
reconstruction areas.
# models # iterations PSNR ↑LPIPS ↓
STG6 (50 frames) 6 ×2∗×30K 31.96 0.046
1 (300 frames) 2∗×60K 31.25 0.053
Ours 1 (300 frames) 120K 32.03 0.041
Table S1: Comparison with STG on the N3DV dataset.
Our model outperforms the average of six STGs trained on
50 frames and a single STG trained on 300 frames. ∗Note
that STG uses a batch size of 2 while our model uses a batch
size of 1.
Understanding Temporal Opacity
We explore the effects of the temporal opacity by observ-
ing actual distributions. Specifically, we choose an example
pixel in sear steak , which contains an object that appears
in a certain period (the frying pan). We visualize Gaussians
that primarily contribute to render this pixel. We compare
t = 20 t = 60 t = 90
generalized Gaussian univariate Gaussiant tα(t) α(t)
t
t = 20 t = 60 t = 90Figure S2: Temporal opacity analysis. We visualize Gaus-
sians contributing to the rendering of a sample pixel in the
sear steak scene from the N3DV dataset. For clarity, we fil-
ter out Gaussians distant from the frying pan based on depth
and image-plane probability Our modified temporal opac-
ity better aligns with real-world dynamics, enabling a single
Gaussian to represent the element.
our temporal opacity with the previous one based on a uni-
variate Gaussian function.
As shown in Figure S2, the resulting Gaussian with our
modified temporal opacity better fits the actual distributions
of the object. This enables our method to represent a scene
element with a single Gaussian. On the other hand, previous
univariate Gaussian is not well-aligned to real-world dynam-
ics, resulting in multiple Gaussians representing the same
element. As a result, our temporal opacity reduces the num-
ber of anchors while retaining visual quality, which leads to
compact storage usage.
Effects of Hyperparameters
K= 3 K= 5 K= 8 K= 10
PSNR 27.40 28.63 29.09 29.20
#Anchors 579K 585K 494K 476K
Table S2: Results with various K.
We investigate the impact of hyperparameters on the
performance and their controllability. cook spinach and
flame salmon scenes from N3DV are selected for analysis.
Number of Gaussians per anchor. The parameter Kde-
termines the number of Gaussians generated from an anchor
feature. We observed changes in image quality and the num-
ber of anchors as Kvaried. As Kincreased, image quality
improved while the number of anchors decreased. This is
because a larger Kallows the same spatial region to be rep-
resented with higher density.
Dynamic-aware anchor growing. The parameter γin
Equation (9) controls the influence of temporal scale on
densification. As γincreases, densification becomes more
concentrated in dynamic regions, improving reconstruction

---

## Page 11

𝛾=0.0 𝛾=1.0 𝛾=2.0flame_ salmon
(dynamic region )flame_ salmon
(static region )cook_ spinach
(dynamic region )
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013 /uni00000014/uni00000011/uni00000015 /uni00000014/uni00000011/uni00000017 /uni00000014/uni00000011/uni00000019
/uni0000002a/uni00000044/uni00000050/uni00000050/uni00000044/uni00000003/uni0000000b /uni0000000c
/uni00000017/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni00000013/uni00000018/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni00000019/uni00000013/uni00000013/uni00000013/uni00000013/uni00000013/uni00000019/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni0000001a/uni00000013/uni00000013/uni00000013/uni00000013/uni00000013/uni0000001a/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni0000001b/uni00000013/uni00000013/uni00000013/uni00000013/uni00000013/uni0000001b/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni00000006/uni00000033/uni00000052/uni0000004c/uni00000051/uni00000057/uni00000056
/uni00000049/uni0000004f/uni00000044/uni00000050/uni00000048/uni00000042/uni00000056/uni00000044/uni0000004f/uni00000050/uni00000052/uni00000051
/uni00000046/uni00000052/uni00000052/uni0000004e/uni00000042/uni00000056/uni00000053/uni0000004c/uni00000051/uni00000044/uni00000046/uni0000004bFigure S3: Effect of γ. (upper) As the value of γincreases,
the quality of dynamic regions improves, whereas the qual-
ity of static regions declines. (lower) The number of anchor
points changes with γ.
/uni00000015 /uni00000016 /uni00000017 /uni00000018 /uni00000019 /uni0000001a
/uni00000025/uni00000048/uni00000057/uni00000044/uni00000003/uni0000000b /uni0000000c
/uni00000017/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni00000013/uni00000018/uni00000018/uni00000013/uni00000013/uni00000013/uni00000013/uni00000019/uni00000013/uni00000013/uni00000013/uni00000013/uni00000013/uni00000006/uni00000033/uni00000052/uni0000004c/uni00000051/uni00000057/uni00000056
/uni00000049/uni0000004f/uni00000044/uni00000050/uni00000048/uni00000042/uni00000056/uni00000044/uni0000004f/uni00000050/uni00000052/uni00000051
/uni00000046/uni00000052/uni00000052/uni0000004e/uni00000042/uni00000056/uni00000053/uni0000004c/uni00000051/uni00000044/uni00000046/uni0000004b
Figure S4: Effect of β.The graph shows how βaffects the
number of anchor points across two scenes, with an increase
inβleading to a reduction in the number of anchor points.quality in these areas while reducing quality in static re-
gions, as demonstrated in Figure S3. In scenes with distant
background regions, such as flame salmon , larger γvalues
result in less densification in the static background, leading
to a decline in reconstruction quality. Additionally, exces-
sively high γvalues result in over-densification in dynamic
regions, increasing the number of anchors. We empirically
observe that γ= 1 provides a balanced reconstruction be-
tween static and dynamic regions.
Temporal opacity. The parameter βdetermines the steep-
ness of our generalized Gaussian curve. Higher values of
βresult in steeper slopes, enabling shapes to be represented
with fewer Gaussian components compared to a typical mix-
ture of univariate Gaussians. As shown in Figure S4, increas-
ingβreduces the number of required anchor points while
maintaining image quality. However, excessively high βval-
ues (e.g., above 8) cause instability during training.
Visualization of Anchor Growing
We visualize the anchor growing process in Figure S5. Dur-
ing the early stages of training, the temporal scale of 4D
Gaussians generated from the initial anchors increases, en-
abling rapid reconstruction of static regions. As training pro-
gresses, anchors are created in the dynamic regions at each
time step tin under-reconstructed regions, improving image
quality in these areas.
Additional Method Details
Network architecture. In this section, we describe the de-
tailed architecture of the MLPs utilized in our model. Each
anchor feature is represented as a 32-dimensional vector,
which is passed through the shared MLPs to compute the
properties of Kneural 4D Gaussians. Each MLP consists
of a 2-layer architecture with a hidden layer of width 32,
matching the size of the feature dimension. The hidden layer
uses a ReLU activation function. We employ a total of four
MLPs, referred to as the Opacity MLP, Shape MLP, Color
MLP, and Velocity MLP.
The Opacity MLP outputs the time-invariant opacity ρof
the Gaussians. The Shape MLP produces the quaternion q
and scaling sfor covariance calculation, along with the tem-
poral scale σ. The Color MLP takes the direction dconcate-
nated with the anchor feature as input and outputs the view-
dependent color c. The Velocity MLP computes the 3D ve-
locity uof the Gaussians. tanh for opacity ρ,expfor scaling
sand temporal scale σ, and sigmoid for color care applied
as activation function.
Additional implementation detail. The temporal grid size
is set to 0.0333 for the N3DV dataset and 0.02 for the Tech-
nicolor dataset. For initial points, we downsample it to fewer
than 100,000 points. All other hyperparameters, including
the learning rate and learning decay schedule, follow those
of Scaffold-GS (Lu et al. 2024a). Components such as the
feature bank, level of detail (LOD), and appearance features
are excluded.

---

## Page 12

Initial Anchort=194 t=05000 iter 20000 iter 35000 iter 50000 iter
Figure S5: Visualization of Anchor Growing. The blue dots on the left represent the initial anchor points, which are obtained
exclusively at t= 0. For a specific frame at t= 194 , new anchor points grow as iterations progress, shown as green dots on the
right. These new anchor points are primarily generated in under-reconstructed dynamic regions.
Results in Challenging Scenarios
PSNR SSIM Train time Storage
4DGS 24.28 0.833 4 h 30 m 2359 MB
Ours 28.82 0.913 2 h 25 m 114 MB
Table S3: Results with limited views.
Few shots. To evaluate the robustness of our method, we
report the results in a 4 shot scenario, in which sparse 4
views are used for reconstruction. Results in a 4 shot sce-
nario on the N3DV dataset are reported in Table S3. While
our method was not specifically designed for few-shot sce-
narios, it exhibits a smaller performance drop compared to
4DGS.
More Results
We report the quantitative results in Table S4-S5 and qual-
itative results in Figure S7-S8 of each scene of the N3DV
and the Technicolor datasets, respectively. We also provide
video quality comparisons in HTML format. Please check
theindex.html file.

---

## Page 13

FabienPainter
4DGaussians E-D3DGS4DGS
STGEx4DGS Ours GT
C3DGS Birthday
4DGaussians E-D3DGS4DGS
STGEx4DGS Ours GT
C3DGS
4DGaussians E-D3DGS4DGS
STGEx4DGS Ours GT
C3DGS
4DGaussians E-D3DGS4DGS
STGEx4DGS Ours GT
C3DGS Train
Figure S6: Qualitative comparisons on the Technicolor dataset.

---

## Page 14

Dynamic region Full region
Scene Model PSNR ↑SSIM↑LPIPS ↓#Gaussians PSNR ↑SSIM↑LPIPS ↓#Gaussians Storage ↓
coffee martini4DGaussian 23.83 0.821 0.182 - 28.44 0.919 0.060 193K 58 MB
E-D3DGS 26.05 0.901 0.096 - 29.10 0.931 0.042 263K 95 MB
Grid4D 25.76 0.874 0.139 - 28.69 0.919 0.058 153K 36 MB
STG 25.01 0.884 0.087 63K 27.45 0.912 0.070 481K 70 MB
4DGS 26.90 0.920 0.065 4229K 28.63 0.918 0.071 4293K 7978 MB
Ex4DGS 25.01 0.880 0.103 48K 28.79 0.918 0.067 340K 130 MB
Ours 27.07 0.921 0.053 447K 28.80 0.923 0.060 729K 116 MB
cook spinach4DGaussian 24.99 0.783 0.202 - 33.10 0.953 0.042 196K 58 MB
E-D3DGS 26.08 0.824 0.140 - 32.96 0.956 0.034 140K 51 MB
Grid4D 25.88 0.830 0.163 - 32.90 0.957 0.035 231K 55 MB
STG 25.02 0.797 0.195 40K 32.25 0.948 0.049 430K 62 MB
4DGS 27.32 0.879 0.088 3514K 33.54 0.956 0.039 3552K 6599 MB
Ex4DGS 26.10 0.844 0.137 60K 33.22 0.955 0.041 247K 118 MB
Ours 27.39 0.878 0.093 584K 33.36 0.955 0.036 782K 198 MB
cutroasted beef4DGaussian 27.19 0.834 0.183 - 33.12 0.954 0.044 184K 55 MB
E-D3DGS 29.22 0.891 0.123 - 33.57 0.958 0.034 143K 52 MB
Grid4D 28.74 0.886 0.133 - 33.61 0.958 0.034 229K 54 MB
STG 27.87 0.864 0.150 38K 32.73 0.951 0.048 383K 56 MB
4DGS 31.21 0.936 0.060 3010K 34.18 0.959 0.038 3047K 5661 MB
Ex4DGS 29.56 0.904 0.099 64K 33.72 0.957 0.039 249K 123 MB
Ours 31.80 0.953 0.043 745K 33.35 0.957 0.036 916K 170 MB
flame salmon4DGaussian 21.98 0.832 0.163 - 28.80 0.926 0.060 197K 58 MB
E-D3DGS 24.29 0.890 0.104 - 29.61 0.936 0.038 264K 96 MB
Grid4D 24.52 0.888 0.113 - 29.86 0.930 0.051 158K 37 MB
STG 22.09 0.848 0.120 62K 28.27 0.916 0.065 562K 81 MB
4DGS 24.43 0.900 0.078 4774K 29.25 0.929 0.057 4782K 8886 MB
Ex4DGS 22.75 0.860 0.128 49K 28.78 0.924 0.064 330K 128 MB
Ours 24.82 0.911 0.059 473K 29.20 0.928 0.054 896K 162 MB
flame steak4DGaussian 25.43 0.855 0.134 - 33.55 0.961 0.032 186K 56 MB
E-D3DGS 26.54 0.891 0.101 - 33.57 0.964 0.028 135K 50 MB
Grid4D 25.77 0.880 0.117 - 32.98 0.963 0.029 214K 50 MB
STG 25.95 0.868 0.122 31K 33.35 0.958 0.037 373K 54 MB
4DGS 26.74 0.893 0.081 2439K 33.90 0.961 0.037 2451K 4553 MB
Ex4DGS 27.03 0.905 0.077 47K 33.90 0.962 0.032 226K 101 MB
Ours 27.84 0.922 0.049 500K 33.43 0.962 0.030 734K 142 MB
sear steak4DGaussian 28.59 0.876 0.127 - 34.02 0.963 0.031 198K 59 MB
E-D3DGS 29.36 0.906 0.107 - 33.45 0.963 0.030 135K 49 MB
Grid4D 29.27 0.905 0.109 - 33.49 0.965 0.028 228K 54 MB
STG 29.09 0.901 0.087 32K 33.40 0.960 0.034 371K 54 MB
4DGS 29.31 0.913 0.082 1866K 33.37 0.960 0.040 1880K 3492 MB
Ex4DGS 30.52 0.927 0.063 41K 33.68 0.960 0.034 221K 94 MB
Ours 31.44 0.945 0.045 448K 33.52 0.960 0.030 595K 109 MB
Table S4: Quantitative results on the N3DV dataset.

---

## Page 15

Dynamic region Full region
Scene Model PSNR ↑SSIM↑LPIPS ↓#Gaussians PSNR ↑SSIM↑LPIPS ↓#Gaussians Storage ↓
Birthday4DGaussian 25.14 0.838 0.150 - 28.03 0.862 0.156 297K 79 MB
E-D3DGS 29.10 0.948 0.041 - 33.34 0.951 0.037 231K 84 MB
STG 28.98 0.950 0.037 123K 32.20 0.947 0.029 302K 44 MB
C3DGS 28.03 0.933 0.054 126K 31.37 0.938 0.039 177K 26 MB
4DGS 29.27 0.953 0.032 7233K 31.72 0.936 0.042 7366K 13684 MB
Ex4DGS 29.07 0.949 0.033 169K 32.25 0.942 0.031 460K 162 MB
Ours 29.51 0.956 0.026 395K 32.86 0.952 0.024 623K 183 MB
Fabien4DGaussian 28.54 0.807 0.283 - 33.36 0.865 0.185 224K 61 MB
E-D3DGS 30.23 0.859 0.214 - 34.45 0.875 0.147 173K 63 MB
STG 29.90 0.848 0.240 26K 34.69 0.876 0.166 72K 10 MB
C3DGS 29.54 0.836 0.260 30K 34.36 0.872 0.174 44K 6 MB
4DGS 31.44 0.892 0.170 2304K 35.02 0.884 0.144 2327K 4322 MB
Ex4DGS 31.33 0.890 0.155 116K 34.98 0.889 0.131 202K 83 MB
Ours 31.44 0.896 0.151 380K 35.15 0.895 0.124 398K 136 MB
Painter4DGaussian 27.31 0.819 0.187 - 34.52 0.899 0.138 184K 51 MB
E-D3DGS 31.60 0.916 0.124 - 36.63 0.923 0.097 206K 75 MB
STG 32.10 0.925 0.120 61K 36.66 0.923 0.097 140K 20 MB
C3DGS 30.39 0.903 0.145 61K 35.55 0.913 0.113 75K 11 MB
4DGS 33.51 0.946 0.097 4945K 35.71 0.923 0.104 5015K 9316 MB
Ex4DGS 33.33 0.945 0.096 151K 36.62 0.932 0.091 255K 106 MB
Ours 35.19 0.960 0.075 1729K 37.05 0.939 0.073 1834K 465 MB
Train4DGaussian 17.11 0.470 0.398 - 23.54 0.756 0.206 332K 88 MB
E-D3DGS 28.48 0.914 0.068 - 31.84 0.922 0.074 216K 78 MB
STG 26.56 0.896 0.110 101K 32.20 0.935 0.044 314K 46 MB
C3DGS 25.23 0.855 0.174 138K 30.79 0.910 0.071 193K 28 MB
4DGS 33.09 0.956 0.034 6468K 32.15 0.930 0.047 6980K 12967 MB
Ex4DGS 29.97 0.943 0.058 48K 31.40 0.927 0.055 816K 216 MB
Ours 32.42 0.964 0.026 2339K 32.86 0.946 0.032 2433K 275 MB
Theater4DGaussian 21.86 0.612 0.381 - 28.67 0.835 0.195 302K 80 MB
E-D3DGS 27.56 0.794 0.291 - 30.64 0.864 0.147 237K 86 MB
STG 24.19 0.727 0.360 45K 30.92 0.878 0.143 188K 27 MB
C3DGS 22.85 0.690 0.411 81K 30.54 0.872 0.155 123K 18 MB
4DGS 32.25 0.904 0.132 7036K 31.90 0.877 0.136 7110K 13208 MB
Ex4DGS 29.21 0.862 0.204 119K 31.74 0.879 0.131 399K 132 MB
Ours 31.12 0.885 0.188 980K 32.65 0.893 0.106 1073K 332 MB
Table S5: Quantitative results on the Technicolor dataset.

---

## Page 16

Ours
Ground truthSTGEx4DGS
Ex4DGS
Ex4DGS
Ex4DGS4DGS
E-D3DGSGrid4D
Grid4D
Grid4D
Grid4D4DGaussianscook_spinach
Ours
Ground truthSTG4DGS
E-D3DGS4DGaussianssear_steak
Ours
Ground truthSTG4DGS
E-D3DGS4DGaussiansflame_salmon
Ours
Ground truthSTG4DGS
E-D3DGS4DGaussianscof fee_martini
Figure S7: Qualitative comparisons on the N3DV dataset with full region.

---

## Page 17

OursSTG4DGSEx4DGS
Ex4DGS
Ex4DGSEx4DGSPainter
OursSTG4DGSTheater
OursSTG4DGSFabien
OursSTG4DGST rainGround truthE-D3DGS4DGaussians
Ground truthE-D3DGS4DGaussians
Ground truthE-D3DGS4DGaussians
Ground truthE-D3DGSC3DGS
C3DGS
C3DGS
C3DGS4DGaussians
Figure S8: Qualitative comparisons on the Technicolor dataset with full region.