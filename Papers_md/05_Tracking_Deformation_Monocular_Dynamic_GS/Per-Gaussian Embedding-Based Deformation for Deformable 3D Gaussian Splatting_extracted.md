

---

## Page 1

Per-Gaussian Embedding-Based Deformation for
Deformable 3D Gaussian Splatting
Jeongmin Bae1⋆, Seoha Kim1⋆, Youngsik Yun1,
Hahyun Lee2, Gun Bang2, and Youngjung Uh1
1Yonsei University, Seoul 03722, Korea
{jaymin.bae, hailey07, bbangsik, yj.uh}@yonsei.ac.kr
2Electronics and Telecommunications Research Institute, Daejeon 34129, Korea
{hanilee, gbang}@etri.re.kr
(b) Ours
 + fine resolution grid
 (a) 4DGaussians (CVPR 24)
flame_salmon_frag3
Fig.1: Overview. (a) Existing deformable 3D Gaussian Splatting methods show
blurry results in complex dynamic scenes, even with deformation fields using finer
feature grids. (b) Our model solves the problem by employing per-Gaussian latent em-
beddings to predict deformations for each Gaussian and achieves clearer results.
Abstract. As 3D Gaussian Splatting (3DGS) provides fast and high-
quality novel view synthesis, it is a natural extension to deform a canoni-
cal 3DGS to multiple frames for representing a dynamic scene. However,
previous works fail to accurately reconstruct complex dynamic scenes.
We attribute the failure to the design of the deformation field, which is
built as a coordinate-based function. This approach is problematic be-
cause 3DGS is a mixture of multiple fields centered at the Gaussians,
not just a single coordinate-based framework. To resolve this problem,
we define the deformation as a function of per-Gaussian embeddings and
temporal embeddings. Moreover, we decompose deformations as coarse
and fine deformations to model slow and fast movements, respectively.
Also, we introduce a local smoothness regularization for per-Gaussian
embedding to improve the details in dynamic regions.
Project page: https://jeongminb.github.io/e-d3dgs/
Keywords: Deformable gaussian splatting ·4D scene reconstruction ·
Novel view synthesis
⋆Authors contributed equally to this work.arXiv:2404.03613v5  [cs.CV]  26 Jul 2024

---

## Page 2

2 J. Bae et al.
1 Introduction
Dynamic scene reconstruction from multi-view input videos is an important task
in computer vision, as it can be extended to various applications and industries
such as mixed reality, content production, etc. Neural Radiance Fields (NeRF)
[19], which enable photorealistic novel view synthesis from multi-view inputs,
can represent dynamic scenes by modeling the scene with an additional time
input [12,22]. However, typical NeRFs require querying multilayer perceptron
(MLP) for hundreds of points per camera ray, which limits rendering speed.
On the other hand, the recently emerging 3D Gaussian Splatting (3DGS) [10]
has the advantage of real-time rendering compared to NeRFs using a differen-
tiable rasterizer for 3D Gaussian primitives. 3DGS directly optimizes the param-
eters of 3D Gaussians (position, opacity, anisotropic covariance, and spherical
harmonics coefficients) and renders them via projection and α-blending. Since
3DGS has the characteristics of continuous volumetric radiance fields, some re-
cent studies [3,8,14,29,32,33] represent dynamic scenes by defining a canonical
3DGSanddeformingittoindividualframesasdeformableNeRFs[32]do.Specif-
ically, they model the deformation as a function of 4D (x, y, z, t) coordinates
with MLPs or grids to predict the change in the 3D Gaussian parameters.
However, since 3DGS is a mixture of multiple volumetric fields, it is not
appropriate to model the deformation of Gaussian parameters with a single
coordinate-basednetworktorepresentdynamicscenes.Inaddition,existingfield-
based approaches are constrained by the resolution of the grid which models the
deformation field, the capacity of the model, or the frequencies of the input.
As shown in Figure 1, the existing study does not properly represent complex
dynamic scenes, and even introducing an additional feature grid that is twice
the maximum resolution has only a slight improvement in performance (See
Appendix for more results). We alleviate this problem by introducing a novel
dynamic representation to deform each Gaussian.
In this paper, we model the deformation of Gaussians at frames as 1) a func-
tion of a product space of per-Gaussian embeddings and temporal embeddings.
We expect this rational design to bring quality improvement by precisely model-
ing different deformations of different Gaussians. Additionally, 2) We decompose
temporal variations of the parameters into coarse and fine components, namely
coarse-fine deformation. The coarse deformation represents large or slow move-
ments in the scene, while fine deformation learns the fast or detailed movements
that coarse deformation does not cover. Finally, we propose 3) a local smooth-
ness regularization for per-Gaussian embedding to ensure the deformations of
neighboring Gaussians are similar.
In our experiments, we observe that our per-Gaussian embeddings, coarse-
fine deformation, and regularization improve the deformation quality. Our ap-
proach outperforms baselines in capturing fine details in dynamic regions and
excels even under challenging camera settings. Additionally, our method also
achieves fast rendering speed and relatively low capacity.

---

## Page 3

E-D3DGS 3
2 Related Work
In this section, we review methods for dynamic scene reconstruction that deform
3D canonical space and methods for reconstructing dynamic scenes utilizing
dynamic 3D Gaussians. Afterward, we review methods that use embeddings and
spatial relationships of Gaussians.
Deforming 3D Canonical Space D-NeRF [22] reconstructs dynamic scenes
by deforming ray samples over time, using the deformation network that takes
3D coordinates and timestamps of the sample as inputs. Nerfies [20] and Hyper-
NeRF [21] use per-frame trainable deformation codes instead of time conditions
to deform the canonical space. Instead of deforming from the canonical frame
to the entire frames, HyperReel [1] deforms the ray sample of the keyframe to
represent the intermediate frame. 4DGaussians [29] and D3DGS [32] reconstruct
the dynamic scene with a deformation network which inputs the center posi-
tion of the canonical 3D Gaussians and timestamps. MoDGS [15] learns the
mapping between canonical space and space at a specific timestamp through in-
vertible MLP. In contrast, we demonstrate a novel deformation representation as
a function of a product space of per-Gaussian latent embeddings and temporal
embeddings.
Dynamic 3D Gaussians To extend the fast rendering speed of 3D Gaussian
Splatting [10] into dynamic scene reconstructions. 4DGaussians [29] decodes fea-
tures from multi-resolution HexPlanes [2] for temporal deformation of 3D Gaus-
sians. While D3DGS [32] uses an implicit function that processes the time and
location of the Gaussian. 4DGS [30] decomposes the 4D Gaussians into a time-
conditioned 3D Gaussians and a marginal 1D Gaussians. STG [13] represents
changes in 3D Gaussian over time through a temporal opacity and a polynomial
function for each Gaussian.
Our method uses deformable 3D Gaussians as 4DGaussians [29] and D3DGS
[32] do, but does not necessitate the separated feature field to obtain the input
feature of the deformation decoder. Our approach uses embeddings allocated to
each Gaussian and a temporal embedding shared within a specific frame.
Latent Embedding on Novel View Synthesis Some studies incorporate
latent embeddings to represent different states of the static and dynamic scene.
NeRF-W [18] and Block-NeRF [25] employ per-image embeddings to capture
different appearances of a scene, representing the scenes from in-the-wild image
collections. DyNeRF and MixVoxels [12,27] employ a temporal embedding for
each frame to represent dynamic scenes. Nerfies [20] and HyperNeRF [21] incor-
poratebothper-frameappearanceanddeformationembeddings.Sync-NeRF[11]
introduces time offset to calibrate the misaligned temporal embeddings on dy-
namic scenes from unsynchronized videos. We introduce per-Gaussian latent
embedding to encode the changes over time of each Gaussian and use temporal
embeddings to represent different states in each frame of the scene.

---

## Page 4

4 J. Bae et al.
Considering Spatial Relationships of Gaussians Scaffold-GS [16] recon-
structs 3D scenes by synthesizing Gaussians from anchors, utilizing that the
neighboring Gaussians have similar properties. SAGS [26] creates a graph based
on k-nearest neighbors (KNN) so that each Gaussian is optimized while consid-
ering its neighboring Gaussians. In dynamic scene reconstruction, SC-GS [9] and
GaussianPrediction [35] deform Gaussians by combining the deformations of key
point Gaussians. Dynamic 3D Gaussians [17] utilizes regularization to encourage
that Gaussians and their neighboring Gaussians deform with local rigidity. Sim-
ilarly, we propose a local smoothness regularization that encourages neighboring
Gaussians to have similar embeddings, resulting in similar deformations.
3 Method
Inthissection,wefirstprovideabriefoverviewof3DGaussianSplatting(Section
3.1). Next, we introduce our overall framework, embedding-based deformation
for Gaussians (Section 3.2) and coarse-fine deformation scheme consisting of
coarse and fine deformation functions (Section 3.3). Finally, we present a local
smoothness regularization for per-Gaussian embeddings to achieve better details
on dynamic regions (Section 3.4).
3.1 Preliminary: 3D Gaussian Splatting
3D Gaussian splatting [10] optimizes a set of anisotropic 3D Gaussians through
differentiable tile rasterization to reconstruct a static 3D scene. By its efficient
rasterization, the optimized model enables real-time rendering of high-quality
images. Each 3D Gaussian kernel Gi(x)at the point xconsist with position xi,
rotation Ri, and scale Si:
Gi(x) =e−1
2(x−xi)TΣ−1
i(x−xi),where Σi=RiSiST
iRT
i.(1)
To projecting 3D Gaussians to 2D for rendering, covariance matrix Σ′are cal-
culated by viewing transform Wand the Jacobian Jof the affine approximation
of the projective transfomation [36] as follows:
Σ′=JWΣWTJT. (2)
Blending Ndepth-ordered projected points that overlap the pixel, the Gaussian
kernel Gi(x)is multiplied by the opacity of the Gaussian σiand calculates the
pixel color Cwith the color of the Gaussian ci:
C=X
i∈Nciαii−1Y
j=1(1−αj),where αi=σiGi(x). (3)
The color of Gaussian ciis determined using the SH coefficient with the viewing
direction.

---

## Page 5

E-D3DGS 5
(c) Local smoothness regularization
deformed Gaussian
fine 
decodercoarse 
decoder
3D Gaussian
(b) Per -Gaussian deformation
+
+
(a) Embeddings for deformation
Coarse-Fine temporal embedding
time axis downsamplePer -Gaussian embedding
Fig.2: Framework. Existing coordinate-based network methods struggle to represent
complex dynamic scenes. To this end, we define per-Gaussian deformation. (a) Firstly,
we assign a latent embedding for each Gaussian. Additionally, we introduce coarse and
fine temporal embeddings to represent the slow and fast state of the dynamic scene.
(b) By employing two decoders that take per-Gaussian latent embeddings along with
coarse and fine temporal embeddings as input, we estimate slow or large changes and
fast or detailed changes to model the final deformation, respectively. (c) Finally, we
introduce a local smoothness regularization so that the embeddings of neighboring
Gaussians are similar.
3.2 Embedding-Based Deformation for Gaussians
Deformable NeRFs consist of a deformation field that predicts displacement ∆x
for a given coordinate xfrom the canonical space to each target frame, and a
radiance field that maps color and density from a given coordinate in the canon-
ical space ( x+∆x). Existing deformable Gaussian methods employ the same
approach for predicting the deformation of Gaussians, i.e., utilizing a deforma-
tion field based on coordinates.
Unlike previous methods, we start from the design of 3DGS: the 3D scene
is represented as a mixture of Gaussians that have individual radiance fields.
Accordingly, the deformation should be defined for each Gaussian. Based on this
intuition, we introduce a function Fθthat produces deformation from learnable
embeddings zg∈R32belonging to individual Gaussians (Figure 2a), and typical
temporal embeddings zt∈R256for different frames:
Fθ: (zg,zt)→(∆x, ∆r, ∆s, ∆σ, ∆Y ), (4)
where ris a rotation quaternion, sis a vector for scaling, σis an opacity, and Yis
SHcoefficientsformodelingview-dependentcolor.Weimplement Fθasashallow
multi-layer perceptron (MLP) followed by an MLP head for each parameter. As
a result, the Gaussian parameters at frame tare determined by adding Fθ(zg,zt)
to the canonical Gaussian parameters (Figure 2c).
We jointly optimize the per-Gaussian embeddings zg, the deformation func-
tionFθ, and the canonical Gaussian parameters to minimize the rendering loss.
We use the L1 and periodic DSSIM as the rendering loss between the rendered
image and the ground truth image.

---

## Page 6

6 J. Bae et al.
3.3 Coarse-Fine Deformation
Different parts of a scene may have coarse and fine motions [5]. E.g., a hand
swiftly stirs a pan (fine) while a body slowly moves from left to right (coarse).
Based on this intuition, we introduce a coarse-fine deformation that produces a
summation of coarse and fine deformations.
Coarse-fine deformation consists of two functions with the same architecture:
one for coarse and one for fine deformation (Figure 2c). The functions receive
different temporal embeddings as follows:
Following typical temporal embeddings, we start from a 1D feature grid
Z∈RN×256forNframes and use an embedding zf
t=interp (Z, t)for fine
deformation. For coarse deformation, we linearly downsample Zby a factor of 5
to remove high-frequencies responsible for fast and detailed deformation. Then
we compute zc
tas a linear interpolation of embeddings at enclosing grid points
(Figure 2b).
As a result, coarse deformation Fθc(zg,zc
t) is responsible for representing
large or slow movements in the scene, while fine deformation Fθf(zg,zf
t) learns
the fast or detailed movements that coarse deformation does not cover. This
improves the deformation quality. Refer to the Ablation study section for more
details.
3.4 Local Smoothness Regularization
Neighboring Gaussians constructing dynamic objects tend to exhibit locally sim-
ilar deformation. Inspired by [17], we introduce a local smoothness regularization
for per-Gaussian embedding zg(Figure 2d) to encourage similar deformations
between nearby Gaussians iandj:
Lemb_reg=1
k|S|X
i∈SX
j∈KNN i;k(wi,j∥zgi−zgj∥2),
where wi,j= exp( −λw∥µj−µi∥2
2)is the weighting factor and µis the Gaussian
center.Weset λwto2000and kto20following[17].Toreducethecomputational
cost, we obtain sets of k-nearest-neighbors only when the densification occurs.
Note that unlike previous approaches that directly constrain physical proper-
ties such as rigidity or rotation, We implicitly induce locally similar deformation
by ensuring that per-Gaussian embeddings are locally smooth. Our regulariza-
tion allows better capture of textures and details of dynamic objects.
4 Experiment
In this section, we first describe the criterion for selection of baselines, and
evaluationmetrics.Wethendemonstratetheeffectivenessofourmethodthrough
comparisons with various baselines and datasets (Section 4.1-4.2). Finally, we
conduct analysis and ablations of our method (Section 4.3).

---

## Page 7

E-D3DGS 7
Table 1: Average performance in the test view on Neural 3D Video dataset
The computational cost was measured based on flame_salmon_1 on the A6000.1flame
salmon scene includes only the first segment, comprising 300 frames.2reported time
and DyNeRF is trained on 8 GPUs and tested only on flame salmon.3trained with 90
frames.4trained with 50 frames.
modelmetric computational cost
PSNR↑SSIM↑LPIPS↓Training time ↓FPS↑Model size ↓
DyNeRF12[12] 29.58 - 0.083 1344 hours 0.01 56 MB
NeRFPlayer13[24]30.69 - 0.111 6 hours 0.05 1654 MB
MixVoxels [27] 30.30 0.918 0.127 1 hours 40 mins 0.93 512 MB
K-Planes [6] 30.86 0.939 0.096 1 hours 40 mins 0.13 309 MB
HyperReel4[1]30.37 0.921 0.106 9 hours 20 mins 1.04 1362 MB
4DGS [30] 31.19 0.940 0.051 9 hours 30 mins 33.7 8700 MB
4DGaussians [29] 30.71 0.935 0.056 50 mins 51.9 59 MB
Ours 31.31 0.945 0.037 1 hours 52 mins 74.5 35 MB
Baselines We choose the state-of-the-art method as a baseline in each dataset.
We compared against DyNeRF, NeRFPlayer, MixVoxels, K-Planes, HyperReel,
Nerfies, HyperNeRF, and TiNeuVox on the NeRF baseline. In detail, we use the
version of NeRFPlayer TensoRF VM, HyperNeRF DF, Mixvoxels-L, K-Planes
hybrid. We compared with 4DGaussians, 4DGS, and D3DGS based on the Gaus-
sian baseline. Meanwhile, we have not included STG in our comparison due to
its requirement for per-frame Structure from Motion (SfM) points, which makes
conducting a fair comparison challenging. Also, STG is not a deformable 3D
Gaussian approach. We followed the official code and configuration, except for
increasing the training iterations for the Technicolor dataset to 1.5 times that of
the 4DGaussians, in comparison to the Neural 3D Video dataset.
Metrics We report the quality of rendered images using PSNR, SSIM, and
LPIPS. Peak Signal-to-Noise Ratio (PSNR) quantifies pixel color error between
the rendered video and the ground truth. We utilize SSIM [28] to account for the
perceived similarity of the rendered image. Additionally, we measure higher-level
perceptual similarity using Learned Perceptual Image Patch Similarity (LPIPS)
[34]withanAlexNetBackbone.HigherPSNRandSSIMvaluesandlowerLPIPS
values indicate better visual quality.
4.1 Effectiveness on Dynamic Region
Neural 3D Video Dataset [12] includes 20 multi-view videos, with each scene
consisting of either 300 frames, except for the flame_salmon scene, which com-
prises 1200 frames. These scenes encompass a relatively long duration and var-
ious movements, with some featuring multiple objects in motion. We utilized
the Neural 3D Video dataset to observe the capability to capture dynamic ar-
eas. Total six scenes ( coffee_martini, cook_spinach, cut_roasted_beef,
flame_salmon, flame_steak, sear_steak ) are evaluated in Figure 3 and Ta-
ble 1. The flame_salmon scene is divided into four segments, each containing
300 frames.

---

## Page 8

8 J. Bae et al.
Ours 4DGS 4DGaussians K-Planes MixV oxels
cof fee_martini
 flame_salmon_frag1
 sear_steak
 cook_spinach
Fig.3: Qualitative comparisons on the Neural 3D Video dataset.
Table 1 presents quantitative metrics on the average metrics across the test
views of all scenes, computational and storage costs on the first fragment of
flame_salmon scene. Refer to the Appendix for per-scene details. Our method
demonstrates superior reconstruction quality, FPS, and model size across com-
pared to baselines. As the table shows, NeRF baselines generally required longer
training and rendering times. While 4DGS shows relatively high reconstruction
performance, it demands longer training times and larger VRAM storage capac-
ity compared to other baselines. 4DGaussians requires lower computational and
storage costs but it displays low reconstruction quality in some scenes with rapid
dynamics, as shown in the teaser and Figure 3.
Figure 3 reports the rendering quality. Our method successfully reconstructs
the fine details in moving areas, outperforming baselines on average metrics
across test views. Baselines show blurred dynamic areas or severe artifacts in
low-light scenes such as cook_spinach and flame_steak . 4DGS exhibits the
disappearance of some static areas. In 4DGaussians, a consistent over-smoothing
occurs in dynamic areas. All baselines experienced reduced quality in reflective
or thin dynamic areas like clamps, torches, and windows.

---

## Page 9

E-D3DGS 9
Table 2: Average performance in the test view on Technicolor dataset
modelmetric computational cost
PSNR↑SSIM↑LPIPS↓Training time ↓FPS↑Model size ↓
DyNeRF 31.80 - 0.140 - 0.02 0.6 MB
HyperReel 32.32 0.899 0.118 2 hours 45 mins 0.91 289 MB
4DGaussians 29.62 0.844 0.176 25 mins 34.8 51 MB
Ours 33.24 0.907 0.100 2 hours 55 mins 60.8 77 MB
Ours 4DGaussians HyperReel
Birthday
 Painter
 Fabien
Fig.4: Qualitative comparisons on the Technicolor dataset.
Technicolor Light Field Dataset [23] is a multi-view dataset captured with
a time-synchronized 4×4camera rig, containing intricate details. We train ours
and the baselines on 50 frames of five commonly used scenes ( Birthday ,Fabien,
Painter,Theater,Trains) using full-resolution videos at 2048×1088pixels,
with the second row and second column cameras used as test views.
Table 2 reports the average metrics across the test views of all scenes, com-
putational and storage costs on Painter scene. HyperReel demonstrates overall
high-quality results but struggles with relatively slow training times and FPS,
and a larger model size. 4DGaussians exhibits fast training times and FPS but
significantlyunderperformsinreconstructingfinedetailscomparedtootherbase-
lines. However, our method demonstrates superior reconstruction quality and
faster FPS compared to the baselines.
As shown in Figure 4, HyperReel produces noisy artifacts due to incorrect
predictions of the displacement vector. 4DGaussians fails to capture fine details
in dynamic areas, exhibiting over-smoothing results. All baselines struggle to
accurately reconstruct rapidly moving thin areas like fingers.

---

## Page 10

10 J. Bae et al.
4.2 Challenging Camera Setting
Table 3: Average performance in the test view on Hypernerf dataset
modelmetric computational cost
PSNR↑SSIM↑LPIPS↓Training time ↓FPS↑Model size ↓
Nerfies [20] 22.23 - 0.170 ∼hours <1 -
HyperNeRF DS [21] 22.29 0.598 0.153 32 hours <1 15 MB
TiNeuVox [4] 24.20 0.616 0.393 30 mins 1 48 MB
D3DGS [32] 22.40 0.598 0.275 3 hours 30 mins 6.95 309 MB
4DGaussians 25.03 0.682 0.281 16 mins 96.3 60 MB
Ours 25.43 0.697 0.2311 hours 15 mins 139.3 33 MB
Ours 4DGaussians D3DGS T iNeuV ox
Broom
 Chicken
 Banana
Fig.5: Qualitative comparisons on the HyperNeRF dataset.
HyperNeRFDataset includesvideoscapturedusingtwophonesrigidlymounted
on a handheld rig. We train on all frames of four scenes ( 3D Printer, Banana,
Broom, Chicken ) at a resolution downsampled by half to 536×960. Due to
memory constraints, D3DGS is trained on images downsampled by a quarter.
The table shows that our method outperforms the reconstruction perfor-
mance with previous methods along with compact model size and faster FPS.
Figure 5 shows that previous methods struggle to reconstruct fast-moving parts
such as fingers and broom. Especially D3DGS deteriorates in Broomscene. Table
3 reports the average metrics across the test views of all scenes, computational
and storage costs on Broomscene.

---

## Page 11

E-D3DGS 11
4.3 Analyses and Ablation Study
(d) full rendering (c) fine deform only  (b) coarse deform only (a) canonical rendering
Fig.6: Deformation components. (a) The canonical space contains Gaussians to
represent all target frames of the scene. (b) Applying coarse deformation to the canon-
ical space roughly reflects the dynamics of the scene. (c) The rendering without coarse
deformation and only with fine deformation looks similar to the canonical rendering,
i.e., responsible for fine deformations. (d) Applying both coarse and fine deformation
yields natural rendering results.
Frame 185 Frame 67slowfastslow
fast
Fig.7: Visualization of the magnitude of deformation. Coarse deformation
(blue) captures large and slow changes, such as the movement of the head and torso,
while fine deformation ( red) is responsible for the fast and detailed movements of arms,
tongs, shadows, etc.
Deformation components In Figure 6, we present an analysis of the coarse-
fine deformation. To achieve this, we render a flame_steak scene by omitting
each of our deformation components one by one. Our full rendering results from
adding coarse and fine deformation to the canonical space (Figure 6d). When
both are removed, rendering yields an image in canonical space (Figure 6a).
Rendering with the coarse deformation, which handles large or slow changes in
the scene, produces results similar to the full rendering (Figure 6b). On the other
hand, fine deformation is responsible for fast or detailed changes in the scene,
yielding rendering similar to canonical space (Figure 6c).

---

## Page 12

12 J. Bae et al.
(d) Ours +     injection (c) fine deform only (b) coarse deform only (a) Ours Full
Fig.8: Qualitative ablation results on coarse-fine deformation. (a) Our model
achieves clear results with both coarse and fine decoders. (b-c) The quality of dynamic
areas decreases if one is missing. (d) Additionally, introducing the coordinates of Gaus-
sian as an additional input into our decoders results in a decrease in the quality of both
static and dynamic regions.
Table 4: Quantitative ablation results on coarse-fine deformation.
Method PSNR↑SSIM↑LPIPS↓
Ours 29.70 0.933 0.041
coarse deformation only 29.48 0.931 0.044
fine deformation only 29.23 0.932 0.043
Ours + xinjection 29.60 0.931 0.045
To examine the roles of the coarse and fine deformation in the coarse-fine de-
formation, we conduct a visualization on flame_steak scene. First, we compute
the Euclidean norm of positional shifts between the current and subsequent
frames. We then add the value to the DC components of the SH coefficients
proportionally to the magnitude: blue for coarse deformation and red for fine
deformation. For visual clarity, we render the original scene in grayscale. As il-
lustrated in Figure 7, coarse deformation models slower changes such as body
movement, while fine deformation models faster movements like cooking arms.
Thus, we demonstrate that by downsampling the temporal embedding grid Z,
we can effectively separate and model slow and fast deformations in the scene.
Ablation study We report the results of an ablation study on the deformation
decoder in Figure 8 and Table 4. First, our full method (using both coarse and
fine decoders) produces clear rendering results and models dynamic variations
well (Figure 8a). Training only with the coarse or fine decoder leads to blurred
dynamic areas and a failure to accurately capture detailed motion (Figure 8b-c).
Additionally,wedemonstrateexperimentswheretheGaussiancentercoordinates
xare injected into the input of each decoder. As shown in Figure 8d, including
the Gaussian coordinates degrades the quality of deformation, supporting our
argument that coordinate dependency should be removed from the deformation
function.
Furthermore, we report the results of an ablation study on the local smooth-
ness regularization for per-Gaussian embeddings. As shown in Figure 9, our reg-
ularization improves the details and texture quality of moving objects. In Table
5, we show a performance comparison between the proposed regularization and

---

## Page 13

E-D3DGS 13
Table 5: Quantitative ablation results on local smoothness regularization.
We compare the performance of applying our regularization with the physically-based
regularizationofDynamic3DGaussians[17].Ourregularizationbettercapturesdetails
of dynamic objects.
Method PSNR↑SSIM↑LPIPS↓
Ours w/o embedding reg 32.26 0.951 0.037
+ our embedding reg 32.34 0.952 0.036
+ physically-based reg 32.08 0.950 0.036
Ours w/o regOurs full
Ours w/o reg
Ours full
Fig.9: Qualitative ablation results on local smoothness regularization.
the existing physically-based regularizations. To apply the method proposed by
Luiten et al. [17] to ours, we make some modifications: 1) Like our method, we
find the set of k-nearest neighbors only when the densification occurs to reduce
the computational cost. 2) For long-term local-isometry loss, we use the time
of the video frame used in the previous training step instead of using the time
of the first frame. Our regularization is simple and shows better performance
compared to the previous method.
5 Conclusion and Limitation
We propose a per-Gaussian deformation for 3DGS that takes per-Gaussian em-
beddings as input, instead of using the typical deformation fields from previous
deformable 3DGS works, resulting in high performance. We enhance the re-
construction quality by decomposing the dynamic changes into coarse and fine
deformation. However, our method learns inappropriate Gaussian deformation
with casually captured monocular videos [7], like other baselines. We plan to ad-
dress it in future work by introducing useful prior for monocular video settings.
Ours
 4DGaussians
 Ours
 4DGaussians
Fig.10: Limitation. Ours struggles with the casually captured monocular videos.

---

## Page 14

14 J. Bae et al.
Acknowledgements
This work is supported by the Institute for Information & Communications
Technology Planning & Evaluation (IITP) grant funded by the Korea govern-
ment(MSIT) (No. 2017-0-00072, Development of Audio/Video Coding and Light
Field Media Fundamental Technologies for Ultra Realistic Tera-media)
References
1. Attal, B., Huang, J.B., Richardt, C., Zollhoefer, M., Kopf, J., OâĂŹToole, M.,
Kim, C.: Hyperreel: High-fidelity 6-dof video with ray-conditioned sampling. In:
ProceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecog-
nition. pp. 16610–16620 (2023)
2. Cao, A., Johnson, J.: Hexplane: A fast representation for dynamic scenes. In: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion. pp. 130–141 (2023)
3. Duisterhof,B.P.,Mandi,Z.,Yao,Y.,Liu,J.W.,Shou,M.Z.,Song,S.,Ichnowski,J.:
Md-splatting: Learning metric deformation from 4d gaussians in highly deformable
scenes. arXiv preprint arXiv:2312.00583 (2023)
4. Fang, J., Yi, T., Wang, X., Xie, L., Zhang, X., Liu, W., Nießner, M., Tian, Q.:
Fast dynamic radiance fields with time-aware neural voxels. In: SIGGRAPH Asia
2022 Conference Papers (2022)
5. Feichtenhofer, C., Fan, H., Malik, J., He, K.: Slowfast networks for video recog-
nition. In: Proceedings of the IEEE/CVF international conference on computer
vision. pp. 6202–6211 (2019)
6. Fridovich-Keil, S., Meanti, G., Warburg, F.R., Recht, B., Kanazawa, A.: K-planes:
Explicit radiance fields in space, time, and appearance. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12479–
12488 (2023)
7. Gao, H., Li, R., Tulsiani, S., Russell, B., Kanazawa, A.: Monocular dynamic view
synthesis: A reality check. In: NeurIPS (2022)
8. Huang, Y.H., Sun, Y.T., Yang, Z., Lyu, X., Cao, Y.P., Qi, X.: Sc-gs:
Sparse-controlled gaussian splatting for editable dynamic scenes. arXiv preprint
arXiv:2312.14937 (2023)
9. Huang, Y.H., Sun, Y.T., Yang, Z., Lyu, X., Cao, Y.P., Qi, X.: Sc-gs: Sparse-
controlled gaussian splatting for editable dynamic scenes. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
pp. 4220–4230 (June 2024)
10. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July
2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
11. Kim, S., Bae, J., Yun, Y., Lee, H., Bang, G., Uh, Y.: Sync-nerf: Generalizing
dynamic nerfs to unsynchronized videos. arXiv preprint arXiv:2310.13356 (2023)
12. Li, T., Slavcheva, M., Zollhoefer, M., Green, S., Lassner, C., Kim, C., Schmidt,
T., Lovegrove, S., Goesele, M., Newcombe, R., et al.: Neural 3d video synthesis
from multi-view video. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (2022)
13. Li, Z., Chen, Z., Li, Z., Xu, Y.: Spacetime gaussian feature splatting for real-time
dynamic view synthesis. arXiv preprint arXiv:2312.16812 (2023)

---

## Page 15

E-D3DGS 15
14. Liang, Y., Khan, N., Li, Z., Nguyen-Phuoc, T., Lanman, D., Tompkin, J., Xiao,
L.: Gaufre: Gaussian deformation fields for real-time dynamic novel view synthesis
(2023)
15. Liu, Q., Liu, Y., Wang, J., Lv, X., Wang, P., Wang, W., Hou, J.: Modgs: Dynamic
gaussian splatting from causually-captured monocular videos (2024), https://
arxiv.org/abs/2406.00434
16. Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs: Struc-
tured 3d gaussians for view-adaptive rendering. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 20654–20664 (2024)
17. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking
by persistent dynamic view synthesis. In: 3DV (2024)
18. Martin-Brualla, R., Radwan, N., Sajjadi, M.S.M., Barron, J.T., Dosovitskiy, A.,
Duckworth, D.: NeRF in the Wild: Neural Radiance Fields for Unconstrained
Photo Collections. In: CVPR (2021)
19. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Commu-
nications of the ACM 65(1), 99–106 (2021)
20. Park, K., Sinha, U., Barron, J.T., Bouaziz, S., Goldman, D.B., Seitz, S.M., Martin-
Brualla, R.: Nerfies: Deformable neural radiance fields. ICCV (2021)
21. Park,K.,Sinha,U.,Hedman,P.,Barron,J.T.,Bouaziz,S.,Goldman,D.B.,Martin-
Brualla, R., Seitz, S.M.: Hypernerf: A higher-dimensional representation for topo-
logically varying neural radiance fields. arXiv preprint arXiv:2106.13228 (2021)
22. Pumarola, A., Corona, E., Pons-Moll, G., Moreno-Noguer, F.: D-nerf: Neural ra-
diance fields for dynamic scenes. arXiv preprint arXiv:2011.13961 (2020)
23. Sabater,N.,Boisson,G.,Vandame,B.,Kerbiriou,P.,Babon,F.,Hog,M.,Gendrot,
R., Langlois, T., Bureller, O., Schubert, A., et al.: Dataset and pipeline for multi-
view light-field video. In: Proceedings of the IEEE conference on computer vision
and pattern recognition Workshops. pp. 30–40 (2017)
24. Song, L., Chen, A., Li, Z., Chen, Z., Chen, L., Yuan, J., Xu, Y., Geiger, A.: Nerf-
player: A streamable dynamic scene representation with decomposed neural radi-
ance fields. IEEE Transactions on Visualization and Computer Graphics 29(5),
2732–2742 (2023)
25. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Bar-
ron, J.T., Kretzschmar, H.: Block-nerf: Scalable large scene neural view synthesis
(2022)
26. Ververas, E., Potamias, R.A., Song, J., Deng, J., Zafeiriou, S.: Sags: Structure-
aware 3d gaussian splatting. arXiv:2404.19149 (2024)
27. Wang, F., Tan, S., Li, X., Tian, Z., Liu, H.: Mixed neural voxels for fast multi-view
video synthesis. arXiv preprint arXiv:2212.00190 (2022)
28. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment:
from error visibility to structural similarity. IEEE transactions on image processing
13(4), 600–612 (2004)
29. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Xinggang,
W.: 4d gaussian splatting for real-time dynamic scene rendering. arXiv preprint
arXiv:2310.08528 (2023)
30. Yang, Z., Yang, H., Pan, Z., Zhang, L.: Real-time photorealistic dynamic scene rep-
resentation and rendering with 4d gaussian splatting. In: International Conference
on Learning Representations (ICLR) (2024)
31. Yang, Z., Yang, H., Pan, Z., Zhu, X., Zhang, L.: Real-time photorealistic dynamic
scene representation and rendering with 4d gaussian splatting. arXiv preprint
arXiv:2310.10642 (2023)

---

## Page 16

16 J. Bae et al.
32. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaus-
sians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint
arXiv:2309.13101 (2023)
33. Yu, H., Julin, J., Milacski, Z.Ã., Niinuma, K., Jeni, L.A.: Cogs: Controllable gaus-
sian splatting (2023)
34. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable
effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE
conference on computer vision and pattern recognition. pp. 586–595 (2018)
35. Zhao, B., Li, Y., Sun, Z., Zeng, L., Shen, Y., Ma, R., Zhang, Y., Bao, H., Cui,
Z.: Gaussianprediction: Dynamic 3d gaussian prediction for motion extrapolation
and free view synthesis. In: Special Interest Group on Computer Graphics and
Interactive Techniques Conference Conference Papers âĂŹ24. SIGGRAPH âĂŹ24,
ACM (Jul 2024). https://doi.org/10.1145/3641519.3657417 ,http://dx.doi.
org/10.1145/3641519.3657417
36. Zwicker, M., Pfister, H., Van Baar, J., Gross, M.: Surface splatting. In: Proceedings
of the 28th annual conference on Computer graphics and interactive techniques.
pp. 371–378 (2001)

---

## Page 17

E-D3DGS 17
In the Appendix, we provide additional training details (Appendix A) and
moreresultsofthebaseline,alongwiththeper-scenequantitativeandqualitative
results of our method on all datasets (Appendix B).
A Training Details
A.1 Efficient Training Strategy
multi-view error mapdensified 
Gaussians
(c) Densification induction 
framesprobabilitysampled frameerror heatmap
(b) Frame samplingprevious camera
excluded 
cameras
(a) Camera samplingmulti-camera array
Fig.S1: Components of efficient training strategy.
We introduce a training strategy for faster convergence and higher perfor-
mance. The first is to evenly cover multi-view camera perspectives and exclude
the camera index that was used in the previous iteration. We pre-compute pair-
wise distances of all camera origins before the start of the training. Then we
exclude the cameras with distance to the previous camera less than 40 percentile
of all distances.
The second is to sample frames within the target cameras. Different frames
may have different difficulties for reconstruction. We measure the difficulty of a
frame by the magnitude of error at the frame. Then, we sample training frames
with a categorical distribution which is proportional to the magnitude of error.
For multi-view videos, we average the errors from all viewpoints. For the first
10Kiterations,werandomlysampletheframes,andthenalternatelyuserandom
sampling and error-based frame sampling.
The third is to change the policy for Gaussian densification. Previous meth-
ods invoke the densification by minimizing L1 loss [29] or L1 loss and DSSIM
loss for every iteration [13,31,32]. We observe that the DSSIM loss improves
the visual quality in the background but takes longer training time. Therefore,
we use L1 loss for every iteration and periodically use the additional multi-view
DSSIM loss in the frame with high error. This encourages densification in the
region where the model struggles. We fix the frame obtained by loss-based sam-
pling for every 50 iterations, and minimize the multi-view DSSIM loss through
camera sampling during the next 5 iterations. Similar to frame sampling, DSSIM
loss is applied after the first 10K iterations.

---

## Page 18

18 J. Bae et al.
A.2 Implementation Details
Table S1: Ablation study on the size of temporal and per-Gaussian em-
beddings. We compare the results on the flame_steak scene. We set 256 /32 as the
default setting.
ztdim/ zgdimPSNR ↑SSIM↑LPIPS ↓ztdim/ zgdimPSNR ↑SSIM↑LPIPS ↓
256 / 16 33.53 0.962 0.030 128 / 32 33.61 0.964 0.029
256 / 32 33.570.964 0.028 256 / 32 33.570.964 0.028
256 / 64 33.71 0.964 0.029 512 / 32 33.55 0.964 0.028
We report the performance variation by changing the size of per-Gaussian
and temporal embeddings in Table S1. The embedding size does not significantly
affect the overall performance. In all experiments, considering the capacity and
details, we use a 32-dimensional vector for Gaussian embeddings zgand a 256-
dimensional vector for temporal embeddings zt.
The decoder MLP consists 128 hidden units and the MLP head for Gaussian
parameters are both composed of 2 layers with 128 hidden units. For Technicolor
and HyperNeRF datasets, we use a 1-layer decoder MLP. To efficiently and
stabilizetheinitialtraining,westartthetemporalembeddinggrid Zf
tatthesame
N/5time resolution as Zc
tand gradually increase it to Nresolution over 10K
iterations, with Nset to 150 for 300 frames. The learning rate of our deformation
decoder starts at 1.6×10−4and exponentially decays to 1.6×10−5over the total
training iterations. The learning rate of the temporal embedding ztfollows that
ofthedeformationdecoder,whilethelearningrateofper-Gaussianembedding zg
is set to 2.5×10−3. We eliminate the opacity reset step of 3DGS and instead add
a loss that minimizes the mean of Gaussian opacities with a weight of 1.0×10−4.
We empirically set the start of the training strategy to 10K iterations.
For the Neural 3D Video dataset, we follow Sync-NeRF [11] to introduce
trainable time offsets to each camera. In addition, we learn the deformation
of color, scale, and opacity from 5K iterations and perform densification and
pruning.
Forinitializationpoints,wedownsamplepointcloudsobtainedfromCOLMAP
dense scene reconstruction. For the Neural 3D Video dataset, we train for 80K it-
erations; for the Technicolor dataset, 100K for the Birthday andPainter scenes
and 120K for the rest; for the HyperNeRF dataset, 80K for the Bananascene
and 60K for the others. We periodically use the DSSIM loss at specific steps, ex-
cluding the HyperNeRF dataset. For calculating computational cost, the Neural
3D Video dataset was measured using A6000, while the Technicolor Light Field
dataset and the HyperNeRF dataset were measured using 3090.
For local smoothness embedding regularization, we set the weight of loss to 1
on the Neural 3D Video dataset and 0.1 on the Technicolor Light Field dataset
and the HyperNeRF dataset.

---

## Page 19

E-D3DGS 19
B More Results
We show the additional rendering of 4DGasussians [29] in Figure S2. And, we
show the entire rendering results of ours in Figure S3-S5. In most scenes, our
model shows better perceptual performance than the baselines. Table S2-S4 re-
port the quantitative results of each scene on the Neural 3D Video dataset,
Technicolor Light Field dataset, and HyperNeRF dataset.
B.1 Additional Results of 4DGaussians
flame_salmon_frag4 flame_salmon_frag3 flame_salmon_frag2 flame_salmon_frag1
Fig.S2: Rendering results of 4DGaussians on Neural 3D Video dataset.
The results of 4DGaussians are produced using the official repository. In the
teaser, we choose the thirdfragment of flame salmon scene which contains
more difficult movements than the typical first fragment to demonstrate our
superiority.

---

## Page 20

20 J. Bae et al.
B.2 Per-Scene Results of Neural 3D Video Dataset
flame_salmon_frag4 flame_salmon_frag3 flame_salmon_frag2
flame_steak
cof fee_martini cut_roasted_beef
sear_steak
flame_salmon_frag1cook_spinach
Fig.S3: Rendering results on Neural 3D Video dataset.
Table S2: Per-scene quantitative results on Neural 3D Video dataset
ModelMetric Average coffee_martini cook_spinach cut_roasted_beef flame_salmon_1
PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑
MixVoxels 30.30 0.918 29.44 0.916 29.97 0.934 32.58 0.938 30.50 0.918
K-Planes 30.86 0.939 29.660.926 31.82 0.943 31.82 0.966 30.68 0.928
HyperReel 30.37 0.921 28.37 0.892 32.30 0.941 32.92 0.945 28.26 0.882
4DGS 31.190.940 28.63 0.918 33.540.956 34.18 0.959 29.250.929
4DGaussians 30.71 0.935 28.44 0.919 33.10 0.953 33.32 0.954 28.80 0.926
Ours 31.310.945 29.100.931 32.960.956 33.57 0.958 29.610.936
ModelMetric flame_salmon_2 flame_salmon_3 flame_salmon_4 flame_steak sear_steak
PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑
MixVoxels 30.53 0.915 27.83 0.853 29.49 0.899 30.74 0.945 31.61 0.949
K-Planes 29.98 0.924 30.10 0.924 30.37 0.922 31.85 0.969 31.48 0.951
HyperReel 28.80 0.911 28.97 0.911 28.92 0.908 32.20 0.949 32.57 0.952
4DGS 29.280.929 29.400.927 29.130.925 33.90 0.961 33.37 0.960
4DGaussians 28.21 0.913 28.68 0.917 28.45 0.913 33.55 0.961 34.020.963
Ours 29.690.934 29.940.934 29.900.933 33.57 0.964 33.450.963

---

## Page 21

E-D3DGS 21
B.3 Per-Scene Results of Technicolor Light Field Dataset
Theater Fabien
T rain Painter Birthday
Fig.S4: Rendering results on Technicolor Light Field dataset.
Table S3: Per-scene quantitative results on Technicolor dataset
ModelMetric Average Birthday Fabien
PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑
DyNeRF 31.80 - 29.20 - 32.76 -
HyperReel 32.320.89930.570.91832.49 0.863
4DGaussians 29.62 0.840 28.03 0.862 33.360.865
Ours 33.230.90732.900.95134.710.885
ModelMetric Painter Theater Train
PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑
DyNeRF 35.95 - 29.53 - 31.58 -
HyperReel 35.510.92433.760.89729.300.894
4DGaussians 34.52 0.899 28.67 0.835 23.54 0.756
Ours 36.180.92431.070.86831.330.912

---

## Page 22

22 J. Bae et al.
B.4 Per-Scene Results of HyperNeRF Dataset
Broom 3D Printer Chicken Banana
Fig.S5: Rendering results on HyperNeRF dataset.
Table S4: Per-scene quantitative results on HyperNeRF dataset
ModelMetric Average Broom 3D Printer Chicken Banana
PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑PSNR↑SSIM↑
Nerfies 22.23 - 19.30 - 20.00 - 26.90 - 23.30 -
HyperNeRF DS 22.29 0.598 19.51 0.210 20.04 0.635 27.460.82822.15 0.719
TiNeuVox 24.20 0.616 21.28 0.307 22.800.72528.22 0.785 24.50 0.646
D3DGS 22.40 0.598 20.48 0.313 20.38 0.644 22.64 0.601 26.10 0.832
4DGaussians 25.030.68222.010.36721.99 0.705 28.47 0.806 27.280.845
Ours 25.430.69721.840.37122.340.71528.750.83628.800.867