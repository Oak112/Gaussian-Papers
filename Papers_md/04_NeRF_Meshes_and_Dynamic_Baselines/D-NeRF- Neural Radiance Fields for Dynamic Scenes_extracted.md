

---

## Page 1

D-NeRF: Neural Radiance Fields for Dynamic Scenes
Albert Pumarola1Enric Corona1Gerard Pons-Moll2Francesc Moreno-Noguer1
1Institut de Rob `otica i Inform `atica Industrial, CSIC-UPC
2Max Planck Institute for Informatics
Point of View & Time
Figure 1: We propose D-NeRF, a method for synthesizing novel views, at an arbitrary point in time, of dynamic scenes with complex
non-rigid geometries. We optimize an underlying deformable volumetric function from a sparse set of input monocular views without
the need of ground-truth geometry nor multi-view images. The ﬁgure shows two scenes under variable points of view and time instances
synthesised by the proposed model.
Abstract
Neural rendering techniques combining machine learn-
ing with geometric reasoning have arisen as one of the most
promising approaches for synthesizing novel views of a
scene from a sparse set of images. Among these, stands out
the Neural radiance ﬁelds (NeRF) [26], which trains a deep
network to map 5D input coordinates (representing spatial
location and viewing direction) into a volume density and
view-dependent emitted radiance. However, despite achiev-
ing an unprecedented level of photorealism on the gener-
ated images, NeRF is only applicable to static scenes, where
the same spatial location can be queried from different im-
ages. In this paper we introduce D-NeRF , a method that
extends neural radiance ﬁelds to a dynamic domain, allow-
ing to reconstruct and render novel images of objects under
rigid and non-rigid motions from a single camera moving
around the scene. For this purpose we consider time as an
additional input to the system, and split the learning process
in two main stages: one that encodes the scene into a canon-
ical space and another that maps this canonical represen-
tation into the deformed scene at a particular time. Bothmappings are simultaneously learned using fully-connected
networks. Once the networks are trained, D-NeRF can ren-
der novel images, controlling both the camera view and the
time variable, and thus, the object movement. We demon-
strate the effectiveness of our approach on scenes with ob-
jects under rigid, articulated and non-rigid motions. Code,
model weights and the dynamic scenes dataset will be re-
leased.
1. Introduction
Rendering novel photo-realistic views of a scene from
a sparse set of input images is necessary for many appli-
cations in e.g. augmented reality, virtual reality, 3D con-
tent production, games and the movie industry. Recent
advances in the emerging ﬁeld of neural rendering, which
learn scene representations encoding both geometry and
appearance [26, 23, 19, 50, 29, 35], have achieved re-
sults that largely surpass those of traditional Structure-
from-Motion [14, 41, 38], light-ﬁeld photography [18] and
image-based rendering approaches [5]. For instance, the
1arXiv:2011.13961v1  [cs.CV]  27 Nov 2020

---

## Page 2

Neural Radiance Fields (NeRF) [26] have shown that sim-
ple multilayer perceptron networks can encode the mapping
from 5D inputs (representing spatial locations (x;y;z )and
camera views (;)) to emitted radiance values and volume
density. This learned mapping allows then free-viewpoint
rendering with extraordinary realism. Subsequent works
have extended Neural Radiance Fields to images in the wild
undergoing severe lighting changes [23] and have proposed
sparse voxel ﬁelds for rapid inference [19]. Similar schemes
have also been recently used for multi-view surface recon-
struction [50] and learning surface light ﬁelds [30].
Nevertheless, all these approaches assume a static scene
without moving objects. In this paper we relax this assump-
tion and propose, to the best of our knowledge, the ﬁrst end-
to-end neural rendering system that is applicable to dynamic
scenes, made of both still and moving/deforming objects.
While there exist approaches for 4D view synthesis [2], our
approach is different in that: 1) we only require a single
camera; 2) we do not need to pre-compute a 3D reconstruc-
tion; and 3) our approach can be trained end-to-end.
Our idea is to represent the input of our system with
a continuous 6D function, which besides 3D location and
camera view, it also considers the time component t.
Naively extending NeRF to learn a mapping from (x;y;z;t )
to density and radiance does not produce satisfying results,
as the temporal redundancy in the scene is not effectively
exploited. Our observation is that objects can move and
deform, but typically do not appear or disappear. Inspired
by classical 3D scene ﬂow [44], the core idea to build our
method, denoted Dynamic-NeRF (D-NeRF in short), is to
decompose learning in two modules. The ﬁrst one learns a
spatial mapping (x;y;z;t )!(x;y;z)between each
point of the scene at time tand a canonical scene conﬁg-
uration. The second module regresses the scene radiance
emitted in each direction and volume density given the tu-
ple(x+ x;y+ y;z+ z;; ). Both mappings are
learned with deep fully connected networks without convo-
lutional layers. The learned model then allows to synthesize
novel images, providing control in the continuum (;;t )
of the camera views and time component, or equivalently,
the dynamic state of the scene (see Fig. 1).
We thoroughly evaluate D-NeRF on scenes undergoing
very different types of deformation, from articulated mo-
tion to humans performing complex body poses. We show
that by decomposing learning into a canonical scene and
scene ﬂow D-NeRF is able to render high-quality images
while controlling both camera view and time components.
As a side-product, our method is also able to produce com-
plete 3D meshes that capture the time-varying geometry and
which remarkably are obtained by observing the scene un-
der a speciﬁc deformation only from one single viewpoint.2. Related work
Neural implicit representation for 3D geometry. The
success of deep learning on the 2D domain has spurred a
growing interest in the 3D domain. Nevertheless, which
is the most appropriate 3D data representation for deep
learning remains an open question, especially for non-
rigid geometry. Standard representations for rigid geome-
try include point-clouds [42, 33], voxels [13, 48] and oc-
trees [45, 39]. Recently, there has been a strong burst in
representing 3D data in an implicit manner via a neural net-
work [24, 31, 6, 47, 8, 12]. The main idea behind this ap-
proach is to describe the information ( e.g. occupancy, dis-
tance to surface, color, illumination) of a 3D point xas the
output of a neural network f(x). Compared to the previ-
ously mentioned representations, neural implicit represen-
tations allow for continuous surface reconstruction at a low
memory footprint.
The ﬁrst works exploiting implicit representations [24,
31, 6, 47] for 3D representation were limited by their re-
quirement of having access to 3D ground-truth geometry,
often expensive or even impossible to obtain for in the
wild scenes. Subsequent works relaxed this requirement
by introducing a differentiable render allowing 2D super-
vision. For instance, [20] proposed an efﬁcient ray-based
ﬁeld probing algorithm for efﬁcient image-to-ﬁeld supervi-
sion. [29, 49] introduced an implicit-based method to cal-
culate the exact derivative of a 3D occupancy ﬁeld surface
intersection with a camera ray. In [37], a recurrent neu-
ral network was used to ray-cast the scene and estimate the
surface geometry. However, despite these techniques have
a great potential to represent 3D shapes in an unsupervised
manner, they are typically limited to relatively simple ge-
ometries.
NeRF [26] showed that by implicitly representing a rigid
scene using 5D radiance ﬁelds makes it possible to capture
high-resolution geometry and photo-realistically rendering
novel views. [23] extended this method to handle variable
illumination and transient occlusions to deal with in the wild
images. In [19], even more complex 3D surfaces were rep-
resented by using voxel-bouded implicit ﬁelds. And [50]
circumvented the need of multiview camera calibration.
However, while all mentioned methods achieve impres-
sive results on rigid scenes, none of them can deal with dy-
namic and deformable scenes. Occupancy ﬂow [28] was the
ﬁrst work to tackle non-rigid geometry by learning continu-
ous vector ﬁeld assigning a motion vector to every point in
space and time, but it requires full 3D ground-truth super-
vision. Neural volumes [21] produced high quality recon-
struction results via an encoder-decoder voxel-based repre-
sentation enhanced with an implicit voxel warp ﬁeld, but
they require a muti-view image capture setting.
To the best of our knowledge, D-NeRF is the ﬁrst ap-
proach able to generate a neural implicit representation
2

---

## Page 3

Figure 2: Problem Deﬁnition. Given a sparse set of images of a dynamic scene moving non-rigidly and being captured by a monocular
camera, we aim to design a deep learning model to implicitly encode the scene and synthesize novel views at an arbitrary time. Here,
we visualize a subset of the input training frames paired with accompanying camera parameters, and we show three novel views at three
different time instances rendered by the proposed method.
for non-rigid and time-varying scenes, trained solely on
monocular data without the need of 3D ground-truth super-
vision nor a multi-view camera setting.
Novel view synthesis. Novel view synthesis is a long
standing vision and graphics problem that aims to synthe-
size new images from arbitrary view points of a scene cap-
tured by multiple images. Most traditional approaches for
rigid scenes consist on reconstructing the scene from multi-
ple views with Structure-from-Motion [14] and bundle ad-
justment [41], while other approaches propose light-ﬁeld
based photography [18]. More recently, deep learning based
techniques [36, 16, 10, 9, 25] are able to learn a neural vol-
umetric representation from a set of sparse images.
However, none of these methods can synthesize novel
views of dynamic scenes. To tackle non-rigid scenes most
methods approach the problem by reconstructing a dynamic
3D textured mesh. 3D reconstruction of non-rigid sur-
faces from monocular images is known to be severely ill-
posed. Structure-from-Template (SfT) approaches [3, 7, 27]
recover the surface geometry given a reference known
template conﬁguration. Temporal information is another
prior typically exploited. Non-rigid-Structure-from-Motion
(NRSfM) techniques [40, 1] exploit temporal information.
Yet, SfT and NRSfM require either 2D-to-3D matches or
2D point tracks, limiting their general applicability to rela-
tively well-textured surfaces and mild deformations.
Some of these limitations are overcome by learning
based techniques, which have been effectively used for syn-
thesizing novel photo-realistic views of dynamic scenes.
For instance, [2, 54, 15] capture the dynamic scene at the
same time instant from multiple views, to then generate 4D
space-time visualizations. [11, 32, 53] also leverage on si-
multaneously capturing the scene from multiple cameras to
estimate depth, completing areas with missing information
and then performing view synthesis. In [51], the need of
multiple views is circumvented by using a pre-trained net-
work that estimates a per frame depth. This depth, jointly
with the optical ﬂow and consistent depth estimation across
frames, are then used to interpolate between images andrender novel views. Nevertheless, by decoupling depth es-
timation from novel view synthesis, the outcome of this
approach becomes highly dependent on the quality of the
depth maps as well as on the reliability of the optical ﬂow.
Very recently, X-Fields [4] introduced a neural network
to interpolate between images taken across different view,
time or illumination conditions. However, while this ap-
proach is able to process dynamic scenes, it requires more
than one view. Since no 3D representation is learned, vari-
ation in viewpoint is small.
D-NeRF is different from all prior work in that it does
not require 3D reconstruction, can be learned end-to-end,
and requires a single view per time instance. Another ap-
pealing characteristic of D-NeRF is that it inherently learns
a time-varying 3D volume density and emitted radiance,
which turns the novel view synthesis into a ray-casting pro-
cess instead of a view interpolation, which is remarkably
more robust to rendering images from arbitrary viewpoints.
3. Problem Formulation
Given a sparse set of images of a dynamic scene captured
with a monocular camera, we aim to design a deep learning
model able to implicitly encode the scene and synthesize
novel views at an arbitrary time (see Fig. 2).
Formally, our goal is to learn a mapping Mthat, given
a 3D point x= (x;y;z ), outputs its emitted color c=
(r;g;b )and volume density conditioned on a time instant
tand view direction d= (;). That is, we seek to estimate
the mapping M: (x;d;t)!(c;).
An intuitive solution would be to directly learn the trans-
formation Mfrom the 6D space (x;d;t)to the 4D space
(c;). However, as we will show in the results section, we
obtain consistently better results by splitting the mapping M
into	xand	t, where 	xrepresents the scene in canoni-
cal conﬁguration and 	ta mapping between the scene at
time instant tand the canonical one. More precisely, given
a point xand viewing direction dat time instant twe ﬁrst
transform the point position to its canonical conﬁguration
as	t: (x;t)!x. Without loss of generality, we chose
3

---

## Page 4

Figure 3: D-NeRF Model . The proposed architecture consists of two main blocks: a deformation network 	tmapping all scene
deformations to a common canonical conﬁguration; and a canonical network 	xregressing volume density and view-dependent RGB
color from every camera ray.
t= 0as the canonical scene 	t: (x;0)!0. By doing so
the scene is no longer independent between time instances,
and becomes interconnected through a common canonical
space anchor. Then, the assigned emitted color and vol-
ume density under viewing direction dequal to those in the
canonical conﬁguration 	x: (x+ x;d)!(c;).
We propose to learn 	xand	tusing a sparse set of T
RGB imagesfIt;TtgT
t=1captured with a monocular cam-
era, where It2RHW3denotes the image acquired un-
der camera pose Tt2R44SE(3), at time t. Although
we could assume multiple views per time instance, we want
to test the limits of our method, and assume a single image
per time instance. That is, we do not observe the scene un-
der a speciﬁc conﬁguration/deformation state from different
viewpoints.
4. Method
We now introduce D-NeRF, our novel neural renderer for
view synthesis trained solely from a sparse set of images of
a dynamic scene. We build on NeRF [26] and generalize it
to handle non-rigid scenes. Recall that NeRF requires mul-
tiple views of a rigid scene In contrast, D-NeRF can learn a
volumetric density representation for continuous non-rigid
scenes trained with a single view per time instant.
As shown in Fig. 3, D-NeRF consists of two main neu-
ral network modules, which parameterize the mappings ex-
plained in the previous section 	t;	x. On the one hand we
have the Canonical Network , an MLP (multilayer percep-
tron) 	x(x;d)7!(c;)is trained to encode the scene in
the canonical conﬁguration such that given a 3D point xand
a view direction dreturns its emitted color cand volume
density. The second module is called Deformation Net-
work and consists of another MLP 	t(x;t)7!xwhich
predicts a deformation ﬁeld deﬁning the transformation be-
tween the scene at time tand the scene in its canonical
conﬁguration. We next describe in detail each one of these
blocks (Sec. 4.1), their interconnection for volume render-
ing (Sec. 4.2) and how are they learned (Sec. 4.3).4.1. Model Architecture
Canonical Network. With the use of a canonical conﬁg-
uration we seek to ﬁnd a representation of the scene that
brings together the information of all corresponding points
in all images. By doing this, the missing information from a
speciﬁc viewpoint can then be retrieved from that canonical
conﬁguration, which shall act as an anchor interconnecting
all images.
The canonical network 	xis trained so as to encode vol-
umetric density and color of the scene in canonical conﬁg-
uration. Concretely, given the 3D coordinates xof a point,
we ﬁrst encode it into a 256-dimensional feature vector.
This feature vector is then concatenated with the camera
viewing direction d, and propagated through a fully con-
nected layer to yield the emitted color cand volume density
for that given point in the canonical space.
Deformation Network. The deformation network 	tis op-
timized to estimate the deformation ﬁeld between the scene
at a speciﬁc time instant and the scene in canonical space.
Formally, given a 3D point xat timet,	tis trained to out-
put the displacement xthat transforms the given point to
its position in the canonical space as x+ x. For all ex-
periments, without loss of generality, we set the canonical
scene to be the scene at time t= 0:
	t(x;t) =(
x;ift6= 0
0; ift= 0(1)
As shown in previous works [34, 43, 26], directly feed-
ing raw coordinates and angles to a neural network results in
low performance. Thus, for both the canonical and the de-
formation networks, we ﬁrst encode x,dandtinto a higher
dimension space. We use the same positional encoder as
in [26] where (p) =<(sin(2lp);cos(2lp))>L
0. We in-
dependently apply the encoder ()to each coordinate and
camera view component, using L= 10 forx, andL= 4
fordandt.
4

---

## Page 5

4.2. Volume Rendering
We now adapt NeRF volume rendering equations to ac-
count for non-rigid deformations in the proposed 6D neural
radiance ﬁeld. Let x(h) =o+hdbe a point along the cam-
era ray emitted from the center of projection oto a pixelp.
Considering near and far bounds hnandhfin that ray, the
expected color Cof the pixelpat timetis given by:
C(p;t) =Zhf
hnT(h;t)(p(h;t))c(p(h;t);d)dh; (2)
where p(h;t) =x(h) + 	 t(x(h);t); (3)
[c(p(h;t);d);(p(h;t))] = 	 x(p(h;t);d); (4)
andT(h;t) = exp 
 Zh
hn(p(s;t))ds!
: (5)
The 3D point p(h;t)denotes the point on the camera ray
x(h)transformed to canonical space using our Deformation
Network 	t, andT(h;t)is the accumulated probability that
the ray emitted from hntohfdoes not hit any other particle.
Notice that the density and color care predicted by our
Canonical Network 	x.
As in [26] the volume rendering integrals in Eq. (2)
and Eq. (5) can be approximated via numerical quadrature.
To select a random set of quadrature points fhngN
n=12
[hn;hf]a stratiﬁed sampling strategy is applied by uni-
formly drawing samples from evenly-spaced ray bins. A
pixel color is approximated as:
C0(p;t) =NX
n=1T0(hn;t)(hn;t;n)c(p(hn;t);d);(6)
where(h;t; ) = 1 exp( (p(h;t))); (7)
andT0(hn;t) = exp 
 n 1X
m=1(p(hm;t))m!
;(8)
andn=hn+1 hnis the distance between two quadrature
points.
4.3. Learning the Model
The parameters of the canonical 	xand deformation
	tnetworks are simultaneously learned by minimizing the
mean squared error with respect to the TRGB images
fItgT
t=1of the scene and their corresponding camera pose
matricesfTtgT
t=1. Recall that every time instant is only
acquired by a single camera.
At each training batch, we ﬁrst sample a random set of
pixelsfpt;igNs
i=1corresponding to the rays cast from some
camera position Ttto some pixels iof the corresponding
RGB image t. We then estimate the colors of the chosen
pixels using Eq. (6). The training loss we use is the meansquared error between the rendered and real pixels:
L=1
NsNsX
i=1^C(p;t) C0(p;t)2
2(9)
where ^Care the pixels’ ground-truth color.
5. Implementation Details
Both the canonical network 	xand the deformation net-
work 	tconsists on simple 8-layers MLPs with ReLU ac-
tivations. For the canonical network a ﬁnal sigmoid non-
linearity is applied to cand. No non-linearlity is applied
toxin the deformation network.
For all experiments we set the canonical conﬁguration
as the scene state at t= 0 by enforcing it in Eq. (1). To
improve the networks convergence, we sort the input im-
ages according to their time stamps (from lower to higher)
and then we apply a curriculum learning strategy where we
incrementally add images with higher time stamps.
The model is trained with 400400images during 800k
iterations with a batch size of Ns= 4096 rays, each sam-
pled 64times along the ray. As for the optimizer, we
use Adam [17] with learning rate of 5e 4,1= 0:9,
2= 0:999and exponential decay to 5e 5. The model
is trained with a single Nvidia®GTX 1080 for 2 days.
6. Experiments
This section provides a thorough evaluation of our sys-
tem. We ﬁrst test the main components of the model,
namely the canonical and deformation networks (Sec. 6.1).
We then compare D-NeRF against NeRF and T-NeRF,
a variant in which does not use the canonical mapping
(Sec. 6.2). Finally, we demonstrate D-NeRF ability to syn-
thesize novel views at an arbitrary time in several complex
dynamic scenes (Sec. 6.3).
In order to perform an exhaustive evaluation we have ex-
tended NeRF [26] rigid benchmark with eight scenes con-
taining dynamic objects under large deformations and real-
istic non-Lambertian materials. As in the rigid benchmark
of [26], six are rendered from viewpoints sampled from the
upper hemisphere, and two are rendered from viewpoints
sampled on the full sphere. Each scene contains between
100 and 200 rendered views depending on the action time
span, all at 800 × 800 pixels. We will release the path-
traced images with deﬁned train/validation/test splits for
these eight scenes.
6.1. Dissecting the Model
This subsection provides insights about D-NeRF be-
haviour when modeling a dynamic scene and analyze the
two main modules, namely the canonical and deformation
networks.
5

---

## Page 6

Figure 4: Visualization of the Learned Scene Representation. Given a dynamic scene at a speciﬁc time instant, D-NeRF learns a
displacement ﬁeld xthat maps all points xof the scene to a common canonical conﬁguration. The volume density and view-dependent
emitted radiance for this conﬁguration is learned and transferred to the original input points to render novel views. This ﬁgure represents,
from left to right: the learned radiance from a speciﬁc viewpoint, the volume density represented as a 3D mesh and a depth map, and the
color-coded points of the canonical conﬁguration mapped to the deformed meshes based on x. The same colors on corresponding points
indicate the correctness of such mapping.
Canonical Space t=0.5 t=1
Figure 5: Analyzing Shading Effects. Pairs of corresponding
points between the canonical space and the scene at times t= 0:5
andt= 1.
We initially evaluate the ability of the canonical network
to represent the scene in a canonical conﬁguration. The re-
sults of this analysis for two scenes are shown the ﬁrst row
of Fig. 4 (columns 1-3 in each case). The plots show, for
the canonical conﬁguration ( t= 0), the RGB image, the 3D
occupancy network and the depth map, respectively. The
rendered RGB image is the result of evaluating the canoni-
cal network on rays cast from an arbitrary camera position
applying Eq. (6). To better visualize the learned volumet-
ric density we transform it into a mesh applying marching
cubes [22], with a 3D cube resolution of 2563voxels. Note
how D-NeRF is able to model ﬁne geometric and appear-
ance details for complex topologies and texture patterns,
even when it was only trained with a set of sparse images,
each under a different deformation.
In a second experiment we assess the capacity of the net-
work to estimate consistent deformation ﬁelds that map the
canonical scene to the particular shape at each input image.
The second and third rows of Fig. 4 show the result of ap-plying the corresponding translation vectors to the canon-
ical space for t= 0:5andt= 1. The fourth column in
each of the two examples visualizes the displacement ﬁeld,
where the color-coded points in the canonical shape ( t= 0)
at mapped to the different shape conﬁgurations at t= 0:5
andt= 1. Note that the colors are consistent along the
time instants, indicating that the displacement ﬁeld is cor-
rectly estimated.
Another question that we try to answer is how D-NeRF
manages to model phenomena like shadows/shading ef-
fects, that is, how the model can encode changes of ap-
pearance of the same point along time. We have carried
an additional experiment to answer this. In Fig. 5 we show
a scene with three balls, made of very different materials
(plastic –green–, translucent glass –blue– and metal –red–).
The ﬁgure plots pairs of corresponding points between the
canonical conﬁguration and the scene at a speciﬁc time in-
stant. D-NeRF is able to synthesize the shading effects by
warping the canonical conﬁguration. For instance, observe
how the ﬂoor shadows are warped along time. Note that the
points in the shadow of, e.g. the red ball, at t= 0:5and
t= 1map at different regions of the canonical space.
6.2. Quantitative Comparison
We next evaluate the quality of D-NeRF on the novel
view synthesis problem and compare it against the origi-
nal NeRF [26], which represents the scene using a 5D in-
put(x;y;z;; ), and T-NeRF, a straight-forward exten-
6

---

## Page 7

Figure 6: Qualitative Comparison. Novel view synthesis results of dynamic scenes. For every scene we show an image synthesised
from a novel view at an arbitrary time by our method, and three close-ups for: ground-truth, NeRF, T-NeRF, and D-NeRF (ours).
Hell Warrior Mutant Hook Bouncing Balls
Method MSE#PSNR"SSIM"LPIPS#MSE#PSNR"SSIM"LPIPS#MSE#PSNR"SSIM"LPIPS#MSE#PSNR"SSIM"LPIPS#
NeRF 44e-3 13.52 0.81 0.25 9e-4 20.31 0.91 0.09 21e-3 16.65 0.84 0.19 1e-2 18.28 0.88 0.23
T-NeRF 47e-4 23.19 0.93 0.08 8e-4 30.56 0.96 0.04 18e-4 27.21 0.94 0.06 6e-4 32.01 0.97 0.04
D-NeRF 31e-4 25.02 0.95 0.06 7e-4 31.29 0.97 0.02 11e-4 29.25 0.96 0.11 5e-4 32.80 0.98 0.03
Lego T-Rex Stand Up Jumping Jacks
Method MSE#PSNR"SSIM"LPIPS#MSE#PSNR"SSIM"LPIPS#MSE#PSNR"SSIM"LPIPS#MSE#PSNR"SSIM"LPIPS#
NeRF 9e-4 20.30 0.79 0.23 3e-3 24.49 0.93 0.13 1e-2 18.19 0.89 0.14 1e-2 18.28 0.88 0.23
T-NeRF 3e-4 23.82 0.90 0.15 9e-4 30.19 0.96 0.13 7e-4 31.24 0.97 0.02 6e-4 32.01 0.97 0.03
D-NeRF 6e-4 21.64 0.83 0.16 6e-4 31.75 0.97 0.03 5e-4 32.79 0.98 0.02 5e-4 32.80 0.98 0.03
Table 1: Quantitative Comparison. We report MSE/LPIPS (lower is better) and PSNR/SSIM (higher is better).
sion of NeRF in which the scene is represented by a 6D
input (x;y;z;;;t ), without considering the intermediate
canonical conﬁguration of D-NeRF.
Table 1 summarizes the quantitative results on the 8 dy-
namic scenes of our dataset. We use several metrics for
the evaluation: Mean Squared Error (MSE), Peak Signal-to-
Noise Ratio (PSNR), Structural Similarity (SSIM) [46] and
Learned Perceptual Image Patch Similarity (LPIPS) [52].
In Fig. 6 we show samples of the estimated images under a
novel view for visual inspection. As expected, NeRF is not
able to model the dynamics scenes as it was designed for
rigid cases, and always converges to a blurry mean represen-
tation of all deformations. On the other hand, the T-NeRF
baseline is able to capture reasonably well the dynamics, al-
though is not able to retrieve high frequency details. For ex-
ample, in Fig. 6 top-left image it fails to encode the shoulderpad spikes, and in the top-right scene it is not able to model
the stones and cracks. D-NeRF, instead, retains high details
of the original image in the novel views. This is quite re-
markable, considering that each deformation state has only
been seen from a single viewpoint.
6.3. Additional Results
We ﬁnally show additional results to showcase the wide
range of scenarios that can be handled with D-NeRF. Fig. 7
depicts, for four scenes, the images rendered at different
time instants from two novel viewpoints. The ﬁrst column
displays the canonical conﬁguration. Note that we are able
to handle several types of dynamics: articulated motion
in the Tractor scene; human motion in the Jumping Jacks
andWarrior scenes; and asynchronous motion of several
Bouncing Balls . Also note that the canonical conﬁguration
7

---

## Page 8

t=0.1 t=0.3 t=1.0 t=0.5 t=0.8 Canonical SpaceFigure 7: Time & View Conditioning. Results of synthesising diverse scenes from two novel points of view across time and the learned
canonical space. For every scene we also display the learned scene canonical space in the ﬁrst column.
is a sharp and neat scene, in all cases, expect for the Jump-
ing Jacks, where the two arms appear to be blurry. This,
however, does not harm the quality of the rendered images,
indicating that the network is able warp the canonical con-
ﬁguration so as to maximize the rendering quality. This is
indeed consistent with Sec. 6.1 insights on how the network
is able to encode shading.
7. Conclusion
We have presented D-NeRF, a novel neural radiance ﬁeld
approach for modeling dynamic scenes. Our method can
be trained end-to-end from only a sparse set of images ac-quired with a moving camera, and does not require pre-
computed 3D priors nor observing the same scene conﬁg-
uration from different viewpoints. The main idea behind D-
NeRF is to represent time-varying deformations with two
modules: one that learns a canonical conﬁguration, and an-
other that learns the displacement ﬁeld of the scene at each
time instant w.r.t. the canonical space. A thorough evalu-
ation demonstrates that D-NeRF is able to synthesise high
quality novel views of scenes undergoing different types of
deformation, from articulated objects to human bodies per-
forming complex body postures.
8

---

## Page 9

Acknowledgments This work is supported in part by a Google
Daydream Research award and by the Spanish government with the project
HuMoUR TIN2017-90086-R, the ERA-Net Chistera project IPALM
PCI2019-103386 and Mar ´ıa de Maeztu Seal of Excellence MDM-2016-
0656. Gerard Pons-Moll is funded by the Deutsche Forschungsgemein-
schaft (DFG, German Research Foundation) - 409792180 (Emmy Noether
Programme, project: Real Virtual Humans)
References
[1] Antonio Agudo and Francesc Moreno-Noguer. Simultaneous
pose and non-rigid shape with particle dynamics. In CVPR ,
2015. 3
[2] Aayush Bansal, Minh V o, Yaser Sheikh, Deva Ramanan, and
Srinivasa Narasimhan. 4d visualization of dynamic events
from unconstrained multi-view videos. In CVPR , 2020. 2, 3
[3] Adrien Bartoli, Yan G ´erard, Francois Chadebecq, Toby
Collins, and Daniel Pizarro. Shape-from-template. T-PAMI ,
37(10), 2015. 3
[4] Mojtaba Bemana, Karol Myszkowski, Hans-Peter Seidel,
and Tobias Ritschel. X-ﬁelds: Implicit neural view-, light-
and time-image interpolation. TOG , 39(6), 2020. 3
[5] Chris Buehler, Michael Bosse, Leonard McMillan, Steven
Gortler, and Michael Cohen. Unstructured lumigraph ren-
dering. In SIGGRAPH , 2001. 1
[6] Zhiqin Chen and Hao Zhang. Learning implicit ﬁelds for
generative shape modeling. In CVPR , 2019. 2
[7] Ajad Chhatkuli, Daniel Pizarro, and Adrien Bartoli. Sta-
ble template-based isometric 3d reconstruction in all imag-
ing conditions by linear least-squares. In CVPR , 2014. 3
[8] Julian Chibane, Thiemo Alldieck, and Gerard Pons-Moll.
Implicit functions in feature space for 3d shape reconstruc-
tion and completion. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
6970–6981, 2020. 2
[9] Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H
Kim, and Jan Kautz. Extreme view synthesis. In CVPR ,
2019. 3
[10] John Flynn, Michael Broxton, Paul Debevec, Matthew Du-
Vall, Graham Fyffe, Ryan Overbeck, Noah Snavely, and
Richard Tucker. Deepview: View synthesis with learned gra-
dient descent. In CVPR , 2019. 3
[11] John Flynn, Ivan Neulander, James Philbin, and Noah
Snavely. Deepstereo: Learning to predict new views from
the world’s imagery. In CVPR , 2016. 3
[12] Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna,
and Thomas Funkhouser. Local deep implicit functions for
3d shape. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 4857–
4866, 2020. 2
[13] Rohit Girdhar, David F Fouhey, Mikel Rodriguez, and Ab-
hinav Gupta. Learning a predictable and generative vector
representation for objects. In ECCV , 2016. 2
[14] Richard Hartley and Andrew Zisserman. Multiple view ge-
ometry in computer vision . Cambridge university press,
2003. 1, 3
[15] Hanqing Jiang, Haomin Liu, Ping Tan, Guofeng Zhang, and
Hujun Bao. 3d reconstruction of dynamic scenes with mul-
tiple handheld cameras. In ECCV , 2012. 3[16] Abhishek Kar, Christian H ¨ane, and Jitendra Malik. Learning
a multi-view stereo machine. In NeurIPS , 2017. 3
[17] Diederik Kingma and Jimmy Ba. ADAM: A method for
stochastic optimization. In ICLR , 2015. 5
[18] Marc Levoy and Pat Hanrahan. Light ﬁeld rendering. In
SIGGRAPH , 1996. 1, 3
[19] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua,
and Christian Theobalt. Neural sparse voxel ﬁelds. arXiv
preprint arXiv:2007.11571 , 2020. 1, 2
[20] Shichen Liu, Shunsuke Saito, Weikai Chen, and Hao Li.
Learning to infer implicit surfaces without 3d supervision.
InNeurIPS , 2019. 2
[21] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel
Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural vol-
umes: learning dynamic renderable volumes from images.
TOG , 38(4), 2019. 2
[22] William E Lorensen and Harvey E Cline. Marching cubes:
A high resolution 3d surface construction algorithm. SIG-
GRAPH , 1987. 6
[23] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi,
Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duck-
worth. Nerf in the wild: Neural radiance ﬁelds for uncon-
strained photo collections. arXiv preprint arXiv:2008.02268 ,
2020. 1, 2
[24] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Se-
bastian Nowozin, and Andreas Geiger. Occupancy networks:
Learning 3d reconstruction in function space. In CVPR ,
2019. 2
[25] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light ﬁeld fusion: Practical view syn-
thesis with prescriptive sampling guidelines. TOG , 2019. 3
[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance ﬁelds for view syn-
thesis. arXiv preprint arXiv:2003.08934 , 2020. 1, 2, 4, 5,
6
[27] F. Moreno-Noguer and P. Fua. Stochastic exploration of am-
biguities for nonrigid shape recovery. T-PAMI , 35(2), 2013.
3
[28] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and
Andreas Geiger. Occupancy ﬂow: 4d reconstruction by
learning particle dynamics. In ICCV , 2019. 2
[29] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and
Andreas Geiger. Differentiable volumetric rendering: Learn-
ing implicit 3d representations without 3d supervision. In
CVPR , 2020. 1, 2
[30] Michael Oechsle, Michael Niemeyer, Lars Mescheder, Thilo
Strauss, and Andreas Geiger. Learning implicit surface light
ﬁelds. arXiv preprint arXiv:2003.12406 , 2020. 2
[31] Jeong Joon Park, Peter Florence, Julian Straub, Richard
Newcombe, and Steven Lovegrove. Deepsdf: Learning con-
tinuous signed distance functions for shape representation.
InCVPR , 2019. 2
[32] Julien Philip and George Drettakis. Plane-based multi-view
inpainting for image-based rendering in large scenes. In SIG-
GRAPH , 2018. 3
9

---

## Page 10

[33] Albert Pumarola, Stefan Popov, Francesc Moreno-Noguer,
and Vittorio Ferrari. C-ﬂow: Conditional generative ﬂow
models for images and 3d point clouds. In CVPR , 2020. 2
[34] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix
Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and
Aaron Courville. On the spectral bias of neural networks.
InICML , 2019. 4
[35] Konstantinos Rematas and Vittorio Ferrari. Neural voxel ren-
derer: Learning an accurate and controllable rendering tool.
InCVPR , 2020. 1
[36] Liyue Shen, Wei Zhao, and Lei Xing. Patient-speciﬁc recon-
struction of volumetric computed tomography images from a
single projection view via deep learning. Nature biomedical
engineering , 3(11), 2019. 3
[37] Vincent Sitzmann, Michael Zollh ¨ofer, and Gordon Wet-
zstein. Scene representation networks: Continuous 3d-
structure-aware neural scene representations. In NeurIPS ,
2019. 2
[38] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo
tourism: exploring photo collections in 3d. In SIGGRAPH ,
2006. 1
[39] Maxim Tatarchenko, Alexey Dosovitskiy, and Thomas Brox.
Octree Generating Networks: Efﬁcient Convolutional Archi-
tectures for High-resolution 3D Outputs. In ICCV , 2017. 2
[40] Carlo Tomasi and Takeo Kanade. Shape and motion from
image streams under orthography: a factorization method.
IJCV , 9(2), 1992. 3
[41] Bill Triggs, Philip F McLauchlan, Richard I Hartley, and An-
drew W Fitzgibbon. Bundle adjustment—a modern synthe-
sis. In International workshop on vision algorithms , 1999.
1, 3
[42] Shubham Tulsiani, Tinghui Zhou, Alexei A Efros, and Jiten-
dra Malik. A Point Set Generation Network for 3D Object
Reconstruction from a Single Image. In CVPR , 2017. 2
[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In NeurIPS , 2017. 4
[44] Sundar Vedula, Peter Rander, Robert Collins, and Takeo
Kanade. Three-dimensional scene ﬂow. IEEE transactions
on pattern analysis and machine intelligence , 27(3):475–
480, 2005. 2
[45] Peng-Shuai Wang, Yang Liu, Yu-Xiao Guo, Chun-Yu Sun,
and Xin Tong. O-CNN: Octree-based Convolutional Neural
Networks for 3D Shape Analysis. TOG , 36(4), 2017. 2
[46] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility
to structural similarity. TIP, 13(4), 2004. 7
[47] Qiangeng Xu, Weiyue Wang, Duygu Ceylan, Radomir
Mech, and Ulrich Neumann. Disn: Deep implicit surface
network for high-quality single-view 3d reconstruction. In
NeurIPS , 2019. 2
[48] Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo, and
Honglak Lee. Perspective Transformer Nets: Learning
Single-view 3D object Reconstruction without 3D Supervi-
sion. In NIPS , 2016. 2
[49] Lior Yariv, Matan Atzmon, and Yaron Lipman. Univer-
sal differentiable renderer for implicit neural representations.
arXiv preprint arXiv:2003.09852 , 2020. 2[50] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan
Atzmon, Basri Ronen, and Yaron Lipman. Multiview neu-
ral surface reconstruction by disentangling geometry and ap-
pearance. NeurIPS , 2020. 1, 2
[51] Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park,
and Jan Kautz. Novel view synthesis of dynamic scenes
with globally coherent depths from a monocular camera. In
CVPR , 2020. 3
[52] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR , 2018. 7
[53] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely. Stereo magniﬁcation: learning view syn-
thesis using multiplane images. TOG , 37(4), 2018. 3
[54] C Lawrence Zitnick, Sing Bing Kang, Matthew Uyttendaele,
Simon Winder, and Richard Szeliski. High-quality video
view interpolation using a layered representation. TOG ,
23(3), 2004. 3
10