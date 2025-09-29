

---

## Page 1

Dataset and Pipeline for Multi-View Light-Field Video
N. Sabater1, G. Boisson1, B. Vandame1, P. Kerbiriou1, F. Babon1,
M. Hog1,2, R. Gendrot1, T. Langlois1, O. Bureller1, A. Schubert1and V , Alli ´e1.
1Technicolor Research & Innovation
2INRIA
Name.Surname@technicolor.com
Abstract
The quantity and diversity of data in Light-Field videos
makes this content valuable for many applications such
as mixed and augmented reality or post-production in the
movie industry. Some of such applications require a large
parallax between the different views of the Light-Field, mak-
ing the multi-view capture a better option than plenoptic
cameras. In this paper we propose a dataset and a com-
plete pipeline for Light-Field video. The proposed algo-
rithms are specially tailored to process sparse and wide-
baseline multi-view videos captured with a camera rig. Our
pipeline includes algorithms such as geometric calibration,
color homogenization, view pseudo-rectiﬁcation and depth
estimation. Such elemental algorithms are well known by
the state-of-the-art but they must achieve high accuracy to
guarantee the success of other algorithms using our data.
Along this paper, we publish our Light-Field video dataset
that we believe may be of special interest for the commu-
nity [ 1]. We provide the original sequences, the calibration
parameters and the pseudo-rectiﬁed views. Finally, we pro-
pose a depth-based rendering algorithm for Dynamic Per-
spective Rendering.
1. Introduction
Since the introduction of the concept of integral pho-
tography [20], tremendous advances on Light-Fields have
been done on the computational photography community.
In particular, the availability of plenoptic cameras such as
Lytro1or Raytrix2has originated the bloom of new research
on the ﬁeld during the last years, being now a very ma-
ture topic. Besides plenoptic cameras, Light-Fields can
also be captured with camera arrays [ 33], robotic arms
[19] or hand held cameras [ 7]. However, each acquisi-
tion system samples the plenoptic function (light rays in
1www.lytro.com
2www.raytrix.dethe three-dimensional space) [ 2] very differently. Indeed,
plenoptic cameras produce great angular resolution at the
cost of reducing the spatial resolution. On the contrary,
multi-view systems have good spatial resolution but usu-
ally do not have many available views. Existing multi-view
Light-Field systems with a great number of views are gen-
erally impractical due to the amount of data and the com-
plexity of the capturing system. So, in general, plenop-
tic cameras capture dense andnarrow-baseline Light-Fields
and multi-view systems capture sparse and usually wide-
baseline Light-Fields. While all Light-Field acquisition
systems share the same theoretical principle, depending on
the application, one type of acquisition or another would be
preferred. Indeed, the baseline, the resolution and the num-
ber of views makes each acquisition system very speciﬁc
and suitable for the needs of a given application. As a con-
sequence, due to the data variability, processing algorithms
need to be speciﬁcally tailored for each acquisition system.
Besides the spatio-angular resolution, another particularity
of the Light-Field acquisition systems is the capacity to cap-
ture video.
In terms of applications, the availability of wide-baseline
Light-Field videos opens the door to new possibilities com-
pared to conventional cameras. For example, 3D Television
(3DTV), Free-view Television (FTV) [ 36], or mixed and
augmented reality as proposed by MagicLeap3or Microsoft
Hololens4. In particular, Light-Field videos are fundamen-
tal when the inserted content is not Computer Generated
(CG) and the goal is to produce a plausible and immersive
experience.
In this paper we focus on camera arrays as a video Light-
Field capturing system. In particular we present a 4×4syn-
chronized camera rig. Our system belongs to the multi-view
category and shares the same assumptions than [ 10] con-
cerning the captured scenes. This is, we assume to capture
Lambertian textured surfaces. However, we would like to
make the difference between the general multi-view frame-
3www.magicleap.com
4www.microsoft.com/microsoft-hololens
1
30


---

## Page 2

work in [ 10] and a Light-Field multi-view setup. The dif-
ference remains in the density of views (and the number of
light rays imaging each point of the scene) which requires
different algorithms in order to optimally process the data.
2. Related Work
Existing Light-Field datasets, [ 14] [33] [25] [18] [26],
either synthetic or from real acquisition systems (plenop-
tic cameras, camera arrays or gantries) are essentially still
Light-Fields. The only exception is the video Light-Field
dataset recently proposed by Dabala et al. [ 6] which turns
out to be the closest work to ours, since it also presents a
pipeline for camera rigs. Our pipeline, though, takes into
account color homogenization and the precise geometric
position of the cameras given by calibration, which allows
to relax some constraints on the depth estimation. The pa-
per of Basha et al. [ 3] also deals with multi-view video. The
authors propose to jointly estimate the 3D motion (scene
ﬂow) and the 3D reconstruction of the scene captured with
a camera rig assumed to be calibrated and having a small
baseline.
Pipelines for plenoptic cameras have also been proposed
[13] [29] [5] but due to the different nature of plenoptic
Light-Field data compared with camera rigs, the algorithms
are sorely different.
Geometric calibration is a well studied problem [ 12] but
it is generally not addressed by multi-view pipelines even
if it is of paramount importance for the accuracy of the se-
quel processing. Camera manufacturers do not provide this
information neither, specially when their cameras are used
to build camera rigs. Previous work on multi-camera cali-
bration includes [ 31] that studies a calibration method for
planar camera arrays and [ 34] that assumes a more gen-
eral camera setup but imposes a rigid constraint between
the viewpoints. Other techniques speciﬁcally developed for
Structure from Motion (SfM) such as Sparse Bundle Ad-
justment [ 21] can also provide multi-view calibration.
Regarding color calibration, when the cameras are not
known, a family of algorithms using image correspon-
dences allows to tonally stabilize videos [ 9] or to color ho-
mogenize different cameras of the same scene [ 32]. With
the same philosophy, [ 23] uses spatio-temporal correspon-
dences for multi-view video color correction. Nevertheless,
in our pipeline we exploit the fact that we have full knowl-
edge of our cameras and the capture setup. So we have an
approach more similar to [ 17] in which a method for cali-
brating large camera arrays is presented.
Camera arrays have more capabilities compared to con-
ventional images or video as it has been proved in a num-
ber of related papers. For instance, tracking through occlu-
sions [ 16], multi-object detection [ 28], reconstructing oc-
cluding surfaces [ 30] [27] or creating All-In-Focus images
[35]. Another application of multi-view systems concernSynthetic Aperture refocusing but the reduced number of
available views creates angular aliasing. In [ 15] an algo-
rithm for fast realistic refocusing for sparse Light-Fields is
presented.
All the above applications share the fact that they es-
timate and exploit the depth map of the captured scene.
More precisely, in [ 27], depth estimation is formulated as
an energy minimization problem with an intensity consis-
tency and a smoothness term. In [ 35] a Light-Field visibil-
ity term is also considered in the energy. In [ 30] different
cost functions for large camera arrays are compared in terms
of robustness to occlusions. It is interesting to point out that
most of the proposed methods estimate the depth for a view-
point that is not necessarily one of the available viewpoints
in the Light-Field. We have observed that this strategy is
more prone to errors and instead we estimate a depth map
for each available view-point in the Light-Field.
Finally, in Lu et al. [ 22] a survey on multi-view video
synthesis and editing is presented.
3. Pipeline
In this paper we consider the 2 plane parameterization
as in [ 11][19]. So, the 4D Light-Field L(s,t,u,v)is pa-
rameterized such that each view (s,t)has pixels (u,v). We
also consider that the light rays coming from the same scene
point should be captured with the same radiance in the dif-
ferent views when the object is Lambertian. This is, corre-
sponding pixels from different views should have the same
color. As a consequence, we have included in our pipeline
a color homogenization step. Besides, our camera rig has
carefully been calibrated. Calibration parameters are used
to project the images to a reference plane while removing its
distortion. We call such images pseudo-rectiﬁed images to
differentiate them from epipolar rectiﬁed images. Our strat-
egy has the advantage that point correspondences between
images can be found with simple translations without the
need to deproject in the space and project in a new view
each image point, which accelerates our pipeline consider-
ably.
Our pipeline also includes a depth estimation step, which
provides a depth for each camera. Our algorithm is multi-
scale and uses all images for estimating each depth map.
Finally, we propose a real time algorithm for Light-Field
rendering which estimates intermediate (virtual) views of
the light-ﬁeld, so the captured Light-Field sequence can be
watched with a dynamic parallax.
3.1. Capture
Our camera rig is made of 16 cameras whose sensors
are manufactured by CMOSIS (CMV2000) and packaged
by IDS. The 16 cameras are controlled through the UEye
API. Our multi-view Light-Field video is fully synchro-
nized. Fig. 1shows our camera setup.
31


---

## Page 3

Figure 1. Our camera rig setup.
3.2. Color homogenization and Demosaicking
LetRAW c: Ω⊂N2→N3be the c-th captured raw
image. In particular,
RAW c= (RAWr
c,RAWg
c,RAWb
c), (1)
whereRAWg
c(u,v) =RAWb
c(u,v) = 0 if(u,v)is a red
pixel in the Bayer pattern (and respectively for green and
blue pixels).
Our goal is to homogenize the color of all captured im-
ages with respect to a reference camera c0. In order to do
so, we describe here all the steps that need to be done before
and after capturing the sequences.
•Black level setting - The black level is a hardware pa-
rameter that allows to control the pixel sensitivities in
total darkness. It is important to tune this parameter for
each camera to correctly capture intensities in low light
conditions. Indeed, if the black level is set to 0, the sen-
sor looses information because it records an intensity
of0in dark scenes instead of low intensities. In order
to avoid this to happen, we set our camera rig in total
darkness (covering the cameras) and we increment the
black level of each camera until 95% of pixels record
an intensity different of 0. We have observed that after
this manipulation, the black level hardware parameter
of each camera is close to 4and all captured images in
total darkness have an intensity between 0and10.
•Bias map estimation - After the black level setting,
we capture a dozen of raw images in total darkness
for each camera c. Averaging such captured images
for each camera c, we obtain the bias map Bc, which
records the minimum count for each pixel.
•Aperture rig calibration - In order to calibrate the
apertures of the cameras, we ﬁrst set the desired aper-
ture on the reference camera c0. Afterwards, a ﬂat illu-
minated led panel covered with a diffuser (white scene)
is placed in front of the camera rig, so all cameras ob-
serve it while white raw images Wcare captured with
Figure 2. Left: Captured white scene with exposure time of 2ms.
Images are demosaicked for the sake of visualization. Before cor-
rection the cameras capture a greenish color with many differences
for each camera. Right: raw white images after bias and gain
map correction. The corrected intensities have been clipped to
[200,210] to better evaluate the similarity. Vignetting is also cor-
rected by applying the gain map.
the same exposure time. After subtracting the corre-
sponding bias map to the raw white images, the aver-
age intensity µcis computed:
µc=1
|Ω|/summationdisplay
(u,v)∈Ω/summationdisplay
ch=r,g,b/parenleftbig
Wch
c(u,v)−Bch
c(u,v)/parenrightbig
.
(2)
Finally, the aperture rings of the other cameras c/ne}ationslash=c0
are tuned untilµc
µc0= 1±0.02.
•Gain map estimation - When the apertures of all cam-
eras are homogeneous, we capture new raw white im-
ages for each camera Wc,i, i= 1,···,Nand we com-
pute the gain map as:
Gc(u,v) =1
NN/summationdisplay
i=1Wc,i(u,v)−Bc(u,v)
µc0. (3)
Then, the raw image corrected with the bias and gain
maps is computed as
RAW c=RAW c−Bc
Gc, (4)
The resulting image is vignetting-free and homoge-
neous in colors with the remaining images. Fig. 2
shows the 16 captured images with our rig of a white
scene before and after bias and gain correction. Note
that the intensity values of all cameras after correction
are very similar and independently of the color chan-
nel, meaning that all color channels are homogeneous.
However the bias and gain correction may not be sufﬁ-
cient to have homogeneous colors with a different ex-
position. Indeed, the gain map is estimated with a ref-
erence exposure time but not all cameras have the same
response with a different exposure. For this reason, we
have a last step in our color homogenization method.
•Color correction - In order to be robust to the different
exposures, we measure the average µch
cof each color
32


---

## Page 4

Figure 3. 16 patches side by side from all cameras corresponding
to 4 different colors in the MacBeth color chart.
channelchand each camera cfor different exposures
exp. Then, we estimate the regression line via least
squares ﬁtting of µch
c(exp)for eachchandc. Letαch
c
andβch
cbe the slope and offset of each regression line.
In this manner, the color corrected raw images are de-
ﬁned as
/hatwiderRAWch
c(u,v) =αch
c·RAWch
c(u,v)+βch
c.(5)
After color homogenization, the images /hatwiderRAW care de-
mosaicked using the algorithm in [ 8] which has proved to
outperform with respect to the state-of-the-art. In the se-
quel, the resulting demosaicked images are noted Icolor
c .
In order to measure the accuracy of our colorimetric cor-
rection we have captured a MacBeth color chart. After cor-
recting the images with the aforementioned processing, we
have measured the color average of 25×25homogeneous
patches for each color in the MacBeth color chart. We have
measured the standard deviation among all views. We have
observed that the red channel has a slightly less accurate
homogenization ( σr= 2.2compared to σg= 0.8and
σb= 0.9). See Fig. 3illustrating the color correction for
some of the MacBeth colors.
Note that the described method needs to be done once
for all. Then, the bias and gain maps, as well as the 16×3
slopes and offsets αandβare registered and used during
each capture. However, if the aperture has to be changed,
the homogenization of the aperture needs to be done again
before the capture. It is worth noting that the procedure for
color correction deﬁnes a linear correction which follows
the assumption of linear sensitivity of the pixels. It is also
interesting to point out that our method aims to homogenize
the colors and intensities of all cameras with respect to a
reference camera but we have not tried to calibrate our rig
to a referent illuminant. In the case that we require such
a calibration, it would be enough to calibrate the reference
camera with the desired illuminant before we run our ho-
mogenization method.3.3. Calibration and Geometry Processing
Our rig has carefully been arranged trying to place the
cameras in the same plane, having parallel principal axis
and being equidistant (same horizontal and vertical base-
line). However, the manual alignment not being perfect, a
calibration has been implemented in our pipeline. Intrin-
sic and extrinsic parameters are estimated with Sparse Bun-
dle Adjustement, based on the software package in [ 21].
The cameras are calibrated to ﬁt a distorted pinhole projec-
tion model similar to the one proposed in [ 4]. In particular,
the calibration module considers a set of corner pixel posi-
tions computed from several checkerboard captured images.
Considering a camera, we denote P= [R T]∈R3×4
the camera pose matrix in the World coordinate system and
Q= [R−1−R−1T]∈R3×4its extrinsic matrix. Now
ifXwis a 3D point in the World coordinate system and X
is the corresponding point in the camera coordinate system,
then Xw=P·/parenleftbigX
1/parenrightbig
andX=Q·/parenleftbigXw
1/parenrightbig
.
LetK=
f γ c u
0λ·f cv
0 0 1
be the intrinsic matrix of
the camera, where fis the distance from the optical cen-
ter to the sensor expressed in pixels, (cu,cv)is the principal
point,λis the aspect ratio, and γis the skew coefﬁcient.
LetWbe the distortion warping operator that affects 3D
points projections in the cameras coordinate system. The
radial distortion is expressed as a polynomial function in
the plane z= 1m as W/parenleftbigp
q/parenrightbig
= (1+a1r2+a2r4)/parenleftbigp
q/parenrightbig
, where
r=/radicalbig
p2+q2.
Then, given a 3D point Xwin the World coordinate sys-
tem, its projection in pixel coordinates (u,v)in the camera
image plane is determined by

u
v
1
=K·/parenleftbiggW/parenleftbigp
q/parenrightbig
1/parenrightbigg
(6)
where 
p
q
1
≡Q·/parenleftbigg
Xw
1/parenrightbigg
. (7)
Note that, using homogeneous coordinates

p
q
1
≡
x
y
z
⇐⇒/braceleftigg
p=x
z
q=y
z(8)
Image Pseudo-Rectiﬁcation:
After calibration, K,WandQare known for each camera c
which allows to determine for a given depth zcorrespondent
points in different images using Eq. 6and Eq. 7. However,
the projection and deprojection process has a high compu-
tational complexity due to the non linear distortion. In order
33


---

## Page 5

to accelerate our pipeline, our images are warped such that
corresponding points between images are found with simple
translations. This assumption stands in our setup because
our cameras are almost coplanar.
Formally, let Icolor
c be the original color images that
have been color corrected and demosaicked, where c=
(s,t);s,t= 0,...,3;is the camera index of the camera
placed at the s-th column and t-th row of the camera ar-
ray. From Icolor
c we compute the so-called pseudo-rectiﬁed
viewsIc, which are the projections onto a reference cam-
erac0at a reference depth z0of the original images Icolor
c .
More precisely, the pseudo-rectiﬁed images Icare deﬁned
at pixel(u,v)∈N2as
Ic(u,v) =Icolor
c(u′,v′), (9)
where 
u′
v′
1
=Kc·/parenleftbiggWc/parenleftbigp
q/parenrightbig
1/parenrightbigg
(10)
and

p
q
1
≡Qc·/parenleftbiggPc0
0 0 0 1/parenrightbigg
·
z0/hatwidestKc0−1
u
v
1

1
,
(11)
Wc,KcandQcbeing the distortion, the intrinsic and ex-
trinsic matrices of camera c,Pc0the pose matrix of the ref-
erence camera c0and/hatwidestKc0the intrinsic matrix of a virtual
pinhole camera derived from Kc0which skew coefﬁcient
and aspect ratio are respectively set to 0and1:
/hatwidestKc0=
f0cu
0f cv
0 0 1
. (12)
Note that in order to compute Icin Eq. 9, the images
Icolor
c need to be interpolated since (u′,v′)∈R2. In our
pipeline, a Lanczos kernel has been used for interpolation.
Note also that the projection at a reference depth z0of
each image Icolor
c does not ensure the image domains to be
equal. This is, (u′,v′)in Eq. 9may not belong to the image
domain of Icolor
c . In that case, empty pixels are colored
with pure green RGB= (0,255,0). Nevertheless, in order to
minimize the number of such empty pixels, the reference
depthz0has been set to an arbitrarily large distance from
the cameras.
Using Pseudo-Rectiﬁed Images:
With the notations above, let Zc0:N2→Rbe the
depth map of the reference camera c0. Then, given a pixel
(uc0,vc0)∈N2in the pseudo-rectiﬁed image Ic0, its cor-
respondent point (uc,vc)∈R2inIccan be found with asimple pixel translation:
/parenleftbigguc
vc/parenrightbigg
=/parenleftbigguc0
vc0/parenrightbigg
+D(uc0,vc0)·/parenleftbiggδuc
δvc/parenrightbigg
, (13)
whereD:N2→Ris deﬁned as
D(uc0,vc0) =1
Zc0(uc0,vc0)−1
z0
1
z1−1
z0, (14)
and(δuc,δv c)is the disparity shift which corresponds to the
shift in pixels between the projected point at depth z1/ne}ationslash=z0
and the projected point at the reference depth z0.
Thanks to the coplanar assumption, for each camera c,
the disparity shift (δuc,δv c)is constant over the whole im-
age. Nevertheless, if the cameras are not coplanar, Eq. 13
does not stand. For this reason, since our rig may not be
perfectly coplanar (Fig. 4-(a)), we have evaluated the differ-
ent pixels positions when computing exact pixel correspon-
dences via projection and deprojection (Eq. 6and Eq. 7)
and the approximate pixel correspondences via pixel trans-
lation (Eq. 13). We have measured that such difference is
only of 0.32 pixels in average at a distance of 1.9m, which
is totally acceptable, knowing the computational complex-
ity drop of using Eq. 13instead of Eq. 6and Eq. 7. Fig.
4-(b) shows the largest and the average error per camera.
Therefore, Synthetic Aperture Refocusing can be com-
puted easily. In particular, a refocused image at depth z
can be computed with the provided images Icfor each pixel
(u,v)∈N2as
Sz(u,v) =1
16/summationdisplay
cIc/parenleftbig
u+d(z)·δuc, v+d(z)·δvc/parenrightbig
,(15)
where
d(z) =1
z−1
z0
1
z1−1
z0. (16)
3.4. Depth Estimation
In order to estimate the depth, our pipeline has a multi-
resolution matching approach that estimates a depth map for
each image of the camera rig. The multi-resolution strategy
allows to compute accurately the depth maps in a fast man-
ner.
In this sense, the closest work to ours is [ 6]. However
our algorithm uses a different similarity measure and does
not impose a coherence matching among all views at each
scale. Compared to other existing depth estimation methods
[30][27][35], our approach is signiﬁcantly different since
depth estimates are not done for one single and virtual view.
In our experiments we have observed that this is a key factor
on the depth estimation quality.
34


---

## Page 6

(a)
(b)
Figure 4. (a) Shift in z with respect to the camera reference c0= 5.
The rig is almost coplanar since the biggest shift is 2.73mm. (b)
In the reference camera c0= 5, position differences (in pixels)
between corresponding exact points (Eq. 6and Eq. 7) and approx-
imate points (Eq. 13) . The largest errors are located in the border
of the images.
Correspondence matching:
Let us ﬁrst present the correspondence matching done at
each scale of our multi-resolution algorithm. So, we as-
sume that the images are at the current resolution. Now,
we consider the Zero-mean Normalized Cross-Correlation
(ZNCC) as the similarity measure. More precisely, we note
µ(I(u,v),n)the average of image Iin a squared neigh-
borhood of size (2n+ 1)2centered at (u,v),I(u,v) :=
I(u,v)−µ(I(u,v),n)andσ(I(u,v),n)the standard devi-
ation of image Iin the same neighborhood. We also deﬁne
/hatwideI(u,v,i,j) =I(u+i,v+j)−µ(I(u,v),n)
σ(I(u,v),n).(17)
Then, given a reference view c0, the ZNCC is deﬁned as
ZNCC(uc0,vc0,z) =
1
15(2n+1)2/summationdisplay
c/negationslash=c0n/summationdisplay
i,j=−n/hatwideIc0(uc0,vc0,i,j)·/hatwideIc(uc,vc,i,j).
(18)where(uc,vc) = (uc0+d(z)·δuc, vc0+d(z)·δvc).
With the notations above, depth estimation at each im-
age point (uc0,vc0)∈Ic0is performed minimizing the cost
function
Zc0(uc0,vc0) = argmin
z∈[zmin,zmax]ZNCC(uc0,vc0,z).(19)
Multi-Resolution strategy:
Multi-resolution is a well-know strategy in stereo matching
[24]. Here, we have considered a pyramid in which, by
deﬁnition, I(0)
c=Ic,∀c, and at each scale k= 0,...,K ,
the image I(k)
cis a downsampling of I(k−1)
c by a factor of
2. Now, if we aim to estimate the depth of a reference view
c0, we start estimating the depth Z(K)at the coarsest scale
Kusing Eq. 19(for the sake of simplicity we avoid writing
the index c0). The cost function is tested for all z=z(K)
min+
l·∆z(K);l= 0,...,L where∆z(K)= (z(K)
max−z(K)
min)/L.
Then, for the estimation of Z(K−1)we minimize again
Eq.19but using a different depth range depending on the
pixel position. Indeed, for each (u,v)∈I(K−1), we con-
sider the depth estimated values in the previous scale in a
given neighborhood
z(K)
i,j=Z(K)(u/2+i,v/2+j);i,j=−n,...,n (20)
and the depth ranges/bracketleftbig
z(K)
i,j−∆z(K)
2, z(K)
i,j+∆z(K)
2/bracketrightbig
. So,
our algorithm minimizes Eq. 19for all
z=z(K)
i,j−∆z(K)
2+m·∆z(K)
M,
i,j=−n,...,n;m= 0,...,M. (21)
The same reasoning is valid for the next scales until the
ﬁnest scale k= 0. In our implementation we have ﬁxed
a squared neighborhood of size 3×3(n=1), we consider
K= 4,L= 50 (subdivisions at the coarsest scale) and
M= 2 (subdivisions for the other scales). Note that the
initial depth range in the coarsest scale [z(K)
min,z(K)
max]varies
for each Light-Field sequence.
It is interesting to point out that our depth estimation
beneﬁts from the fact that the previous algorithms in our
pipeline are extremely accurate, so our simple but efﬁcient
algorithm produces precise depth estimates. Our pipeline
does therefore not include any particular ﬁltering of the
depth maps. Also, our video sequences are processed in-
dependently for each view and without temporal coherence
constraints. This strategy allows us to capture and process
in a very fast manner Light-Field videos.
35


---

## Page 7

Figure 5. Novel virtual view rendered at an intermediate position
of our camera rig.
3.5. Rendering
After depth estimation, a Multi-View plus Depth (MVD)
video is available. Different options for depth-based im-
age rendering are possible. While Synthetic Aperture Refo-
cusing has been proposed in the literature, when the Light-
Fields have been captured with a camera rig the resulting
images suffer from angular aliasing due to the poor angular
sampling. Instead, we believe that sparse Light-Fields are
better adapted to Dynamic Perspective Rendering. This is,
the estimated depth is used to render novel views different
from the captured available views. To this end the MVD
data is turned into a point cloud {Xw(c,u,v);Ic(u,v)}c,u,v
as follows:
Xw(c,u,v) =Pc·
Zc(u,v)·/hatwiderKc−1·
u
v
1

1
.(22)
Then the novel view is rendered by projecting the point
cloud onto a virtual pinhole camera deﬁned by its intrinsic
and extrinsic matrices KRandQR.
4. Dataset and Experimental Results
We provide a set of synchronized Light-Field video se-
quences captured by a 4×4camera rig at 30fps. Each cam-
era has a resolution of 2048×1088 pixels and a 12mm lens.
The Field Of View (FOV) is 50◦×37◦. Fig. 7shows one
camera image of one frame of the Light-Field sequences we
have captured. Our dataset has a number of close-ups se-
quences that are interesting for some speciﬁc use cases such
as realistic telepresence. Indeed, recovering 3D accurate in-
formation of faces is still a challenging problem because
very small errors may create unpleasant results. We have
also captured medium angle scenes ( Painter, Birthday ) and
other animated scenes where the movement does not come
from a human ( Automaton, Theater, Train ).
In our dataset, we consider the reference camera c0=
(1,1). This is s=t= 1. For each Light-Field sequence,we will provide the intrinsic matrix of the reference pinhole
camera/hatwidestKc0, the reference depth z0and the chosen depth z1
for which the shifts (δuc,δv c)in Eq. 13are computed.
For example, for the sequence Painter ,
/hatwidestKc0=
2340.14 0 1043 .09
0 2340 .14 480 .46
0 0 1
, (23)
we have chosen z0= 100 m andz1= 1.630m and Table
1shows the shifts (δuc,δu c). Note that, a different shift
table has to be computed for each sequence with different
calibration settings. Besides, given the geometric position
of our cameras in he rig and the fact that they are physically
well aligned, the computed shifts are close to be equispaced.
For example, if s= 0, the correspondent points at z=
z1have a row shift of respectively 98.28,98.14,98.07and
97.35pixels. And at any z, the corresponding points have a
row shift of respectively 98.28·d(z),98.14·d(z),98.07·d(z)
and97.35·d(z)pixels. Considering these shifts is more
accurate than considering a perfect epipolar rectiﬁcation of
the images.
st 0 1 2 3
0/parenleftbig100.00
98.28/parenrightbig/parenleftbig−0.36
98.14/parenrightbig/parenleftbig−97.19
98.07/parenrightbig/parenleftbig−195.55
97.35/parenrightbig
1/parenleftbig98.67
−1.73/parenrightbig/parenleftbig0.00
0.00/parenrightbig/parenleftbig−96.18
0.74/parenrightbig/parenleftbig−197.85
0.11/parenrightbig
2/parenleftbig99.17
−99.93/parenrightbig/parenleftbig0.21
−99.11/parenrightbig/parenleftbig−98.33
−101.12/parenrightbig/parenleftbig−197.00
−99.07/parenrightbig
3/parenleftbig99.08
−197.68/parenrightbig/parenleftbig−1.22
−198.14/parenrightbig/parenleftbig−99.26
−198.89/parenrightbig/parenleftbig−198.36
−199.37/parenrightbig
Table 1. Values of the shifts (δuc,δv c)for the sequence Painter
with the reference camera c= (1,1).
Fig. 6shows the depth maps for the ﬁrst frame of the
sequence Painter . The scene has many different objects
and a person walking on it. Fig. 8shows the point clouds
obtained with our pipeline for the sequences Face1 and
Rugby . In particular, since our rig has been calibrated, our
16 depth maps are all projected into a precise point cloud.
Our pipeline does not have a proper ﬁltering of the depth
maps. Instead, the only manipulation that has been done
in the point clouds is to remove completely isolated points
and points that have not been coherently estimated by at
least half of the cameras (8 cameras). While our camera
rig is not intended to provide complete 3D points clouds
of objects as 360-camera rigs would do, the visualization
of the point clouds from different viewpoints allows to as-
sess the accuracy of our depth estimates. Finally, Fig. 5
shows an image rendered from a virtual position different
from the camera positions of the camera rig using Eq. 22.
36


---

## Page 8

Figure 6. Depth maps for each camera using our pipeline. No ﬁltering has been done.
The rendering of such images allows to render the scene
with dynamic parallax.
Computational time Our ﬁrst goal is to implement an
accurate pipeline that precisely captures and manipulates
data. We have also implemented our pipeline in GPU to
meet the computational time requirements of some applica-
tions. In particular, our fast implementation captures data in
real time using the registered geometry and color calibration
parameters. Demosaicking is done with a linear algorithm
in this case. Depth estimation, the step with highest com-
plexity, is performed at 22fps at the full image resolution
(2048×1088 color images) on an NVidia GTX 1080 Ti and
at 32fps on a Nvidia Quadro P6000. Our image rendering
for dynamic parallax is achieved in real time in GPU.
5. Conclusion
In this paper we have presented a complete pipeline for
accurately capture and process Light-Fields. Our pipeline
is suitable for real-time applications. We have also created
a dataset available at [ 1] that is our major contribution and
we believe will be of interest for the scientiﬁc community.
At this moment we believe that one of the major chal-
lenges for the community is to address the problem of
Light-Field video compression. Indeed, many capturing
systems generating a huge amount of data and many appli-
cations having very constrained transmission requirements,
compression is of utmost importance for the technology to
become popular.
Furthermore, while image and video editing is a well-
known problem an many tools exist, not many solutions
to edit Light-Field video have been studied. Such methods
would need to handle a big amount of data and to guaranteeinter-view coherence to succeed.
References
[1] Technicolor light-ﬁeld dataset. http:
//www.technicolor.com/en/
innovation/scientific-community/
scientific-data-sharing/
light-field-dataset .1,8
[2] E. H. Adelson and J. R. Bergen. The plenoptic function and
the elements of early vision. Computational Models of Visual
Processing. Cambridge, MA: MIT Press, , 1991. 1
[3] T. Basha, S. Avidan, A. Hornung, and W. Matusik. Struc-
ture and motion from scene registration. In Computer Vision
and Pattern Recognition (CVPR), 2012 IEEE Conference on ,
pages 1426–1433. IEEE, 2012. 2
[4] J.-Y . Bouguet. www.vision.caltech.edu/
bouguetj/calib_doc .4
[5] D. Cho, M. Lee, S. Kim, and Y .-W. Tai. Modeling the cal-
ibration pipeline of the lytro camera for high quality light-
ﬁeld image reconstruction. In Proceedings of the IEEE Inter-
national Conference on Computer Vision , pages 3280–3287,
2013. 2
[6] Ł. Dabała, M. Ziegler, P. Didyk, F. Zilly, J. Keinert,
K. Myszkowski, H.-P. Seidel, P. Rokita, and T. Ritschel.
Efﬁcient multi-image correspondences for on-line light
ﬁeld video processing. In Computer Graphics Fo-
rum, volume 35-7, pages 401–410. Wiley Online Li-
brary, 2016. http://resources.mpi-inf.mpg.
de/LightFieldVideo/ .2,5
[7] A. Davis, M. Levoy, and F. Durand. Unstructured light ﬁelds.
InComputer Graphics Forum , volume 31, pages 305–314.
Wiley Online Library, 2012. 1
[8] J. Duran and A. Buades. A Demosaicking Algorithm with
Adaptive Inter-Channel Correlation. Image Processing On
Line, 5:311–327, 2015. 4
37


---

## Page 9

Face1
Face5
Face2
Rugby
Face3
Hands
Face4
Automaton
Train
Painter
Birthday
Theater
Figure 7. Reference images for the ﬁrst frame of the Technicolor Light-Field dataset.
38


---

## Page 10

Figure 8. Point clouds from different viewpoints using one frame of the Light-Field sequences Face andrugby . Background has been
removed for the sake of visualization.
[9] O. Frigo, N. Sabater, J. Delon, and P. Hellier. Motion driven
tonal stabilization. IEEE Transactions on Image Processing ,
25(11):5455–5468, 2016. 2
[10] Y . Furukawa, C. Hern ´andez, et al. Multi-view stereo: A tu-
torial. Foundations and Trends R/circlecopyrtin Computer Graphics and
Vision , 9(1-2):1–148, 2015. 1,2
[11] S. J. Gortler, R. Grzeszczuk, R. Szeliski, and M. F. Cohen.
The lumigraph. In Proceedings of the 23rd annual con-
ference on Computer graphics and interactive techniques ,
pages 43–54. ACM, 1996. 2
[12] R. Hartley and A. Zisserman. Multiple view geometry in
computer vision . Cambridge university press, 2003. 2
[13] M. Hog, N. Sabater, B. Vandame, and V . Drazic. An im-
age rendering pipeline for focused plenoptic cameras. IEEE
Transactions on Computational Imaging , 2017. To appear. 2[14] K. Honauer, O. Johannsen, D. Kondermann, and B. Gold-
luecke. A dataset and evaluation methodology for depth es-
timation on 4d light ﬁelds. In Asian Conference on Com-
puter Vision , pages 19–34. Springer, 2016. http://www.
lightfield-analysis.net .2
[15] C.-T. Huang, J. Chin, H.-H. Chen, Y .-W. Wang, and L.-G.
Chen. Fast realistic refocusing for sparse light ﬁelds. In
Acoustics, Speech and Signal Processing (ICASSP), 2015
IEEE International Conference on , pages 1176–1180. IEEE,
2015. 2
[16] N. Joshi, S. Avidan, W. Matusik, and D. J. Kriegman. Syn-
thetic aperture tracking: tracking through occlusions. In
Computer Vision, 2007. ICCV 2007. IEEE 11th International
Conference on , pages 1–8. IEEE, 2007. 2
[17] N. Joshi, B. Wilburn, V . Vaish, M. L. Levoy, and
M. Horowitz. Automatic color calibration for large camera
39


---

## Page 11

arrays . [Department of Computer Science and Engineering],
University of California, San Diego, 2005. 2
[18] C. Kim, H. Zimmer, Y . Pritch, A. Sorkine-Hornung, and
M. H. Gross. Scene reconstruction from high spatio-angular
resolution light ﬁelds. ACM Trans. Graph. , 32(4):73–
1, 2013. https://www.disneyresearch.com/
project/lightfields .2
[19] M. Levoy and P. Hanrahan. Light ﬁeld rendering. In Pro-
ceedings of the 23rd annual conference on Computer graph-
ics and interactive techniques , pages 31–42. ACM, 1996. 1,
2
[20] G. Lippmann. Epreuves reversibles donnant la sensation du
relief. J. Phys. Theor. Appl. , 7(1):821–825, 1908. 1
[21] M. I. Lourakis and A. A. Argyros. Sba: A software package
for generic sparse bundle adjustment. ACM Transactions on
Mathematical Software (TOMS) , 36(1):2, 2009. 2,4
[22] S. Lu, T. Mu, and S. Zhang. A survey on multiview video
synthesis and editing. Tsinghua Science and Technology ,
21(6):678–695, 2016. 2
[23] S.-P. Lu, B. Ceulemans, A. Munteanu, and P. Schelkens.
Spatio-temporally consistent color and structure optimiza-
tion for multiview video color correction. IEEE Transactions
on Multimedia , 17(5):577–590, 2015. 2
[24] B. D. Lucas, T. Kanade, et al. An iterative image registration
technique with an application to stereo vision. In DARPA
Image Understanding Workshop . Vancouver, BC, Canada,
1981. 6
[25] K. Marwah, G. Wetzstein, Y . Bando, and R. Raskar.
Compressive light ﬁeld photography using overcom-
plete dictionaries and optimized projections. ACM
Transactions on Graphics (TOG) , 32(4):46, 2013.
http://web.media.mit.edu/ ˜gordonw/
SyntheticLightFields/index.php .2
[26] A. Mousnier, E. Vural, and C. Guillemot. Par-
tial light ﬁeld tomographic reconstruction from a ﬁxed-
camera focal stack. arXiv preprint arXiv:1503.01903 ,
2015.https://www.irisa.fr/temics/demos/
lightField/index.html .2
[27] Z. Pei, Y . Zhang, X. Chen, and Y .-H. Yang. Synthetic aper-
ture imaging using pixel labeling via energy minimization.
Pattern Recognition , 46(1):174–187, 2013. 2,5
[28] Z. Pei, Y . Zhang, T. Yang, X. Zhang, and Y .-H. Yang. A novel
multi-object detection method in complex scene using syn-
thetic aperture imaging. Pattern Recognition , 45(4):1637–
1658, 2012. 2
[29] N. Sabater, M. Seiﬁ, V . Drazic, G. Sandri, and P. P ´erez. Accu-
rate Disparity Estimation for Plenoptic Images , pages 548–
560. Springer International Publishing, Cham, 2015. 2
[30] V . Vaish, M. Levoy, R. Szeliski, C. L. Zitnick, and S. B.
Kang. Reconstructing occluded surfaces using synthetic
apertures: Stereo, focus and robust measures. In Computer
Vision and Pattern Recognition, 2006 IEEE Computer Soci-
ety Conference on , volume 2, pages 2331–2338. IEEE, 2006.
2,5
[31] V . Vaish, B. Wilburn, N. Joshi, and M. Levoy. Using plane+
parallax for calibrating dense camera arrays. In ComputerVision and Pattern Recognition, 2004. CVPR 2004. Proceed-
ings of the 2004 IEEE Computer Society Conference on , vol-
ume 1, pages I–I. IEEE, 2004. 2
[32] J. Vazquez-Corral and M. Bertalm ´ıo. Color stabilization
along time and across shots of the same scene, for one or
several cameras of unknown speciﬁcations. IEEE Transac-
tions on Image Processing , 23(10):4564–4575, 2014. 2
[33] B. Wilburn, N. Joshi, V . Vaish, E.-V . Talvala, E. Antunez,
A. Barth, A. Adams, M. Horowitz, and M. Levoy. High per-
formance imaging using large camera arrays. In ACM Trans-
actions on Graphics (TOG) , volume 24, pages 765–776.
ACM, 2005. http://lightfield.stanford.edu .
1,2
[34] Y . Xu, K. Maeno, H. Nagahara, and R.-i. Taniguchi. Cam-
era array calibration for light ﬁeld acquisition. Frontiers of
Computer Science , 9(5):691–702, 2015. 2
[35] T. Yang, Y . Zhang, J. Yu, J. Li, W. Ma, X. Tong, R. Yu, and
L. Ran. All-in-focus synthetic aperture imaging. In Euro-
pean Conference on Computer Vision , pages 1–15. Springer,
2014. 2,5
[36] C. Zhang. Multiview imaging and 3dtv. IEEE signal pro-
cessing magazine , 24(6):10–21, 2007. 1
40
