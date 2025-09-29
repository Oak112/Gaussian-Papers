

---

## Page 1

Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields
Jonathan T. Barron1Ben Mildenhall1Dor Verbin1;2
Pratul P. Srinivasan1Peter Hedman1
1Google2Harvard University
Abstract
Though neural radiance ﬁelds (NeRF) have demon-
strated impressive view synthesis results on objects and
small bounded regions of space, they struggle on “un-
bounded” scenes, where the camera may point in any di-
rection and content may exist at any distance. In this set-
ting, existing NeRF-like models often produce blurry or
low-resolution renderings (due to the unbalanced detail and
scale of nearby and distant objects), are slow to train, and
may exhibit artifacts due to the inherent ambiguity of the
task of reconstructing a large scene from a small set of
images. We present an extension of mip-NeRF (a NeRF
variant that addresses sampling and aliasing) that uses a
non-linear scene parameterization, online distillation, and
a novel distortion-based regularizer to overcome the chal-
lenges presented by unbounded scenes. Our model, which
we dub “mip-NeRF 360” as we target scenes in which the
camera rotates 360 degrees around a point, reduces mean-
squared error by 57% compared to mip-NeRF , and is able to
produce realistic synthesized views and detailed depth maps
for highly intricate, unbounded real-world scenes.
Neural Radiance Fields (NeRF) synthesize highly realis-
tic renderings of scenes by encoding the volumetric density
and color of a scene within the weights of a coordinate-
based multi-layer perceptron (MLP). This approach has
enabled signiﬁcant progress towards photorealistic view
synthesis [30]. However, NeRF models the input to the
MLP using inﬁnitesimally small 3D points along a ray,
which causes aliasing when rendering views of varying res-
olutions. Mip-NeRF rectiﬁed this problem by extending
NeRF to instead reason about volumetric frustums along a
cone [3]. Though this improves quality, both NeRF and
mip-NeRF struggle when dealing with unbounded scenes,
where the camera may face any direction and scene con-
tent may exist at any distance. In this work, we present an
extension to mip-NeRF we call “mip-NeRF 360” that is ca-
pable of producing realistic renderings of these unbounded
scenes, as shown in Figure 1.
Applying NeRF-like models to large unbounded scenes
(a) mip-NeRF [3], SSIM=0.526 (b) Our Model, SSIM=0.804
Figure 1. (a) Though mip-NeRF is able to produce accurate ren-
derings of objects, for unbounded scenes it often generates blurry
backgrounds and low-detail foregrounds. (b) Our model produces
detailed realistic renderings of these unbounded scenes, as evi-
denced by the renderings (top) and depth maps (bottom) from both
models. See the supplemental video for additional results.
raises three critical issues:
1. Parameterization. Unbounded 360 degree scenes can
occupy an arbitrarily large region of Euclidean space,
but mip-NeRF requires that 3D scene coordinates lie
in a bounded domain.
2. Efﬁciency. Large and detailed scenes require more net-
work capacity, but densely querying a large MLP along
each ray during training is expensive.
3. Ambiguity. The content of unbounded scenes may lie
at any distance and will be observed by only a small
number of rays, exacerbating the inherent ambiguity
of reconstructing 3D content from 2D images.
Parameterization. Due to perspective projection, an ob-
ject placed far from the camera will occupy a small portion
of the image plane, but will occupy more of the image and
be visible in detail if placed nearby. Therefore, an ideal pa-
rameterization of a 3D scene should allocate more capacity
to nearby content and less capacity to distant content. Out-
side of NeRF, traditional view-synthesis methods address
this by parameterizing the scene in projective panoramic
5470


---

## Page 2

space [2, 4, 8, 14, 21, 24, 33, 42, 49] or by embedding scene
content within some proxy geometry [15, 23, 38] that has
been recovered using multi-view stereo.
One aspect of NeRF’s success is its pairing of speciﬁc
scene types with their appropriate 3D parameterizations.
The original NeRF paper [30] focused on 360 degree cap-
tures of objects with masked backgrounds and on front-
facing scenes where all images face roughly the same di-
rection. For masked objects NeRF directly parameterized
the scene in 3D Euclidean space, but for front-facing scenes
NeRF used coordinates deﬁned in projective space (normal-
ized device coordinates, or “NDC” [5]). By warping an
inﬁnitely deep camera frustum into a bounded cube where
distance along the z-axis corresponds to disparity (inverse
distance), NDC effectively reallocates the NeRF MLP’s ca-
pacity in a way that is consistent with the geometry of per-
spective projection.
However, scenes that are unbounded in alldirections,
not just in a single direction, require a different parame-
terization. This idea was explored by NeRF++ [46], which
used an additional network to model distant objects, and by
DONeRF [31], which proposed a space-warping procedure
to shrink distant points towards the origin. Both of these ap-
proaches behave somewhat analogously to NDC but in ev-
erydirection, rather than just along the z-axis. In this work,
we extend this idea to mip-NeRF and present a method for
applying any smooth parameterization to volumes (rather
than points), and also present our own parameterization for
unbounded scenes.
Efﬁciency. One fundamental challenge in dealing with
unbounded scenes is that such scenes are often large and
detailed . Though NeRF-like models can accurately repro-
duce objects or regions of scenes using a surprisingly small
number of weights, the capacity of the NeRF MLP saturates
when faced with increasingly intricate scene content. Ad-
ditionally, larger scenes require signiﬁcantly more samples
along each ray to accurately localize surfaces. For exam-
ple, when scaling NeRF from objects to buildings, Martin-
Brualla et al. [27] doubled the number of MLP hidden units
and increased the number of MLP evaluations by 8 . This
increase in model capacity is expensive — a NeRF already
takes multiple hours to train, and multiplying this time by
an additional40is prohibitively slow for most uses.
This training cost is exacerbated by the coarse-to-ﬁne re-
sampling strategy used by NeRF and mip-NeRF: MLPs are
evaluated multiple times using “coarse” and “ﬁne” ray inter-
vals, and are supervised using an image reconstruction loss
on both passes. This approach is wasteful, as the “coarse”
rendering of the scene does not contribute to the ﬁnal im-
age. Instead of training a single NeRF MLP that is super-
vised at multiple scales, we will instead train two MLPs: a
“proposal MLP” and a “NeRF MLP”. The proposal MLPpredicts volumetric density (but not color) and those densi-
ties are used to resample new intervals that are provided to
the NeRF MLP, which then renders the image. Crucially,
the weights produced by the proposal MLP are not super-
vised using the input image, but are instead supervised with
the histogram weights generated by the NeRF MLP. This al-
lows us to use a large NeRF MLP that is evaluated relatively
few times, alongside a small proposal MLP that is evaluated
many more times. As a result, our whole model’s total ca-
pacity is signiﬁcantly larger than mip-NeRF’s ( 15), re-
sulting in greatly improved rendering quality, but our train-
ing time only increases modestly ( 2).
We can think of this approach as a kind of “online dis-
tillation”: while “distillation” commonly refers to train-
ing a small network to match the output of an already-
trained large network [17], here we distill the structure of
the outputs predicted by the NeRF MLP into the proposal
MLP “online” by training both networks simultaneously.
NeRV [43] performs a similar kind of online distillation
for an entirely different task: training MLPs to approximate
rendering integrals for the purpose of modeling visibility
and indirect illumination. Our online distillation approach
is similar in spirit to the “sampling oracle networks” used
in DONeRF, though that approach uses ground-truth depth
for supervision [31]. A related idea was used in TermiN-
eRF [36], though that approach only accelerates inference
and actually slows training (a NeRF is trained to conver-
gence, and an additional model is trained afterwards). A
learned “proposer” network was explored in NeRF in De-
tail [1] but only achieves a speedup of 25%, while our ap-
proach accelerates training by 300%.
Several works have attempted to distill or “bake” a
trained NeRF into a format that can be rendered quickly
[16, 37, 45], but these techniques do not accelerate training.
The idea of accelerating ray-tracing through a hierarchical
data structure such as octrees [40] or bounding volume hi-
erarchies [39] is well-explored in the rendering literature,
though these approaches assume a-priori knowledge of the
geometry of the scene and therefore do not naturally gener-
alize to an inverse rendering context in which the geometry
of the scene is unknown and must be recovered. Indeed,
despite building an octree acceleration structure while opti-
mizing a NeRF-like model, the Neural Sparse V oxel Fields
approach does not signiﬁcantly reduce training time [25].
Ambiguity. Though NeRFs are traditionally optimized
using many input images of a scene, the problem of re-
covering a NeRF that produces realistic synthesized views
from novel camera angles is still fundamentally undercon-
strained — an inﬁnite family of NeRFs can explain away
the input images, but only a small subset produces accept-
able results for novel views. For example, a NeRF could
recreate all input images by simply reconstructing each im-
5471


---

## Page 3

age as a textured plane immediately in front of its respec-
tive camera. The original NeRF paper regularized ambigu-
ous scenes by injecting Gaussian noise into the density head
of the NeRF MLP before the rectiﬁer [30], which encour-
ages densities to gravitate towards either zero or inﬁnity.
Though this reduces some “ﬂoaters” by discouraging semi-
transparent densities, we will show that it is insufﬁcient for
our more challenging task. Other regularizers for NeRF
have been proposed, such as a robust loss on density [16]
or smoothness penalties on surfaces [32, 48], but these so-
lutions address different problems than ours (slow render-
ing and non-smooth surfaces, respectively). Additionally,
these regularizers are designed for the point samples used
by NeRF, while our approach is designed to work with the
continuous weights deﬁned along each mip-NeRF ray.
These three issues will be addressed in Sections 2, 3, and
4 respectively, after a review of mip-NeRF. We will demon-
strate our improvement over prior work using a new dataset
consisting of challenging indoor and outdoor scenes. We
urge the reader to view our supplemental video, as our re-
sults are best appreciated when animated.
1. Preliminaries: mip-NeRF
Let us ﬁrst describe how a fully-trained mip-NeRF [3]
renders the color of a single ray cast into the scene r(t) =
o+td, where oanddare the origin and direction of the ray
respectively, and tdenotes distance along the ray. In mip-
NeRF, a sorted vector of distances tis deﬁned and the ray is
split into a set of intervals Ti= [ti;ti+1). For each interval
iwe compute the mean and covariance (;) =r(Ti)of
the conical frustum corresponding to the interval (the radii
of which are determined by the ray’s focal length and pixel
size on the image plane), and featurize those values using
an integrated positional encoding:
(;) =(sin(2`) exp 
 22` 1diag()
cos(2`) exp 
 22` 1diag())L 1
`=0(1)
This is the expectation of the encodings used by NeRF with
respect to a Gaussian approximating the conical frustum.
These features are used as input to an MLP parameterized
by weights NeRF that outputs a density and color c:
8Ti2t;(i;ci) = MLP((r(Ti));  NeRF ):(2)
The view direction dis also provided as input to the MLP,
but we omit this for simplicity. With these densities and
colors we approximate the volume rendering integral using
numerical quadrature [28]:
C(r;t) =X
iwici; (3)
wi=
1 e i(ti+1 ti)
e P
i0<ii0(ti0+1 ti0)(4)where C(r;t)is the ﬁnal rendered pixel color. By con-
struction, the alpha compositing weights ware guaranteed
to sum to less than or equal to 1.
The ray is ﬁrst rendered using evenly-spaced “coarse”
distances tc, which are sorted samples from a uniform dis-
tribution spanning [tn;tf], the camera’s near and far planes:
tcU[tn;tf];tc= sort(ftcg): (5)
During training this sampling is stochastic, but during eval-
uation samples are evenly spaced from tntotf. After the
MLP generates a vector of “coarse” weights wc, “ﬁne” dis-
tances tfare sampled from the histogram deﬁned by tcand
wcusing inverse transform sampling:
tfhist(tc;wc);tf= sort(ftfg): (6)
Because the coarse weights wctend to concentrate around
scene content, this strategy improves sampling efﬁciency.
A mip-NeRF is recovered by optimizing MLP param-
eters NeRF via gradient descent to minimize a weighted
combination of coarse and ﬁne reconstruction losses:
X
r2R1
10Lrecon(C(r;tc);C(r)) +Lrecon(C(r;tf);C(r))(7)
whereRis the set of rays in our training data, C(r)is
the ground truth color corresponding to ray rtaken from an
input image, andLrecon is mean squared error.
2. Scene and Ray Parameterization
Though there exists prior work on the parameterization
ofpoints for unbounded scenes, this does not provide a
solution for the mip-NeRF context, in which we must re-
parameterize Gaussians . To do this, ﬁrst let us deﬁne f(x)
as some smooth coordinate transformation that maps from
Rn!Rn(in our case, n= 3). We can compute the linear
approximation of this function:
f(x)f() +Jf()(x ) (8)
Where Jf()is the Jacobian of fat. With this, we can
applyfto(;)as follows:
f(;) = 
f();Jf()Jf()T
(9)
This is functionally equivalent to the classic Extended
Kalman ﬁlter [19], where fis the state transition model.
Our choice for fis the following contraction:
contract( x) =(
x kxk1
2 1
kxk
x
kxk
kxk>1(10)
This design shares the same motivation as NDC: distant
points should be distributed proportionally to disparity (in-
verse distance) rather than distance. In our model, instead
5472


---

## Page 4

Figure 2. A 2D visualization of our scene parameterization. We
deﬁne a contract()operator (Equation 10, shown as arrows) that
maps coordinates onto a ball of radius 2 (orange), where points
within a radius of 1 (blue) are unaffected. We apply this contrac-
tion to mip-NeRF Gaussians in Euclidean 3D space (gray ellipses)
similarly to a Kalman ﬁlter to produce our contracted Gaussians
(red ellipses), whose centers are guaranteed to lie within a ball of
radius 2. The design of contract()combined with our choice to
space ray intervals linearly according to disparity means that rays
cast from a camera located at the origin of the scene will have
equidistant intervals in the orange region, as demonstrated here.
of using mip-NeRF’s IPE features in Euclidean space as per
Equation 1 we use similar features (see supplement) in this
contracted space: (contract( ;)). See Figure 2 for a
visualization of this parameterization.
In addition to the question of how 3D coordinates should
be parameterized, there is the question of how ray distances
tshould be selected. In NeRF this is usually done by sam-
pling uniformly from the near and far plane as per Equa-
tion 5. However, if an NDC parameterization is used, this
uniformly-spaced series of samples is actually uniformly
spaced in inverse depth (disparity). This design decision
is well-suited to unbounded scenes when the camera faces
in only one direction, but is not applicable to scenes that
are unbounded in all directions. We will therefore explicitly
sample our distances tlinearly in disparity (see [29] for a
detailed motivation of this spacing).
To parameterize a ray in terms of disparity we deﬁne an
invertible mapping between Euclidean ray distance tand a
“normalized” ray distance s:
s,g(t) g(tn)
g(tf) g(tn); t,g 1(sg(tf) + (1 s)g(tn));(11)
whereg()is some invertible scalar function. This gives us
“normalized” ray distances s2[0;1]that map to [tn;tf].
Throughout this paper we will refer to distances along a ray
in eithert-space ors-space, depending on which is more
convenient or intuitive. By setting g(x) = 1=xand con-
structing ray samples that are uniformly distributed in s-
space, we produce ray samples whose t-distances are dis-
tributed linearly in disparity (additionally, setting g(x) =
log(x)yields DONeRF’s logarithmic spacing [31]). In ourmodel, instead of performing the sampling in Equations 5
and 6 using tdistances, we do so with sdistances. This
means that, not only are our initial samples spaced lin-
early in disparity, but subsequent resamplings from indi-
vidual intervals of the weights wwill also be distributed
similarly. As can be seen from the camera in the cen-
ter of Figure 2, this linear-in-disparity spacing of ray sam-
ples counter-balances contract(). Effectively, we have co-
designed our scene coordinate space with our inverse-depth
spacing, which gives us a parameterization of unbounded
scenes that closely resembles the highly-effective setting of
the original NeRF paper: evenly-spaced ray intervals within
a bounded space.
3. Coarse-to-Fine Online Distillation
As discussed, mip-NeRF uses a coarse-to-ﬁne resam-
pling strategy (Figure 3) in which the MLP is evaluated
once using “coarse” ray intervals and again using “ﬁne” ray
intervals, and is supervised using an image reconstruction
loss at both levels. We instead train two MLPs, a “NeRF
MLP” NeRF (which behaves similarly to the MLPs used
by NeRF and mip-NeRF) and a “proposal MLP” prop.
The proposal MLP predicts volumetric density, which is
converted into a proposal weight vector ^waccording to
Equation 4, but does not predict color. These proposal
weights ^ware used to sample s-intervals that are then pro-
vided to the NeRF MLP, which predicts its own weight vec-
torw(and color estimates, for use in rendering an image).
Critically, the proposal MLP is not trained to reproduce the
Figure 3. A comparison of our model’s architecture with mip-
NeRF’s. Mip-NeRF uses one multi-scale MLP that is repeatedly
queried (only two repetitions shown here) for weights that are re-
sampled into intervals for the next stage, and supervises the ren-
derings produced at all scales. We use a “proposal MLP” that
emits weights (but not color) that are resampled, and in the ﬁnal
stage we use a “NeRF MLP” to produce weights and colors that
result in the rendered image, which we supervise. The proposal
MLP is trained to produce proposal weights ^wthat are consistent
with the NeRF MLP’s woutput. By using a small proposal MLP
and a large NeRF MLP we obtain a combined model with a high
capacity that is still tractable to train.
5473


---

## Page 5

input image, but is instead trained to bound the weights w
produced by the NeRF MLP. Both MLPs are initialized ran-
domly and trained jointly, so this supervision can be thought
of as a kind of “online distillation” of the NeRF MLP’s
knowledge into the proposal MLP. We use a large NeRF
MLP and a small proposal MLP, and repeatedly evaluate
and resample from the proposal MLP with many samples
(some ﬁgures and discussion illustrate only a single resam-
pling for clarity) but evaluate the NeRF MLP only once with
a smaller set of samples. This gives us a model that behaves
as though it has a much higher capacity than mip-NeRF but
is only moderately more expensive to train. As we will
show, using a small MLP to model the proposal distribu-
tion does not reduce accuracy, which suggests that distilling
the NeRF MLP is an easier task than view synthesis.
This online distillation requires a loss function that en-
courages the histograms emitted by the proposal MLP
(^t;^w)and the NeRF MLP (t;w)to be consistent. At ﬁrst
this problem may seem trivial, as minimizing the dissim-
ilarity between two histograms is a well-established task,
but recall that the “bins” of those histograms tand^tneed
not be similar — indeed, if the proposal MLP successfully
culls the set of distances where scene content exists, ^tand
twill be highly dissimilar. Though the literature contains
numerous approaches for measuring the difference between
two histograms with identical bins [11, 26, 35], our case is
relatively underexplored. This problem is challenging be-
cause we cannot assume anything about the distribution of
contents within one histogram bin: an interval with non-
zero weight may indicate a uniform distribution of weight
over that entire interval, a delta function located anywhere
in that interval, or myriad other distributions. We therefore
construct our loss under the following assumption: If it is
in any way possible that both histograms can be explained
using any single distribution of mass, then the loss must be
zero. A non-zero loss can only be incurred if it is impossi-
blethat both histograms are reﬂections of the same “true”
continuous underlying distribution of mass. See the supple-
ment for visualizations of this concept.
To do this, we ﬁrst deﬁne a function that computes the
sum of all proposal weights that overlap with interval T:
bound ^t;^w;T
=X
j:T\^Tj6=?^wj: (12)
If the two histograms are consistent with each other, then
it must hold that wibound ^t;^w;Ti
for all intervals
(Ti;wi)in(t;w). This property is similar to the additivity
property of an outer measure in measure theory [13]. Our
loss penalizes any surplus histogram mass that violates this
inequality and exceeds this bound:
Lprop 
t;w;^t;^w
=X
i1
wimax 
0;wi bound ^t;^w;Ti2;(13)This loss resembles a half-quadratic version of the chi-
squared histogram distance that is often used in statistics
and computer vision [35]. This loss is asymmetric because
we only want to penalize the proposal weights for under-
estimating the distribution implied by the NeRF MLP —
overestimates are to be expected, as the proposal weights
will likely be more coarse than the NeRF weights, and will
therefore form an upper envelope over it. The division by
wiguarantees that the gradient of this loss with respect to
the bound is a constant value when the bound is zero, which
leads to well-behaved optimization. Because tand^tare
sorted, Equation 13 can be computed efﬁciently through the
use of summed-area tables [10]. Note that this loss is in-
variant to monotonic transformations of distance t(assum-
ing that wand^whave already been computed in t-space)
so it behaves identically whether applied to Euclidean ray
t-distances or to normalized ray s-distances.
We impose this loss between the NeRF histogram (t;w)
and all proposal histograms (^tk;^wk). The NeRF MLP is
supervised using a reconstruction loss with the input im-
ageLrecon , as in mip-NeRF. We place a stop-gradient on
the NeRF MLP’s outputs tandwwhen computingLprop
so that the NeRF MLP “leads” and the proposal MLP “fol-
lows” — otherwise the NeRF may be encouraged to pro-
duce a worse reconstruction of the scene so as to make the
proposal MLP’s job less difﬁcult. The effect of this pro-
posal supervision can be seen in Figure 4, where the NeRF
MLP gradually localizes its weights waround a surface in
the scene, while the proposal MLP “catches up” and pre-
dicts coarse proposal histograms that envelope the NeRF
weights.
wi=(ti+1 ti)
Proposal 1
Proposal 2
NeRF
s
(a) 0% optimized (b) 4% optimized (c) 100% optimized
Figure 4. A visualization of the histograms (t;w)emitted from
the NeRF MLP (black) and the two sets of histograms (^t;^w)emit-
ted by the proposal MLP (yellow and orange) for a single ray from
our dataset’s bicycle scene over the course of training. Below we
visualize the entire ray with ﬁxed x and y axes, but above we crop
both axes to better visualize details near scene content. Histogram
weights are plotted as distributions that integrate to 1. (a) When
training begins, all weights are uniformly distributed with respect
to ray distance t. (b, c) As training progresses, the NeRF weights
begin to concentrate around a surface and the proposal weights
form a kind of envelope around those NeRF weights.
5474


---

## Page 6

4. Regularization for Interval-Based Models
Due to ill-posedness, trained NeRFs often exhibit two
characteristic artifacts we will call “ﬂoaters” and “back-
ground collapse”, both shown in Figure 5(a). By “ﬂoaters”
we refer to small disconnected regions of volumetrically
dense space which serve to explain away some aspect of
a subset of the input views, but when viewed from another
angle look like blurry clouds. By “background collapse”
we mean a phenomenon in which distant surfaces are incor-
rectly modeled as semi-transparent clouds of dense content
close to the camera. Here we presents a regularizer that, as
shown in Figure 5, prevents ﬂoaters and background col-
lapse more effectively than the approach used by NeRF of
injecting noise into volumetric density [30].
Our regularizer has a straightforward deﬁnition in terms
of the step function deﬁned by the set of (normalized) ray
distances sand weights wthat parameterize each ray:
Ldist(s;w) =1ZZ
 1ws(u)ws(v)ju vjdudv; (14)
where ws(u)is interpolation into the step function deﬁned
by(s;w)atu:ws(u) =P
iwi 1[si;si+1)(u). We use nor-
malized ray distances sbecause using tsigniﬁcantly up-
weights distant intervals and causes nearby intervals to be
effectively ignored. This loss is the integral of the dis-
tances between all pairs of points along this 1D step func-
tion, scaled by the weight wassigned to each point by the
NeRF MLP. We refer to this as “distortion”, as it resem-
bles a continuous version of the distortion minimized by
k-means (though it could also be thought of as maximizing
a kind of autocorrelation). This loss is minimized by setting
w=0(recall that wsums to no more than 1, not exactly
(a) noLdist (b) noLdist, w/noise [30] (c) withLdist
Figure 5. Our regularizer suppresses “ﬂoaters” (pieces of semi-
transparent material ﬂoating in space, which are easy to identify
in the depth map) and prevents a phenomenon in which surfaces
in the background “collapse” towards the camera (shown in the
bottom left of (a)). The noise-injection approach of Mildenhall et
al. [30] only partially eliminates these artifacts, and reduces recon-
struction quality (note the lack of detail in the depths of the distant
trees). See the supplemental video for more visualizations.
s0.00.10.20.30.4wws(·)
∇sLdist
∇wLdistFigure 6. A visualization of rLdist, the gradient of our regular-
izer, as a function of sandwon a toy step function. Our loss
encourages each ray to be as compact as possible by 1) minimiz-
ing the width of each interval, 2) pulling distant intervals towards
each other, 3) consolidating weight into a single interval or a small
number of nearby intervals, and 4) driving all weights towards zero
when possible (such as when the entire ray is unoccupied).
1). If that is not possible (i.e., if the ray is non-empty), it is
minimized by consolidating weights into as small a region
as possible. Figure 6 illustrates this behavior by showing
the gradient of this loss on a toy histogram.
Though Equation 14 is straightforward to deﬁne, it is
non-trivial to compute. But because ws()has a constant
value within each interval we can rewrite Equation 14 as:
Ldist(s;w) =X
i;jwiwjsi+si+1
2 sj+sj+1
2
+1
3X
iw2
i(si+1 si) (15)
In this form, our distortion loss is trivial to compute. This
reformulation also provides some intuition for how this loss
behaves: the ﬁrst term minimizes the weighted distances
between all pairs of interval midpoints, and the second term
minimizes the weighted size of each individual interval.
5. Optimization
Now that we have described our model components in
general terms, we can detail the speciﬁc model used in all
experiments. We use a proposal MLP with 4layers and
256hidden units and a NeRF MLP with 8layers and 1024
hidden units, both of which use ReLU internal activations
and a softplus activation for density . We do two stages of
evaluation and resampling of the proposal MLP each using
64 samples to produce (^s0;^w0)and(^s1;^w1), and then one
stage of evaluation of the NeRF MLP using 32 samples to
produce (s;w). We minimize the following loss:
Lrecon(C(t);C) +Ldist(s;w) +1X
k=0Lprop 
s;w;^sk;^wk
;(16)
averaged over all rays in each batch (rays are not included
in our notation). The hyperparameter balances our data
5475


---

## Page 7

termLrecon and our regularizer Ldist; we set= 0:01in
all experiments. The stop-gradient used in Lprop makes the
optimization of prop independent from the optimization
ofNeRF , and as such there is no need for a hyperparam-
eter to scale the effect of Lprop. ForLrecon we use Char-
bonnier loss [9]:p
(x x)2+2with= 0:001, which
achieves slightly more stable optimization than the mean
squared error used in mip-NeRF. We train our model (and
all reported NeRF-like baselines) using a slightly modiﬁed
version of mip-NeRF’s learning schedule: 250k iterations
of optimization with a batch size of 214, using Adam [22]
with hyperparameters 1= 0:9,2= 0:999,= 10 6,
a learning rate that is annealed log-linearly from 210 3
to210 5with a warm-up phase of 512iterations, and
gradient clipping to a norm of 10 3.
6. Results
We evaluate our model on a novel dataset: 9 scenes (5
outdoors and 4 indoors) each containing a complex cen-
tral object or area and a detailed background. During cap-
ture we attempted to prevent photometric variation by ﬁx-
ing camera exposure settings, minimizing lighting varia-
tion, and avoiding moving objects — we do not intend to
probe all challenges presented by “in the wild” photo col-
lections [27], only scale. Camera poses are estimated using
COLMAP [41], as in NeRF. See the supplement for details.
Compared methods. We compare our model with
NeRF [30] and mip-NeRF [3], both using additional po-
sitional encoding frequencies so as to bound the entire
scene inside the coordinate space used by both models.
We evaluate against NeRF++ [46], which uses two MLPs
to separately encode the “inside” and “outside” of each
scene. We also evaluate against a version of NeRF that
uses DONeRF’s [31] scene parameterization, which uses
logarithmically-spaced samples and a different contraction
from our own. We also evaluate against mip-NeRF and
NeRF++ variants in which the MLP(s) underlying each
model have been scaled up to roughly match our own
model in terms of number of parameter count (1024 hid-
PSNR"SSIM"LPIPS#Time (hrs) # Params
NeRF [12, 30] 23.85 0.605 0.451 4.16 1.5M
NeRF w/ DONeRF [31] param. 24.03 0.607 0.455 4.59 1.4M
mip-NeRF [3] 24.04 0.616 0.441 3.17 0.7M
NeRF++ [46] 25.11 0.676 0.375 9.45 2.4M
Deep Blending [15] 23.70 0.666 0.318 - -
Point-Based Neural Rendering [23] 23.71 0.735 0.252 - -
Stable View Synthesis [38] 25.33 0.771 0.211 - -
mip-NeRF [3] w/bigger MLP 26.19 0.748 0.285 22.71 9.0M
NeRF++ [46] w/bigger MLPs 26.39 0.750 0.293 19.88 9.0M
Our Model 27.69 0.792 0.237 6.89 9.9M
Our Model w/GLO 26.26 0.786 0.237 6.90 9.9M
Table 1. A quantitative comparison of our model with several prior
works using the dataset presented in this paper.den units for mip-NeRF, 512 hidden units for both MLPs in
NeRF++). We evaluate against Stable View Synthesis [38],
a non-NeRF model that represents the state-of-the-art of a
different view-synthesis paradigm in which neural networks
are trained on external scenes and combined with a proxy
geometry produced by structure-from-motion [41]. We ad-
ditionally compare with the publicly available SIBR imple-
mentations [7] of Deep Blending [15] and Point-Based Neu-
ral Rendering [23], two real-time IBR-based view synthesis
approaches that also depend on an external proxy geometry.
We also present a variant of our own model in which we use
the latent appearance embedding (4 dimensions) presented
in NeRF-W [6, 27] which ameliorates artifacts caused by
inconsistent lighting conditions during scene capture (be-
cause our scenes do not contain transient objects, we do not
beneﬁt from NeRF-W’s other components).
Comparative evaluation. In Table 1 we report mean
PSNR, SSIM [44], and LPIPS [47] across the test images
in our dataset. For all NeRF-like models, we report train
times from a TPU v2 with 32 cores [18], as well as model
size (the train times and model sizes of SVS, Deep Blend-
ing, and Point-Based Neural Rendering are not presented, as
this comparison would not be particularly meaningful). Our
model outperforms all prior NeRF-like models by a signif-
icant margin, and we see a 57% reduction in mean squared
error relative to mip-NeRF with a 2.17 increase in train
time. The mip-NeRF and NeRF++ baselines that use larger
MLPs are more competitive, but are 3slower to train
than our model and still achieve signiﬁcantly lower accu-
racies. Our model outperforms Deep Blending and Point-
Based Neural Rendering across all error metrics. It also
outperforms SVS for PSNR and SSIM, but not LPIPS. This
may be due to SVS being supervised to directly minimize an
LPIPS-like perceptual loss, while we minimize a per-pixel
reconstruction loss. See the supplement for renderings from
SVS that achieve lower LPIPS scores than our model de-
spite having reduced image quality [20]. Our model has
several advantages over SVS and Deep Blending in addi-
tion to image quality: those models require external train-
ing data while our model does not, those models require
the proxy geometry produced by a MVS package (and may
fail when that geometry is incorrect) while we do not, and
our model produces extremely detailed depth maps while
SVS and Deep Blending do not (the “SVS depths” we show
were produced by COLMAP [41] and are used as input to
the model). Figure 7 shows model outputs, though we en-
courage the reader to view our supplemental video.
Ablation study. In Table 2 we present an ablation study
of our model on the bicycle scene in our dataset, the ﬁndings
of which we summarize here. A) Removing Lprop signif-
icantly reduces performance, as the proposal MLP is not
5476


---

## Page 8

Depth Map
 Color Image
(a) Ground-Truth Test-Set Image (b) Our Model, SSIM=0.738 (c) SVS [38, 41], SSIM=0.680 (d) NeRF++ [46], SSIM=0.622 (e) mip-NeRF [3], SSIM=0.522
Figure 7. (a) A test-set image from our dataset’s stump scene, with (b) our model’s rendered image and depth map (median ray termination
distance [34]). Cropped patches are shown to highlight details. Compared to prior work (c-e) our renderings more closely resemble the
ground-truth and our depths look more plausible (though no ground-truth depth is available). See the supplement for more results.
supervised during training. B) Removing Ldistdoes not
substantially affect our metrics but results in “ﬂoater” ar-
tifacts in scene geometry, as shown in Figure 5. C) the reg-
ularization proposed by Mildenhall et al. [30] of injecting
Gaussian noise ( = 1) into density degrades performance
(and as shown in Figure 5 is less effective at eliminating
ﬂoaters). D) Removing the proposal MLP and using a sin-
gle MLP to model both the scene and the proposal weights
does not degrade performance but increases training time
by3, hence our small proposal MLP. E) Removing the
proposal MLP and training our model using mip-NeRF’s
approach (applying Lrecon at all coarse scales instead of
using ourLprop) worsens both speed and accuracy, jus-
tifying our supervision strategy. F) Using a small NeRF
MLP (256 hidden units instead of our 1024 hidden units)
accelerates training but reduces quality, demonstrating the
value of a high-capacity model when dealing with detailed
scenes. G) Removing IPE completely and using NeRF’s
positional encoding [30] reduces performance, showing the
value in building upon mip-NeRF instead of NeRF. H) Ab-
PSNR"SSIM"LPIPS#Time (hrs) # Params
A) NoLprop 20.49 0.406 0.573 6.21 9.0M
B) NoLdist 24.41 0.687 0.300 7.08 9.0M
C) NoLdist, w/Noise Injection 24.00 0.655 0.328 7.08 9.0M
D) No Proposal MLP 24.26 0.682 0.307 18.89 8.7M
E) No Prop. MLP w/[3]’s Training 23.45 0.659 0.328 18.89 8.7M
F) Small NeRF MLP 22.80 0.515 0.480 4.31 1.1M
G) No IPE 23.87 0.664 0.322 7.08 9.0M
H) No Contraction 23.77 0.642 0.347 8.79 10.9M
I) w/DONeRFs Contraction [31] 23.99 0.654 0.334 7.20 9.0M
Our Complete Model 24.37 0.687 0.300 7.09 9.0M
Table 2. An ablation study in which we remove or replace model
components to measure their effect. See the text for details.lating the contraction and instead adding positional encod-
ing frequencies to bound the scene decreases accuracy and
speed. I) Using the parameterization and logarithmic ray-
spacing presented in DONeRF [31] reduces accuracy.
Limitations. Though mip-NeRF 360 signiﬁcantly outper-
forms mip-NeRF and other prior work, it is not perfect.
Some thin structures and ﬁne details may be missed, such as
the tire spokes in the bicycle scene (Figure 5), or the veins
on the leaves in the stump scene (Figure 7). View synthesis
quality will likely degrade if the camera is moved far from
the center of the scene. And, like most NeRF-like models,
recovering a scene requires several hours of training on an
accelerator, precluding on-device training.
7. Conclusion
We have presented mip-NeRF 360, a mip-NeRF ex-
tension designed for real-world scenes with unconstrained
camera orientations. Using a novel Kalman-like scene pa-
rameterization, an efﬁcient proposal-based coarse-to-ﬁne
distillation framework, and a regularizer designed for mip-
NeRF ray intervals, we are able to synthesize realistic novel
views and complex depth maps for challenging unbounded
real-world scenes, with a 57% reduction in mean-squared
error compared to mip-NeRF.
Acknowledgements. Our sincere thanks to David Salesin
and Ricardo Martin-Brualla for their help in reviewing this
paper before submission, and to George Drettakis and Geor-
gios Kopanas for their help in evaluating baselines on our
360 dataset.
5477


---

## Page 9

References
[1] Relja Arandjelovi ´c and Andrew Zisserman. NeRF in detail:
Learning to sample for view synthesis. arXiv:2106.05264 ,
2021. 2
[2] Benjamin Attal, Selena Ling, Aaron Gokaslan, Christian
Richardt, and James Tompkin. MatryODShka: Real-
time 6DoF video view synthesis using multi-sphere images.
ECCV , 2020. 2
[3] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter
Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan.
Mip-NeRF: A Multiscale Representation for Anti-Aliasing
Neural Radiance Fields. ICCV , 2021. 1, 3, 7, 8
[4] Tobias Bertel, Mingze Yuan, Reuben Lindroos, and Christian
Richardt. OmniPhotos: Casual 360° VR photography. ACM
Transactions on Graphics , 2020. 2
[5] J.F. Blinn. A trip down the graphics pipeline: pixel coor-
dinates. IEEE Computer Graphics and Applications , 1991.
2
[6] Piotr Bojanowski, Armand Joulin, David Lopez-Pas, and
Arthur Szlam. Optimizing the latent space of generative net-
works. ICML , 2018. 7
[7] Sebastien Bonopera, Peter Hedman, Jerome Esnault, Sid-
dhant Prakash, Simon Rodriguez, Theo Thonat, Mehdi Be-
nadel, Gaurav Chaurasia, Julien Philip, and George Dret-
takis. sibr: A system for image based rendering, 2020. 7
[8] Michael Broxton, John Flynn, Ryan Overbeck, Daniel Er-
ickson, Peter Hedman, Matthew DuVall, Jason Dourgarian,
Jay Busch, Matt Whalen, and Paul Debevec. Immersive light
ﬁeld video with a layered mesh representation. SIGGRAPH ,
2020. 2
[9] Pierre Charbonnier, Laure Blanc-Feraud, Gilles Aubert, and
Michel Barlaud. Two deterministic half-quadratic regular-
ization algorithms for computed imaging. International Con-
ference on Image Processing , 1994. 7
[10] Franklin C Crow. Summed-area tables for texture mapping.
SIGGRAPH , 1984. 5
[11] Navneet Dalal and Bill Triggs. Histograms of oriented gra-
dients for human detection. CVPR , 2005. 5
[12] Boyang Deng, Jonathan T. Barron, and Pratul P. Srini-
vasan. JaxNeRF: an efﬁcient JAX implementation of NeRF,
2020. http://github.com/google-research/
google-research/tree/master/jaxnerf . 7
[13] Lawrence C Evans and Ronald F Garzepy. Measure theory
and ﬁne properties of functions . Routledge, 2018. 5
[14] Peter Hedman, Suhib Alsisan, Richard Szeliski, and Jo-
hannes Kopf. Casual 3D Photography. SIGGRAPH Asia ,
2017. 2
[15] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
George Drettakis, and Gabriel Brostow. Deep blending for
free-viewpoint image-based rendering. SIGGRAPH Asia ,
2018. 2, 7
[16] Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall,
Jonathan T. Barron, and Paul Debevec. Baking neural ra-
diance ﬁelds for real-time view synthesis. ICCV , 2021. 2,
3
[17] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the
knowledge in a neural network. arXiv:1503.02531 , 2015. 2[18] Norman P Jouppi, Cliff Young, Nishant Patil, David Patter-
son, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh
Bhatia, Nan Boden, Al Borchers, et al. In-datacenter per-
formance analysis of a tensor processing unit. International
Symposium on Computer Architecture , 2017. 7
[19] Rudolph E. Kalman. A new approach to linear ﬁltering and
prediction problems. Journal of Basic Engineering , 1960. 3
[20] Markus Kettunen, Erik H ¨ark¨onen, and Jaakko Lehtinen. E-
lpips: robust perceptual image similarity via random trans-
formation ensembles. arXiv:1906.03973 , 2019. 7
[21] Wesley Khademi and Jonathan Ventura. View synthesis in
casually captured scenes using a cylindrical neural radiance
ﬁeld with exposure compensation. ACM SIGGRAPH 2021
Posters , 2021. 2
[22] Diederik P. Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. ICLR , 2015. 7
[23] Georgios Kopanas, Julien Philip, Thomas Leimk ¨uhler, and
George Drettakis. Point-based neural rendering with per-
view optimization. Computer Graphics Forum , 2021. 2, 7
[24] Kai-En Lin, Zexiang Xu, Ben Mildenhall, Pratul P Srini-
vasan, Yannick Hold-Geoffroy, Stephen DiVerdi, Qi Sun,
Kalyan Sunkavalli, and Ravi Ramamoorthi. Deep multi
depth panoramas for view synthesis. ECCV , 2020. 2
[25] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and
Christian Theobalt. Neural sparse voxel ﬁelds. NeurIPS ,
2020. 2
[26] Subhransu Maji, Alexander C Berg, and Jitendra Malik.
Classiﬁcation using intersection kernel support vector ma-
chines is efﬁcient. CVPR , 2008. 5
[27] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi,
Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duck-
worth. NeRF in the Wild: Neural Radiance Fields for Un-
constrained Photo Collections. CVPR , 2021. 2, 7
[28] Nelson Max. Optical models for direct volume rendering.
IEEE TVCG , 1995. 3
[29] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local Light Field Fusion: Practical View
Synthesis with Prescriptive Sampling Guidelines. ACM
Transactions on Graphics (TOG) , 2019. 4
[30] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing scenes as neural radiance ﬁelds for view syn-
thesis. ECCV , 2020. 1, 2, 3, 6, 7, 8
[31] Thomas Neff, Pascal Stadlbauer, Mathias Parger, Andreas
Kurz, Joerg H. Mueller, Chakravarty R. Alla Chaitanya, An-
ton S. Kaplanyan, and Markus Steinberger. DONeRF: To-
wards Real-Time Rendering of Compact Neural Radiance
Fields using Depth Oracle Networks. Computer Graphics
Forum , 2021. 2, 4, 7, 8
[32] Michael Oechsle, Songyou Peng, and Andreas Geiger.
Unisurf: Unifying neural implicit surfaces and radiance
ﬁelds for multi-view reconstruction. ICCV , 2021. 3
[33] Ryan S Overbeck, Daniel Erickson, Daniel Evangelakos,
Matt Pharr, and Paul Debevec. A system for acquiring, pro-
cessing, and rendering panoramic light ﬁeld stills for virtual
reality. ACM Transactions on Graphics , 2018. 2
5478


---

## Page 10

[34] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Soﬁen
Bouaziz, Dan B Goldman, Steven M. Seitz, and Ricardo
Martin-Brualla. Nerﬁes: Deformable Neural Radiance
Fields. ICCV , 2021. 8
[35] Oﬁr Pele and Michael Werman. The quadratic-chi histogram
distance family. ECCV , 2010. 5
[36] Martin Piala and Ronald Clark. Terminerf: Ray termination
prediction for efﬁcient neural rendering. 3DV, 2021. 2
[37] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas
Geiger. Kilonerf: Speeding up neural radiance ﬁelds with
thousands of tiny mlps. ICCV , 2021. 2
[38] Gernot Riegler and Vladlen Koltun. Stable view synthesis.
CVPR , 2021. 2, 7, 8
[39] Steven M. Rubin and Turner Whitted. A 3-dimensional
representation for fast rendering of complex scenes. SIG-
GRAPH , 1980. 2
[40] Hanan Samet. The design and analysis of spatial data struc-
tures . Addison-Wesley, 1990. 2
[41] Johannes Lutz Sch ¨onberger and Jan-Michael Frahm.
Structure-from-motion revisited. CVPR , 2016. 7, 8
[42] Heung-Yeung Shum and Li-Wei He. Rendering with con-
centric mosaics. SIGGRAPH , 1999. 2
[43] Pratul P. Srinivasan, Boyang Deng, Xiuming Zhang,
Matthew Tancik, Ben Mildenhall, and Jonathan T. Barron.
NeRV: Neural reﬂectance and visibility ﬁelds for relighting
and view synthesis. CVPR , 2021. 2
[44] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image quality assessment: from error visibility
to structural similarity. IEEE TIP , 2004. 7
[45] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and
Angjoo Kanazawa. PlenOctrees for real-time rendering of
neural radiance ﬁelds. ICCV , 2021. 2
[46] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen
Koltun. NeRF++: Analyzing and Improving Neural Radi-
ance Fields. arXiv:2010.07492 , 2020. 2, 7, 8
[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. CVPR , 2018. 7
[48] Xiuming Zhang, Pratul P. Srinivasan, Boyang Deng, Paul
Debevec, William T. Freeman, and Jonathan T. Barron. NeR-
Factor: Neural Factorization of Shape and Reﬂectance Under
an Unknown Illumination. SIGGRAPH Asia , 2021. 3
[49] Ke Colin Zheng, Sing Bing Kang, Michael F Cohen, and
Richard Szeliski. Layered depth panoramas. CVPR , 2007. 2
5479
