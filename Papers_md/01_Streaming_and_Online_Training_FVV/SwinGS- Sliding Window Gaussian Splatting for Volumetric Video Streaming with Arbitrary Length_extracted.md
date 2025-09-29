

---

## Page 1

SwinGS: Sliding Windo w Gaussian Splatting for
V olumetric Vide o Str eaming with Arbitrar y Length
Bangya Liu
Univ ersity of Wisconsin-Madison
bangya@cs.wisc.e duSuman Banerje e
Univ ersity of Wisconsin-Madison
suman@cs.wisc.e du
Abstract
Re cent advances in 3D Gaussian Splatting (3DGS) hav e garner e d
significant attention in computer vision and computer graphics
due to its high r endering sp e e d and r emarkable quality . While
e xtant r esear ch has endeav or e d to e xtend the application of 3DGS
fr om static to dynamic scenes, such efforts hav e b e en consistently
imp e de d by e xcessiv e mo del sizes, constraints on vide o duration,
and content de viation. These limitations significantly compr omise
the str eamability of dynamic 3D Gaussian mo dels, ther eby r estrict-
ing their utility in v olumetric vide o str eaming.
This pap er intr o duces SwinGS , a str eaming-friendly paradigm
r epr esenting v olumetric vide o as a p er-frame-up date 3D Gaussian
mo del that could b e easily scale d to arbitrar y vide o length. Sp e cif-
ically , w e incorp orate a sliding-windo w base d continuous training
during the train stage as w ell as a straightfor war d r endering at
client side . W e implement a pr ototyp e of SwinGS  and demonstrate
its str eamability with D yNeRF dataset. A dditionally , w e de v elop
an interactiv e W ebGL vie w er enabling r eal-time v olumetric vide o
playback on most de vices with mo dern br o wsers, including smart-
phones and tablets. Exp erimental r esults sho w that SwinGS  r e duces
transmission costs by 83.6% compar e d to pr e vious w ork and could
b e easily scale d to v olumetric vide os with arbitrar y length with no
incr easing of r e quir e d GP U r esour ces.
K e y w or ds
3D Gaussian splatting, v olumetric vide o str eaming, neural r ender-
ing
A CM Refer ence Format:
Bangya Liu, and Suman Banerje e . 2023 SwinGS: Sliding Windo w Gaussian
Splatting for V olumetric Vide o Str eaming with Arbitrar y Length . In
XXXXX (XXXXX), Octob er 2–6, 2023, Madrid, Spain. 1  pages. https://doi.
org/10.1145/1234567890
1. Intr o duction
V olumetric vide o , also kno wn as fr e e vie w vide o (F V V), r epr esents
a r e v olutionar y me dia format that enables vie w ers to e xp erience
content as if physically pr esent in the scene . Unlike traditional
vide o captur e d fr om a single p ersp e ctiv e , v olumetric vide o encap-
sulates the depth, shap e , and motion of obje cts as w ell as p e ople
within a scene . This 3D r epr esentation can b e vie w e d fr om any
p osition or p ersp e ctiv e in virtual r eality (VR), augmente d r eality
( AR), or on flat scr e ens with user interaction. Be y ond entertain-
ment, v olumetric vide o plays crucial r oles in autonomous v ehicles
[ 40 ] , r ob otics vision, and tele op eration [ 30 ] .
Historically , v olumetric vide o has r elie d on p oint clouds [ 9 , 15 ]
and meshes as foundational elements. Ho w e v er , these appr oaches
hav e struggle d to balance vide o quality with storage and band-
width efficiency . Re cent advances in computer graphics hav e
intr o duce d a ne w family of 3D scene r epr esentations: neural r en-
dering. This includes Neural Radiance Fields (NeRF) [ 25 ]  and the
emerging 3D Gaussian Splatting (3DGS) [ 13 ] . While NeRF achie v essup erior r endering quality with compact storage , it suffers fr om
intensiv e computational costs due to its sampling pr o cess, r esult-
ing in lo w frame rates. The 3D Gaussian mo del, a “v olumetric and
differ entiable ” variant of p oint clouds, has emerge d as a pr omising
alternativ e .
Industr y demos of 3DGS, including T elep ort fr om V arjo [ 33 ]  and
Gracia [ 1 ] , hav e garner e d significant attention up on r elease . Mor e
r e cently , static 3D Gaussian Splatting has b e en sho w case d on the
Pico XR headset [ 2 ]  and Apple Vision Pr o via MetalSplatter [ 3 ] ,
offering impr essiv e immersiv e e xp eriences.
Re cent r esear ch w orks [ 6 , 10 , 12 , 18 , 23 , 31 , 34 , 36 – 38 ]  hav e
demonstrate d the p otential of 3DGS in r epr esenting dynamic 3D
scenes. Ho w e v er , significant gaps r emain b etw e en dynamic 3DGS
and a fully r ealize d 3D Gaussian-base d v olumetric vide o . Pr e vious
attempts hav e fallen short in thr e e ke y ar eas: (i) e xcessiv e mo del
sizes, (ii) limite d vide o duration, and (iii) lack of me chanisms to
handle content de viation acr oss e xtende d time spans. All thr e e ar e
crucial for str eaming v olumetric vide o fr om ser v er to client.
T o addr ess these challenges, w e pr op ose a no v el paradigm
r epr esenting v olumetric vide o as a dynamic 3D Gaussian mo del,
SwinGS , y et in a str eaming-friendly style compar e d with pr e vious
w ork. The mo del maintains a activ e set of Gaussians to r ender
each frame of the v olumetric vide o . It r etir es a subset of 3D Gaus-
sians fr om pr e vious frames and intr o duces ne w Gaussians for the
ne xt frame . Thr ough assigning each Gaussian an e xplicitly define d
lifespan indicating when it joins and leav es the mo del, the mo del
can b e easily adapte d to ne w content in the subse quent frames.
This solv es the content de viation issue (iii). On the other hand, the
training and transmission of a single bulky mo del ar e dismemb er e d
into training and transmission of conse cutiv e pie ces of data that
encapsulates p er-frame up date , making deriving and str eaming
of arbitrar y length vide os the or etically viable , hence addr essing
concerns (i) and (ii).
T o facilitate such paradigm, w e inno vativ ely pr op ose a sliding-
windo w base d continuous training appr oach. Within the windo w ,
Gaussians contributing to later frames will b e jointly optimize d
with Gaussians contributing to earlier frames. In this setup , Gaus-
sians ar e shar e d within a constraine d lo cal windo w achie ving
a compact r epr esentation. During sliding windo w mo ving fr om
the head to the tail of vide o se quence , the continuous training
appr oach guarante es that the r endering quality of earlier frames
ar e unaffe cte d during the optimization for later frames, in the
meantime largely r e duce the r e quir e d GP U r esour ces that use d to
hold bulky training frames in pr e vious dynamic 3DGS metho ds.
Sp e cifically , w e train the 3D Gaussian mo del using Sto chastic
Gradient Lange vin D ynamics (SGLD ) and Gaussian r elo cation, as
pr op ose d in 3DGS-MCMC [ 14 ] . This allo ws the mo del to adapt to
various contents acr oss differ ent frames, while ke eping a constant
numb er of Gaussians thr oughout training.
Our contributions can b e summarize d as follo ws:
1

---

## Page 2

• Instead of pr e vious w orks fo cusing on short vide o clips
mainly , our w ork tries to apply 3DGS onto long v olumetric
vide o str eaming, identifying its unique challenges.
• T argeting on those challenge , w e pr op ose SwinGS , a ne w
paradigm to r epr esent v olumetric vide o with a p er-frame-
up date 3D Gaussian mo del. W e also designe d and de v elop e d
a sliding-windo w base d continuous training appr oach to
facilitate the mo del training, as w ell as conv enient str eaming
and r endering at client side .
• W e implement and e valuate the pr op ose d SwinGS , demon-
strating its feasibility with D yNeRF datasets. W e also de v elop
a W ebGL-base d vie w er that enables easy playback of v olu-
metric vide o hoste d on the cloud storage . A liv e demo
is available at https://swingsplat.github .io/demo/ . Co debase
will b e op en-sour ce d on the acceptance of the pap er .
The organization of the pap er go es as follo ws: w e first pr o vide
a pr eliminar y backgr ound for 3DGS and conv entional v olumetric
vide o r epr esentation. Then in Se ction  3 , w e talk ab out ho w w e
tackles the challenges in the long v olumetric vide o se quence with
our pr op ose d metho d. Se ction  4  pr o vides a div e in of our imple-
mentation fr om a system p ersp e ctiv e . Finally , w e e valuate our
metho d in Se ction 5 .
2. Backgr ound
2.1. 3D Gaussian Splatting
3D Gaussian mo del is an emerging graphic primitiv es to r epr esent
a 3D scene , and 3D Gaussian splatting (3DGS) [ 13 ]  is the te chnique
r endering a mo del into 2D images with giv en camera p oses. In a
3D Gaussian mo del, the scene is r epr esente d with a set of Gaussian
p oints parameterize d by co variance Σ , center p osition 𝜇 , color 𝑐 ,
and opacity 𝛼 . The intensity of a 3D gaussian at a giv en lo cation 𝑥
in the 3D space could b e define d as:
𝐺 ( 𝑥 , 𝜇 , Σ ) = 𝑒−1
2( 𝑥 − 𝜇 )𝑇Σ− 1( 𝑥 − 𝜇 )(1)
In practice , Σ  is de comp ose d into a r otation matrix 𝑅  and a scaling
matrix 𝑆 , to guarante e semi-definiteness, as sho wn in Equation  2 .
Usually , 3D Gaussians could b e visualize d as ellipsoids in the 3D
space , in which case 𝑅  could b e interpr ete d as the orientation and
𝑆  as the length of axises of visualize d ellipsoid.
Σ = 𝑅 𝑆 𝑆𝑇𝑅𝑇(2)
3DGS is a rasterization-base d r endering te chnique . When it comes
to r endering, as sho wn by Figur e  1 , ray 𝑟  emitting fr om camera
center to the querie d pixel on the r endering image will trav erse
a subset of 3D Gaussians, and the final color 𝐶  of that pixel is
compute d by alpha blending this subset with the or der of z-depth
fr om close to far:
𝐶 ( 𝑟 ) = ∑
𝑖 ∈ 𝑁𝑐𝑖 𝛼′
𝑖 ∏𝑖 − 1
𝑗 = 1( 1 − 𝛼′
𝑗 ) (3)
Her e 𝛼′
𝑖  is determine d by multiplying opacity 𝛼𝑖  with the integra-
tion of 3D Gaussian 𝐺 ’s intensity along the ray . In practice ,
integration will b e p erforme d by sampling fr om a 2D Gaussian
pr oje cte d fr om the original 3D Gaussian.
2.2. D ynamic 3DGS
Re cent w orks hav e e xtende d vanilla 3D Gaussian Splatting (3DGS)
to dynamic scenes, primarily follo wing tw o typ e of appr oaches:
deformation field-base d and 4D primitiv e-base d metho ds.
Figur e 1: Rasterization pr o cess of 3DGS
Deformable 3DGS [ 38 ]  pione er e d the use of multilay er p er cep-
tr ons (MLPs) to implement deformation fields, allo wing 3D Gaus-
sians to e xhibit differ ent pr op erties acr oss time frames, follo wing
which, [ 36 ]  enhance d the deformation field’s fitting capacity by
intr o ducing a he xplane [ 5 ]  ahead of the MLP . Another early
stage w ork, D ynamicGS [ 23 ]  incorp orate d rigidity loss to impr o v e
tracking accuracy b etw e en frames. Se v eral r e cent w orks, including
SC-GS [ 10 ] , HiFi4G [ 12 ] , and Sup erp oint Gaussian [ 34 ] , pr op ose d
hierar chical structur es wher e higher-lay er Gaussians act as de-
formable skeletons, while lo w er-lay er Gaussians b ound to them fit
app earance . T aking a differ ent appr oach, SpacetimeGS [ 18 ]  use d
MLPs to enco de app earance and parametrize d p olynomials to r ep-
r esent deformation.
4D primitiv e-base d appr oaches incorp orate time as the fourth
dimension of a Gaussian, pione er e d by 4DGS [ 37 ]  and 4D-Roter
GS [ 6 ] . During r endering, these metho ds pr oje ct ( also calle d slice
or condition) 4D primitiv es into 3D Gaussians b efor e pr o cessing
them thr ough the alpha blending pip eline .
2.3. V olumetric Vide o Str eaming
T raditional primitiv es for str eamable v olumetric vide o include
p oint clouds and 3D meshes. Gr o ot [ 15 ]  pr esents a system that
le v erages an advance d o ctr e e-base d co de c to efficiently compr ess
p oint cloud payloads. Another w ork, ViV o [ 9 ] , fo cusing on p oint
clouds, r e duces bandwidth usage by considering visibility base d
on user vie wp ort. Mor e r e cently , MetaStr eam [ 8 ]  intr o duce d a
compr ehensiv e , p oint cloud-base d system for cr eating, deliv ering,
and r endering v olumetric vide os in an end-to-end fashion.
T ransco ding offers another p opular paradigm for v olumetric
vide o str eaming. V ues [ 22 ] , for instance , offloads r endering tasks
to an e dge ser v er instead of r endering r e ceiv e d 3D primitiv es on
the user side .
Ther e ar e some pr eliminar y attempt tr ying to r epr esent and
str eam v olumetric vide o with neural radiance field (NeRF) [ 20 , 21 ] .
Y et the use of 3D Gaussians as primitiv es for v olumetric vide o
str eaming r emains a largely une xplor e d field, with most dynamic
3DGS r esear ch fo cusing on short vide o clips lasting less than a
dozen se conds. The r esear ch most similar to our pr op ose d SwinGS
is 3DGStr eam [ 31 ] , which intr o duces a Neural T ransformation
Cache (N T C) as a p er-frame deformation field. A r e cent w ork,
MGA [ 32 ]  tr eats netw ork bandwidth as a constraint and optimizes
bitrate allo cation by tuning various enco ding parameters for 3DGS
frames to achie v e optimal o v erall r endering quality .
2

---

## Page 3

Primitiv e Rendering Quality Rendering Sp e e d A ccessibility Storage
3DGS [ 13 ] high high ( >100fps) me dium me dium
NeRF [ 25 ] high lo w ( <10fps) me dium lo w
p oint cloud p o or high easy me dium
mesh dep ends high p o or me dium
T able 1: Comparison b etw e en differ ent 3D primitiv es for v olumetric vide o
3. Motivation
3.1. Pitfall of Pr e vious Metho ds
Among various primitiv es r epr esenting 3D scenes, including NeRF
[ 25 ] , p oint cloud, and mesh, 3D Gaussian achie v es a balance
b etw e en r endering quality , sp e e d, accessibility , and storage cost,
as sho wn in T able  1 . While NeRF deliv ers high-quality r enderings
with r easonable storage , its computational demands limit r ender-
ing sp e e d. W orks including Str eamRF [ 16 ]  and NeRFP lay er [ 29 ]
attempte d to incr ease r endering sp e e d, but at a cost of dramatically
incr ease d mo del size . Meshes, although r elativ ely compact and
r endering-friendly , r e quir e e xtensiv e manual lab or to cr eate , esp e-
cially for high-quality v olumetric vide os. Point clouds offer de cent
frame rates and ar e easily accessible fr om 3D dense r e construction,
but their r endering quality corr elates with p oint count, leading to
high storage costs and challenges like hole filling.
Re cent r esear ch on dynamic Gaussian splatting [ 6 , 10 , 12 , 18 , 23 ,
31 , 34 , 36 – 38 ]  demonstrates the p otential of 3D Gaussian mo dels
for r epr esenting 3D scenes. Ho w e v er , these appr oaches r emain
incompatible with v olumetric vide o str eaming due to se v eral limi-
tations:
Excessiv e Mo del Size : A naiv e appr oach could b e constructing
static 3D Gaussian mo dels for each frame , which leads to substan-
tial traffic o v erhead. T o tackling with this, one of the baselines,
3DGStr eam [ 31 ] , emplo ys p er-frame Neural T ransformation Co des
(N T C) to transform Gaussians b etw e en frames. Ho w e v er , this still
incurs a storage o v erhead of appr o ximately 7.8MB/frame , r e quir-
ing a minimum bandwidth of 200MB/s for 30fps vide o . Other w orks
[ 10 , 12 , 34 , 36 , 38 ]  utilize a shar e d neural netw ork for Gaussian
deformation or app earance enco ding, along with initial Gaussians
for the first frame . While this r e duces storage costs, the entir e
mo del still o ccupies hundr e ds of megabytes [ 18 ]  and must b e trans-
mitte d at once . Any packet loss during transmission could blo ck
the entir e r endering pip eline , making it unsuitable for str eaming
applications.
Limite d Vide o Length : One barrier for scaling up the vide o
length is the linearly incr ease d GP U r esour ce r e quir ement. Pr e vi-
ous deformation base d metho ds [ 18 , 36 , 38 ]  traine d the whole set
of Gaussian with a random sampling dataloader on the whole set
of training vie ws which r e quir es a gr eat amount of GP U memor y
to pr eload them. Further , longer vide o se quence typically r e quir es
mor e Gaussians which also takes up a large fraction of GP U
r esour ces. Another barrier is the limite d capacity of a single neural
netw ork r esp onsible for Gaussian deformation. [ 18 ]  confirms that
incr easing clip length fr om 50 to 300 frames r esults in a PSNR dr op
fr om 29.48 to 29.17 for the F lame Salmon dataset of D yNeRF [ 17 ] .
Scenes with mor e substantial motion and longer duration ar e likely
to e xp erience mor e se v er e degradation, as the neural netw ork must
learn and r ememb er incr easingly div erse and distinct Gaussian
deformations acr oss frames.
Figur e 2: Differ ent Gaussian densification metho ds
( dotte d line r eferring to the shap e to fit)
Long-term Content De viation : Most pr e vious metho ds lack
me chanisms for intr o ducing ne w Gaussians when obje cts enter
the scene or r emo ving them when obje cts e xit. This limitation not
only r e duces r endering quality but also e xacerbates the mo del size
issue , as all Gaussians must b e transmitte d simultane ously . While
fragmentation with multiple dynamic Gaussian mo dels could p o-
tentially addr ess this concern, it intr o duces ne w challenges such
as visual discontinuities b etw e en fragments and do es not alle viate
the burst natur e of mo del transmission, in addition to e xtra storage
cost coming with multiple deformation netw orks.
3.2. 3DGS-MCMC for Bandwidth Shaping
For str eamable content, maintaining a uniform data v olume acr oss
time is crucial. Significant variations in the numb er of Gaussians
b etw e en differ ent time frames of a v olumetric vide o can compr o-
mise the quality of ser vice ( QoS) on the client side , esp e cially when
the user is in a high mobility scenario with constraine d netw ork
bandwidth.
Re cent w ork, 3DGS-MCMC [ 14 ] , intr o duces a no v el appr oach
to Gaussian densification during mo del optimization, as sho wn
in Figur e  2 . Instead of simply splitting one Gaussian into se v eral,
3DGS-MCMC r elo cates “ dead” Gaussians (those with lo w opacity )
to the p ositions of “aliv e ” Gaussians (those with high opacity hence
high pr esence in the scene). After that, parameters of b oth “ dead”
and “aliv e ” Gaussians ar e adjuste d in a way that the Gaussians
distribution ke eps appr o ximately consistent. With total numb er of
Gaussians ke eps constant after densification, this metho d allo ws
for pr e cise contr ol o v er the numb er of Gaussians.
Notably , this r elo cation op eration also naturally facilitates
smo oth transitions of Gaussian distributions b etw e en conse cutiv e
frames. Gaussians r epr esenting e xiting obje cts ar e optimize d into
“ dead” Gaussians with diminishing opacity and scale . In subse-
3

---

## Page 4

Figur e 4: O v er vie w of SwinGS
Figur e 3: Differ ent paradigm for dynamic 3D Gaussian mo del
( dotte d arr o w r eferring to the lifespan of each Gaussian)
quent iterations, these “ dead” Gaussians can b e r epurp ose d to
r epr esent ne wly app earing obje cts.
3.3. Sliding Windo w for Continuous T raining
The ke y differ ence b etw e en pr op ose d SwinGS  and pr e vious w ork
is visualize d in Figur e  3 . Pr e vious metho ds [ 4 , 10 , 18 , 37 , 38 ]
dominantly adopt a deformation-base d paradigm wher e a fixe d
set of Gaussians in the canonical space is deforme d by a car efully
designe d deformation netw ork at differ ent frame to fitting the dy-
namics in the vide o . Y et the limite d fitting capacities of deformation
netw ork and such tight coupling b etw e en frames makes the mo del
har d to train. Hence , complicate d mo dules including he xplane [ 5 ]
and r esfields [ 24 ]  ar e integrate d to comp ensate for that, usually
with an e xtra o v erhead of storage and transmission.
Y et our pr op ose d paradigm assigns a clear lifespan for each
Gaussian so that the optimization is always fo cusing on a short
snipp et instead of the whole vide o . W e could conv eniently opti-
mize those “temp oral-lo cal” Gaussians using a sliding windo w ,
then deriv e a str eamable mo del in an incr emental style . During the
windo w optimization, Gaussians who also contributes to out-of-
windo w frames ar e fr ozen to av oid degradation of pr e vious frames’
r endering quality . This combination of p er-frame-up date paradigm
and continuous training appr oach, makes training on long vide o
se quence feasible and also r e quir es much fe w er GP U r esour ces
compar e d with pr e vious metho ds.4. Design of SwinGS
4.1. O v er vie w
In this pap er , w e intr o duce SwinGS , a str eaming-friendly paradigm
r epr esenting v olumetric vide o as a p er-frame-up date 3D Gaussian
mo del. Figur e 4  pr o vides a o v er vie w .
SwinGS  b egins with multi-vie w vide o input, accompanie d by cor-
r esp onding camera p oses. The initial step inv olv es de comp osing
these vide os into individual frames, which ar e then cluster e d base d
on their frame inde x. This pr o cess yields a se quence of folders, each
corr esp onding to a sp e cific frame in the v olumetric vide o .
Follo wing frame clustering, w e train a 3D Gaussians mo del using
SGLD and Gaussian r elo cation to fit various frames within the
sliding windo w . For continuous training, w e fr e eze and ar chiv e a
small p ortion of Gaussians at the end of each windo w training, in
the meantime , inje ct some ne w optimizable Gaussians for the fu-
tural frames. Conse quently , w e allo w each Gaussian to contribute
to image r endering acr oss a small spanning of frames.
When it comes to vide o str eaming, similarly , w e only ne e d to
str eam and up date the small p ortion of Gaussians to up date the
mo del, which helps substantially r e duce bandwidth r e quir ements
and makes efficient str eaming of v olumetric vide o p ossible . Y et to
further r e duce the r e quir e d transmission bandwidth w e quantize
Gaussian attributes just like [ 7 , 26 ]  with minimal degradation of
the final r endering quality .
The follo wing tw o subse ctions will delv e into the details of the
training pr o cess and client r endering r esp e ctiv ely .
4.2. Continuous T raining
The general w orkflo w of mo del training is outline d in Algorithm  1 .
The pr o cess inv olv es an outer lo op  that shifts a sliding windo w
acr oss the vide o , iterating fr om [0, swin_size ) to the vide o ’s end,
and an inner lo op  trains on frames randomly sample d within this
sliding windo w . Her e , constant swin_size  r epr esents the windo w
length, which also defines the maximum lifespan of a Gaussian and
num_gs  r epr esents the maximum of Gaussians that will co e xist and
b e activ e in the mo del to r ender an image .
Each of the Gaussian uses tw o integers to indicates its lifespan:
“start” and “ e xpir e ” . The Gaussian will participate into r endering
only if the curr ent frame fall into its lifespan. During mo del
training, w e hav e tw o set of Gaussians in the GP U memor y , gs  and
matured , as sho wn in the Algorithm  1  as global variables. T o make
4

---

## Page 5

Algorithm 1 :  SwinGS Continuous T raining
1 global  gs , matur e d , str eam , trainset
2 const  swin_size , num_gs , relocate_period , iterations
3 pr o ce dur e  train_swin( st, e d):
4 for  iter in  range(iterations ):
5 gt = sample_frame_b etw e en( trainset , st, e d)
6 frame = gr ound_truth.frame
7 activ e_idx = gs .filter( start ≤ frame < expire )
8 activ e_ma_idx = matur e d .filter( start ≤ frame < expire )
9 activ e_gs = gs [ activ e_idx] + matur e d [ activ e_ma_idx]
10 pr e d = r ender( gt.cam, activ e_gs)
11 loss = loss_func( gt.image , pr e d) + r eg( activ e_gs)
12 loss.backwar d()
13 with  no_grad:
14 gs [ activ e_idx].param += 𝜆noise ∗ 𝜀
15 if  iter % relocate_period  == 0 then
16 r elo cate( gs [ activ e_idx])
17 pr o ce dur e  matur e( st):
18 matur e_idx = gs .filter( start < st)
19 str eam .write( gs [matur e_idx].detach())
20 matur e d  += gs [matur e_idx].detach()
21 matur e d  = matur e d [ -num_gs :]
22 gs [matur e_idx].birth = gs [matur e_idx].e xpir e
23 gs [matur e_idx].start = gs [matur e_idx].e xpir e
24 gs [matur e_idx].e xpir e = gs [matur e_idx].start + swin_size
25 pr o ce dur e  main:
26 gs [num_gs ].param = random()
27 gs [num_gs ].birth = 0
28 gs [num_gs ].start = 0
29 gs [num_gs ].e xpir e = swin_size
30 train_swin(0, swin_size )
31 sche dule_e xpir e( gs)
32 for  st in  range(1, trainset .total_frames):
33 matur e( st)
34 train_swin( st, st + swin_size )
35 matur e( trainset .total_frames)
Figur e 5: Continuous training with sliding windo w
it simple , gs  ar e those Gaussians w e ar e curr ently optimizing, usu-
ally helping fitting the content in the ne w frames, while matured
ar e those Gaussians that has b e en snapshotte d alr eady and ar e not
optimizable , contributing to the r endering of pr e vious frames as
w ell as curr ent frames. Whene v er ther e is a frame training, b oth
gs  and matured  will b e use d to r endering the image and deriv e
photometric loss y et ther e is no gradient for matured  Gaussians.Algorithm 2 :  SwinGS Str eaming and Rendering
1 global  frame , buffer , e v ents , str eam , user
2 const  swin_size , num_gs , FPS
3 pr o ce dur e  r ender_thr ead:
4 while  true:
5 activ e_idx = buffer .filter( start ≤ frame < expire )
6 r ender( user .cam, buffer [ activ e_idx])
7 pr o ce dur e  up date_thr ead:
8 while  true:
9 sle ep(1 0 0 0
FPS)
10 for  (target_frame , slice , up date) in  e v ents :
11 if  target_frame == frame  then
12 up date_first = slice ∗ slice_size
13 up date_last = ( slice+1) ∗ slice_size
14 buffer [up date_first:up date_last] = up date
15 br eak
16 frame  += 1
17 pr o ce dur e  main:
18 frame  = 0
19 buffer [:num_gs ] = str eam .r ead(num_gs )
20 slice_size = num_gs
swin_size
21 thr eading  r ender_thr ead
22 thr eading  up date_thr ead
23 while  true:
24 up date = str eam .r ead( slice_size)
25 target_frame = up date[0].birth
26 slice = target_frame % swin_size
27 e v ents .app end([target_frame , slice , up date])
Up on completing training within a sliding windo w , w e incr e-
ment b oth the start and end frames by one . W e then che ck if any
Gaussian’s lifespan b egins earlier than the windo w’s start frame .
If so , w e matur e that Gaussian and sav e its parameters.
4.2.1. Slice as the optimization granularity .  T o further accel-
erate this pr o cess, w e e v enly divide the entir e 3D Gaussians mo del
into se v eral smaller slices. Gaussians within the same slice shar e
identical lifespans and matur e together , while differ ent slices hav e
misaligne d lifespans. This arrangement ensur es that e xactly one
slice matur es in each frame .
Figur e  5  illustrates this slicing appr oach, with swin_size  set
to 5. Each bar r epr esents a gr oup of Gaussians with identical
lifespans. Within a slice , multiple bars ar e p ositione d in a bump er-
to-bump er style acr oss differ ent frames, as older Gaussians traine d
for pr e vious frames matur e and ne w optimizable Gaussians ar e in-
tr o duce d. Darker bars denote matur e d Gaussians, while white bars
r epr esent optimizable Gaussians. For training frame #N+1 within
the curr ent sliding windo w , matur e d Gaussians in slices #0, #1, #4,
and optimizable Gaussians in slices#2 and #3 participate in image
r endering. After mo del training within the [N, N+5) windo w , the
sketchy bar of slice #2 will matur e , b e cause ther e will b e no further
chance to optimize mo del on frame #N.
4.3. Real Time Str eaming and Rendering
Algorithm  2  illustrates the r endering pr o cess of the 3D Gaussian
mo del on the client’s de vice . At b eginning, the first swin_size
slices consisting of total num_gs  3D Gaussians is str eame d fr om
the r emote ser v er to the client de vice . Then it is loade d to GP U to
r ender the first frame .
5

---

## Page 6

Figur e 6: Per-frame slice up date in client side
Subse quent slices will b e r e ceiv e d, pr o cesse d, and buffer e d by
the client. Those slices will b e inserte d into the GP U memor y
r eplacing e xpir e d slices. The slice_size  in Algorithm  2  r epr esents
the numb er of Gaussians in each slice , while buffer  abstracts the
GP U memor y . The GP U memor y is divide d into swin_size  slices,
with each slice containing slice_size  Gaussians.
Figur e  6  depicts ho w the client de vice ’s GP U memor y is up date d,
in a slice by slice manner , as ne w str eaming data fr om the ser v er
is r e ceiv e d. For frame#N-1 slice #1 gets up date d, while slice #2 and
#3 get up date d at frame#N and frame#N+1 r e ceptiv ely .
T o de couple GP U r endering fr om str eaming data r e ceiving,
which is IO intensiv e , tw o de dicate d thr eads ar e instantiate d. Re-
ceiv e d slice will first b e buffer e d in CP U memor y , sp e cifically in the
events  queue . When it is the time to r endering its corr esp onding
frame , update_thread  will migrate the slice fr om CP U to GP U
memor y . render_thread  ke eps r endering images at the same FPS
as vanilla 3DGS with curr ent Gaussians in the GP U .
Compar e d to 3DGStr eam [ 31 ] , which r e calculates and r efr eshes
all Gaussians for e v er y frame , our appr oach significantly r e duces
b oth str eaming traffic and GP U op erations.
4.4. Implementation
Our demo is built up on the foundation of 3DGS-MCMC [ 14 ] . W e
e xtende d the original co debase by transitioning fr om a p er-frame
training appr oach to a p er-windo w training strategy , as pr op ose d
in Algorithm  1 . W e r efactor e d the GaussianModel  and Scene
classes to accommo date Gaussians. A ne w SwinManager  class was
implemente d to handle sliding windo w . T o incorp orate the p ertur-
bation r e quir e d by Sto chastic Gradient Lange vin D ynamics (SGLD )
[ 35 ] , w e intr o duce d scale d noise for optimizable activ e Gaussians
p ost-training for each frame . Our loss function adher es to the
practice establishe d in [ 14 ] , encompassing image quality measur e-
ments and r egularization terms for opacity 𝛼  and scaling 𝑆 .
4.5. Other Challenges
A dapting 3DGS-MCMC, originally designe d for static 3D r e con-
struction, to our cr oss-frame Gaussians pr esente d significant chal-
lenges. Follo ws ar e tw o primar y challenges.
4.5.1. A daptiv e gradient scaling for lr de cay .  During the train-
ing pr o ce dur e of vanilla 3DGS [ 13 ] , learning rate for Gaussians’
means is de cay e d along the iterations to achie v es a coarse to fine
optimization. Y et in our setup , Gaussians that hav e b e en traine d for
differ ent numb er of iterations ar e optimize d together . This makes
it har der to optimize each Gaussian and de cay their learning rater esp e ctiv ely considering Gaussians’ means is a single optimizable
parameter fr om the p ersp e ctiv e of the optimizer .
A s an alternativ e , w e do wnscale the gradient of each Gaussians’
means in accor dance to ho w many sliding windo ws the y hav e b e en
traine d for . It is mathematically e quivalent to learning rate do wn-
scaling for a SGD optimizer y et a cheap appr o ximate for A dam,
which has b e en pr o v e d to b e effe ctiv e in our practice .
4.5.2. D ynamic dataloader for training set.  The se cond major
challenge stemme d fr om the limite d GP U memor y available in our
testing envir onment. The original 3DGS co debase loads all training
set images into GP U memor y at the initiation of mo del training.
This appr oach b e comes infeasible when dealing with v olumetric
vide o containing numer ous image se quences fr om differ ent cam-
eras, each comprising hundr e ds or thousands of frames. T o addr ess
this constraint, w e applie d tw o ke y mo difications:
• W e de v elop e d a LazyCamera  class to load images in a lazy
manner , significantly r e ducing initial memor y r e quir ements
and impr o ving dataset loading sp e e d.
• T o manage memor y constraints when training with longer
sliding windo w sizes, w e maintain a maximum numb er of
frames in GP U memor y . This appr oach inv olv es dynamically
unloading and r eloading differ ent frames as ne e de d during
the training pr o cess.
4.6. W ebGL Vie w er
T o visually demonstrate the feasibility of our pr op ose d paradigm,
w e implemente d a w eb-base d vie w er building up on the op en-
sour ce pr oje ct https://github .com/antimatter15/splat , which was
originally designe d for r endering static 3D Gaussian mo dels in
br o wsers le v eraging W ebGL API. W e e xtende d this vie w er to
supp ort r endering the str eamable mo dels as pr op ose d by SwinGS .
Our mo difie d v ersion incorp orate d the p er-frame slice up dates
as pr op ose d in Algorithm  2 . The str eamable mo del is hoste d on
Huggingface , a p opular online platform for sharing and distribut-
ing machine learning mo dels. On the client side , whene v er a ne w
slice arriv es, the r endering set of Gaussians will b e sche dule d to b e
up date d with that ne w slice . Ho w e v er , due to the limitations of w eb
applications in dir e ctly manipulating GP U memor y , w e p erform
slice up dates on the Gaussians buffer within the r endering w orker .
5. Evaluation
W e compr ehensiv ely e valuate SwinGS  using D yNeRF [ 17 ]  follo w-
ing common practice of pr e vious w orks. For the metrics, w e
e valuate SwinGS  acr oss multiple dimensions, including r endering
quality , sp e e d, and traffic cost. Implementation details could b e
che cke d in Se ction 8  in the supplementar y material.
5.1. O v erall comparison
W e compar e SwinGS  with pr e vious w orks that utilize neural r en-
dering to r e construct dynamic 3D scenes, as sho wn in T able  2 . W e
e valuate d each scenes fr om the D yNeRF [ 17 ]  dataset for 300 frames
as b enchmarks. W e set swin_size =5 with num_gs =200K, up dating
40K Gaussians p er frame .
SwinGS  surpasses NeRF-base d metho ds like [ 16 ]  and achie v es
comparable r endering quality to 3DGStr eam [ 31 ] . While our PSNR
is slightly lo w er than SpacetimeGS [ 18 ] , SwinGS  e xcels in balanc-
ing high str eamability with lo w p er-frame storage r e quir ements.
Unlike baseline metho ds that r e quir e compulsor y neural netw ork
infer ence for numer ous Gaussians, our appr oach primarily incurs
6

---

## Page 7

Metho d Coffe e
MartiniCo ok
SpinachCut
Be efF lame
SalmonF lame
SteakSear
SteakA vg. FPS Storage Long
Se quence
4DGaussians [ 36 ] 27.34 32.46 32.90 29.20 32.51 32.49 31.15 30 0.3MB ✗
SpacetimeGS [ 18 ] 28.61 33.18 33.72 29.48 33.64 33.89 32.05 140 1MB ✗
Str eamRF [ 16 ] 27.84 31.59 31.81 28.26 32.24 32.36 28.26 10 17.7MB ✓
3DGStr eam [ 31 ] 27.75 33.31 33.21 28.42 34.30 33.01 31.67 215 7.8MB ✓
Ours 27.99 33.66 34.03 28.24 32.94 33.32 31.69 300 2.1MB ✓
Ours ( quantize d) 27.70 33.42 33.77 28.17 32.93 32.88 31.47 300 1.2MB ✓
T able 2: O v erall comparison with other neural r endering metho ds
Evaluate on D yNeRF Dataset, PSNR as metrics.
Figur e 7: T raining with various setup but constraint bandwidth
Lab el r efers to num_gs :swin_size
costs fr om GP U memor y op erations, enabling our mo del to achie v e
the b est r endering FPS.
5.2. Sliding windo w for constraine d bandwidth
The ke y design parameters in SwinGS  ar e swin_size  and num_gs .
The former determines ho w many conse cutiv e frames a Gaussian
participates in, while the latter defines the total numb er of Gaus-
sians use d to fit a single frame . A larger swin_size  typically
r esults in higher degr e e of content sharing among frames, which
sav es bandwidth. Conv ersely , mor e num_gs  can p otentially pr o vide
b etter detail in the r ender e d image , alb eit at the cost of incr ease d
traffic and storage . The bandwidth r e quir e d for str eaming a mo del
can b e formate d as:
BW = FPSvideo ×num_gs
swin_size× 𝑁bytes/GS (4)
  This implies that the actual traffic cost is pr op ortional to
slice_size =num_gs
swin_size, giv en a fixe d vide o FPS and storage cost for
each individual Gaussian. Further , Figur e  7  demonstrates the effe c-
tiv eness for sliding windo w me chanism under a constraint trans-
mission bandwidth. With an appr opriate swin_size , our metho d
could b o ost the r endering quality with shar e d Gaussian primitiv es
acr oss frames.
5.3. O v erhead for long v olumetric vide o training
A s sho wn in Figur e  3 , the ke y differ ence b etw e en SwinGS  and pr e-
vious w ork is that: in pr e vious w ork, all the Gaussians of the mo del
will contribute to the r endering of each frame in a de eply couple d
way ( canonical + deformation), while in SwinGS , each Gaussian
only binds to frames within a small windo w . Combine d with the
dynamic dataloader w e pr op ose d in Se ction  4.5.2 , this differ enceNumb er of Frames 10 30 50 100
SpacetimeGS [ 18 ] 7.46 14.37 20.57 OOM
Ours 2.68 2.61 2.71 2.65
T able 3: GP U memor y utilization ( GB)
when training with differ ent vide o length
gr eatly r e duce the r e quir e d GP U r esour ces when training the
mo del, considering at one time , only a small p ortion of the training
vie ws as w ell as a small subset of all the Gaussians ne e de d to b e
loade d into GP U memor y and contributes to back pr opagation.
W e pr ofile the GP U memor y utilization when training the mo del
with differ ent numb er of total frames in the vide o se quence . T able  3
sho ws that, pr e vious metho d [ 18 ]  r e quir es a linearly incr easing
GP U memor y while SwinGS  only r e quir es a fixe d amount of r e-
sour ces. Pr ofiling is p erforme d in a N VIDIA 4090 GP U . This vide o
length invariant r esour ce consumption is crucial when w e want to
train our mo del for a v olumetric vide o with arbitrar y length.
5.4. Efficiency for client side r endering
The primar y cost intr o duce d by SwinGS , compar e d to vanilla 3DGS,
is the memor y op eration to r eplace e xpir e d Gaussians with ne w
ones in the GP U . This allo ws for a r endering sp e e d of o v er 300
FPS when w e use the vanilla differ entiable Gaussian rasterization
mo dule [ 13 ]  to r ender the 3D scene .
Ho w e v er , when str eaming v olumetric vide o with our W ebGL
vie w er , the scenario b e comes a little bit comple x. T ypically , after
chunks of raw byte str eams ar e r ead fr om the cloud host, se v eral
steps ar e r e quir e d b efor e r endering the str eaming data into images:
raw data pr epr o cessing, depth sorting, and te xtur e generation.
The first step inv olv es parsing the binar y raw data into Gaussian
obje cts. Depth sorting arranges all Gaussians accor ding to their
distance fr om the camera center , fr om close to far . The thir d step ,
te xtur e generation, is ne cessar y b e cause 3D Gaussians cannot b e
dir e ctly r ender e d by W ebGL; the y must b e conv erte d into te xtur e
data b efor e b eing deliv er e d to the shader for r endering. All steps
intr o duces additional o v erhead to the r endering latency .
Figur e  8  visualizes the latency comp osition when r endering v ol-
umetric vide o on a laptop . Raw data pr o cessing generally takes less
than 1 ms, while depth sorting and te xtur e generation take longer ,
ranging fr om 5ms to 18ms. Ther e is a clear corr elation b etw e en the
numb er of Gaussians and the time r e quir e d for sorting and te xtur e
generation. This is r easonable considering our W ebGL implemen-
tation do es not manipulate the GP U dir e ctly and r etransmits the
te xtur e data corr esp onding to all activ e Gaussians to the shader as
7

---

## Page 8

Figur e 8: Rendering latency de comp osition for W ebGL vie w er
Stages pr epr o c sort te xtur e o v erall
MacBo ok pr o (M3 pr o) 3.00 5.81 18.46 27.27
iP hone ( A18 pr o) 6.00 4.44 19.90 30.34
iPad (M1) 6.48 4.94 22.29 33.71
Pixel (Snap dragon 765) 13.02 17.37 49.59 79.98
T able  4: Rendering latency de comp osition on differ ent de vices (ms)
a whole for e v er y ne w frame . When configur e d with swin_size  as
4 and num_gs  as 200K, it takes appr o ximately 34ms to complete the
full pip eline for one frame , r esulting in a w orst-case vide o frame
rate of ar ound 30 FPS for serialize d computation. Considering the
thr e e stages could b e fully parallele d among vide o frames, w e
e xp e ct an optimize d v ersion achie ving o v er 60fps for vide o frame
rate with te xtur e generation as b ottlene ck stage for 18ms for our
W ebGL vie w er .
T able  4  further pr ofiles the W ebGL vie w er on a wide range of
mobile de vices fr om p erformance laptop to p ortable smartphones.
Most mo dern de vices ar e capable of vide o playback with 30fps.
5.5. A daptiv e Bitrate Str eaming
num_gs  and swin_size  ar e contr ollable parameters that dir e ctly
impact b oth bandwidth usage and r endering quality . This charac-
teristic enables adaptiv e bitrate contr ol, facilitating a smo oth
str eaming e xp erience for users.
W e hav e implemente d a naiv e ABR demo by simply tail-dr op-
ping the lo w est opacity Gaussians fr om curr ent activ e Gaussians
set, to fulfill the bandwidth constraint of a simulate d p o or netw ork.
Figur e  9  sho ws the rate and distortion trade off in terms of r e quir e d
bitrate and PSNR. In the r eal w orld scenario , during netw ork con-
gestion, w e can dir e ctly r e duce the numb er of Gaussians p er up date
by transmitting only a subset of Gaussians sample d fr om each slice
with r est marke d as empty padding. This appr oach helps maintain-
ing acceptable vide o quality under constraine d bandwidth.
5.6. Ablation Study
5.6.1. A daptiv e gradient scaling.
A s sho wn in T able  5 , applying adaptiv e gradient scaling as an
appr o ximate of learning rate de cay yield a b etter r endering quality .
5.6.2. Quantization of Gaussian attributes.
W e hav e trie d to quantize differ ent Gaussian attributes with a
variety of pr e cisions as sho wn in T able  6 . W e found that, quanti-
Figur e 9: Rate and distortion trade off thr ough subsampling
Gaussians p er slice up date
Scene Co ok Spinach Cut Be ef Sear Steak
w/o gradient scaling 33.35 33.44 33.62
w/ gradient scaling 33.81 33.71 33.62
T able 5: Ablation study for adaptiv e gradient scaling
Attr means r otation scale feat opacity none
Quant fp16 uint8 fp16 fp16 uint8 -
PSNR 33.74 33.51 33.81 31.21 33.63 33.81
T able 6: Ablation study for Gaussian attributes quantization
zation o v er Gaussian means, scales, and opacity will not make a
big differ ence to war ds photometrics, while a subtle degradation
has b e en obser v e d for r otation quantization. Further , spherical
harmonics ar e rather sensitiv e attributes, so our practice is that w e
only r etains the dc of spherical harmonics and ke eps full pr e cision.
6. Conclusion
Drawing inspiration fr om r e cent advancements in neural r ender-
ing, our w ork, SwinGS , adapts 3D Gaussian Splatting (3DGS)
te chniques to the challenging domain of long v olumetric vide o
str eaming. W e first identify the unique challenges inher ent in this
task compar e d to pr e vious dynamic 3DGS task. In r esp onse , w e
pr op ose a no v el metho d that emplo ys a sliding windo w te chnique
for continuously training 3D Gaussian mo dels and captur es Gauss-
ian snapshots for each frame in a slice-by-slice manner .
Our w ork r epr esents a significant step for war d in the r ealm
of v olumetric vide o str eaming, le v eraging the str engths of 3DGS:
compact r epr esentation, high r endering quality , and rapid r ender-
ing sp e e d. W e b elie v e SwinGS  op ens up ne w av enues for r esear ch
and de v elopment in this e xciting field. A s the demand for immer-
siv e and interactiv e visual e xp eriences continues to gr o w , w e
anticipate that our contribution will catalyze further inno vations
in r eal-time v olumetric vide o str eaming.
A ckno wle dgements
Our SwinGS  W ebGL vie w er is built on the basis of K e vin K w ok’s
https://antimatter15.com/splat/
8

---

## Page 9

Refer ences
[1] 2024. Gracia.  Retrie v e d fr om https://stor e .steamp o w er e d.
com/app/2802520/Gracia/
[2] 2024. unlo cking ne xt-gen r endering: 3d gaussian splatting
on pico 4 ultra.  Retrie v e d fr om https://de v elop er .pico xr .com/
ne ws/3dgs-pico4ultra/
[3] 2024. metalsplatter for apple vision pr o .  Re-
trie v e d fr om https://radiancefields.com/metalsplatter-for-
apple-vision-pr o
[4] Je ongmin Bae , Se oha Kim, Y oungsik Y un, Hahyun Le e , Gun
Bang, and Y oungjung Uh. 2025. Per-gaussian emb e dding-
base d deformation for deformable 3d gaussian splatting. In
Eur op ean Confer ence on Computer Vision , 2025. 321–335.
[5] Ang Cao and Justin Johnson. 2023. He xplane: A fast r epr e-
sentation for dynamic scenes. In Pr o ce e dings of the IEEE/CVF
Confer ence on Computer Vision and Pattern Re cognition , 2023.
130–141.
[6] Y uanxing Duan, Fangyin W ei, Qiyu Dai, Y uhang He , W en-
zheng Chen, and Bao quan Chen. 2024. 4D-Rotor Gaussian
Splatting: T o war ds Efficient No v el Vie w Synthesis for D y-
namic Scenes. In A CM SIGGRAPH 2024 Confer ence Pap ers ,
2024. 1–11.
[7] Zhiw en Fan, K e vin W ang, K airun W en, Zehao Zhu, Dejia
Xu, Zhangyang W ang, and others. 2024. Lightgaussian: Un-
b ounde d 3d gaussian compr ession with 15x r e duction and
200+ fps. A dvances in neural information pr o cessing systems
37, (2024), 140138–140158.
[8] Y ongjie Guan, Xue yu Hou, Nan W u, Bo Han, and T ao Han.
2023. Metastr eam: Liv e v olumetric content captur e , cr eation,
deliv er y , and r endering in r eal time . In Pr o ce e dings of the 29th
A nnual International Confer ence on Mobile Computing and
Netw orking , 2023. 1–15.
[9] Bo Han, Y u Liu, and Feng Qian. 2020. ViV o: Visibility-awar e
mobile v olumetric vide o str eaming. In Pr o ce e dings of the 26th
annual international confer ence on mobile computing and net4
w orking , 2020. 1–13.
[10] Yi-Hua Huang, Y ang- Tian Sun, Ziyi Y ang, Xiao yang Lyu,
Y an-Pei Cao , and Xiaojuan Qi. 2024. Sc-gs: Sparse-contr olle d
gaussian splatting for e ditable dynamic scenes. In Pr o ce e dings
of the IEEE/CVF Confer ence on Computer Vision and Pattern
Re cognition , 2024. 4220–4230.
[11] Eue e S Jang, Marius Pr e da, Khale d Mammou, Ale xis M
T ourapis, Jungsun Kim, Danillo B Graziosi, Sungr y eul Rhyu,
and Madhukar Budagavi. 2019. Vide o-base d p oint-cloud-
compr ession standar d in MPEG: Fr om e vidence colle ction to
committe e draft [ standar ds in a nutshell]. IEEE Signal Pr o cess4
ing Magazine  36, 3 (2019), 118–123.
[12] Y uheng Jiang, Zhehao Shen, Penghao W ang, Zhuo Su, Y u
Hong, Yingliang Zhang, Jingyi Y u, and Lan Xu. 2024. Hifi4g:
High-fidelity human p erformance r endering via compact
gaussian splatting. In Pr o ce e dings of the IEEE/CVF Confer ence
on Computer Vision and Pattern Re cognition , 2024. 19734–
19745.
[13] Bernhar d K erbl, Ge orgios K opanas, Thomas Leimkühler , and
Ge orge Dr ettakis. 2023. 3D Gaussian Splatting for Real- Time
Radiance Field Rendering. A CM T rans. Graph.  42, 4 (2023),
139–131.
[14] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, W ei-
w ei Sun, Jeff T seng, Hossam Isack, Abhishek K ar , Andr ea
T agliasacchi, and K wang Mo o Yi. 2024. 3D GaussianSplatting as Marko v Chain Monte Carlo . arXiv pr eprint
arXiv:2404.09591  (2024).
[15] K yungjin Le e , Juhe on Yi, Y oungki Le e , Sunghyun Choi, and
Y oung Min Kim. 2020. GROO T: a r eal-time str eaming system
of high-fidelity v olumetric vide os. In Pr o ce e dings of the 26th
A nnual International Confer ence on Mobile Computing and
Netw orking , 2020. 1–14.
[16] Lingzhi Li, Zhen Shen, Zhongshu W ang, Li Shen, and Ping
T an. 2022. Str eaming radiance fields for 3d vide o synthesis.
A dvances in Neural Information Pr o cessing Systems  35, (2022),
13485–13498.
[17] Tiany e Li, Mira Slav che va, Michael Zollho efer , Simon Gr e en,
Christoph Lassner , Changil Kim, T anner Schmidt, Ste v en
Lo v egr o v e , Michael Go esele , Richar d Ne w comb e , and others.
2022. Neural 3d vide o synthesis fr om multi-vie w vide o . In
Pr o ce e dings of the IEEE/CVF Confer ence on Computer Vision
and Pattern Re cognition , 2022. 5521–5531.
[18] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. 2024. Spacetime
gaussian featur e splatting for r eal-time dynamic vie w synthe-
sis. In Pr o ce e dings of the IEEE/CVF Confer ence on Computer
Vision and Pattern Re cognition , 2024. 8508–8520.
[19] W eikai Lin, Y u Feng, and Y uhao Zhu. 2024. RT GS: Enabling
Real- Time Gaussian Splatting on Mobile De vices Using
Efficiency-Guide d Pruning and Fo v eate d Rendering. arXiv
pr eprint arXiv:2407.00435  (2024).
[20] Junhua Liu, Y uanyuan W ang, Y an W ang, Y ufeng W ang,
Shuguang Cui, and Fang xin W ang. 2023. Mobile v olumetric
vide o str eaming system thr ough implicit neural r epr esenta-
tion. In Pr o ce e dings of the 2023 W orkshop on Emerging Multi4
me dia Systems , 2023. 1–7.
[21] K aiyan Liu, Ruizhi Cheng, Nan W u, and Bo Han. 2023.
T o war d ne xt-generation v olumetric vide o str eaming with
neural-base d content r epr esentations. In Pr o ce e dings of the 1st
A CM W orkshop on Mobile Immersiv e Computing, Netw orking,
and Systems , 2023. 199–207.
[22] Y u Liu, Bo Han, Feng Qian, Ar vind Narayanan, and Zhi-Li
Zhang. 2022. V ues: Practical mobile v olumetric vide o str eam-
ing thr ough multivie w transco ding. In Pr o ce e dings of the 28th
A nnual International Confer ence on Mobile Computing A nd
Netw orking , 2022. 514–527.
[23] Jonathon Luiten, Ge orgios K opanas, Bastian Leib e , and De va
Ramanan. 2023. D ynamic 3d gaussians: T racking by p ersis-
tent dynamic vie w synthesis. arXiv pr eprint arXiv:2308.09713
(2023).
[24] Marko Mihajlo vic, Serge y Pr okudin, Mar c Pollefe ys, and Siyu
T ang. 2023. Resfields: Residual neural fields for spatiotemp o-
ral signals. arXiv pr eprint arXiv:2309.03160  (2023).
[25] Ben Mildenhall, Pratul P Srinivasan, Matthe w T ancik,
Jonathan T Barr on, Ravi Ramamo orthi, and Ren Ng. 2021.
Nerf: Repr esenting scenes as neural radiance fields for vie w
synthesis. Communications of the A CM  65, 1 (2021), 99–106.
[26] Panagiotis Papantonakis, Ge orgios K opanas, Bernhar d K erbl,
Ale xandr e Lanvin, and Ge orge Dr ettakis. 2024. Re ducing the
memor y fo otprint of 3d gaussian splatting. Pr o ce e dings of the
A CM on Computer Graphics and Interactiv e T e chniques  7, 1
(2024), 1–17.
[27] K erui Ren, Lihan Jiang, T ao Lu, Mulin Y u, Linning Xu,
Zhangkai Ni, and Bo Dai. 2024. Octr e e-gs: T o war ds consistent
r eal-time r endering with lo d-structur e d 3d gaussians. arXiv
pr eprint arXiv:2403.17898  (2024).
9

---

## Page 10

[28] Johannes Lutz Schönb erger , Enliang Zheng, Mar c Pollefe ys,
and Jan-Michael Frahm. 2016. Pixelwise Vie w Sele ction for
Unstructur e d Multi- Vie w Ster e o . In Eur op ean Confer ence on
Computer Vision (ECCV) , 2016.
[29] Liangchen Song, Anp ei Chen, Zhong Li, Zhang Chen, Lele
Chen, Junsong Y uan, Yi Xu, and Andr eas Geiger . 2023.
Nerfplay er: A str eamable dynamic scene r epr esentation with
de comp ose d neural radiance fields. IEEE T ransactions on Vi4
sualization and Computer Graphics  29, 5 (2023), 2732–2742.
[30] Patrick Stotko , Stefan Krump en, Max Schwarz, Christian
Lenz, Sv en Behnke , Reinhar d Klein, and Michael W einmann.
2019. A VR system for immersiv e tele op eration and liv e e x-
ploration with a mobile r ob ot. In 2019 IEEE/RSJ International
Confer ence on Intelligent Rob ots and Systems (IROS) , 2019.
3630–3637.
[31] Jiakai Sun, Han Jiao , Guangyuan Li, Zhanjie Zhang, Lei Zhao ,
and W ei Xing. 2024. 3dgstr eam: On-the-fly training of 3d
gaussians for efficient str eaming of photo-r ealistic fr e e-vie w-
p oint vide os. In Pr o ce e dings of the IEEE/CVF Confer ence on
Computer Vision and Pattern Re cognition , 2024. 20675–20685.
[32] Y uan-Chun Sun, Y uang Shi, W ei T sang Ooi, Chun- Ying
Huang, and Cheng-Hsin Hsu. 2024. Multi-frame Bitrate Allo-
cation of D ynamic 3D Gaussian Splatting Str eaming O v er
D ynamic Netw orks. In Pr o ce e dings of the 2024 SIGCOMM
W orkshop on Emerging Multime dia Systems , 2024. 1–7.
[33] V arjo T e chnologies. 2024. V arjo Demonstrates T ele-
p ort, a Po w erful Ne w Ser vice for T urn-
ing Real- W orld P laces into Virtual Exp eri-
ences.  Retrie v e d fr om https://varjo .com/pr ess-r elease/varjo-
demonstrates-telep ort-a-p o w erful-ne w-ser vice-for-turning-
r eal-w orld-places-into-virtual-e xp eriences/
[34] Diw en W an, Ruijie Lu, and Gang Zeng. 2024. Sup erp oint
Gaussian Splatting for Real- Time High-Fidelity D ynamic
Scene Re construction. arXiv pr eprint arXiv:2406.03697  (2024).
[35] Max W elling and Y e e W T eh. 2011. Bay esian learning via
sto chastic gradient Lange vin dynamics. In Pr o ce e dings of the
28th international confer ence on machine learning (ICML411) ,
2011. 681–688.
[36] Guanjun W u, T aoran Yi, Jiemin Fang, Ling xi Xie , Xiaop eng
Zhang, W ei W ei, W enyu Liu, Qi Tian, and Xinggang W ang.
2024. 4d gaussian splatting for r eal-time dynamic scene
r endering. In Pr o ce e dings of the IEEE/CVF Confer ence on Com4
puter Vision and Pattern Re cognition , 2024. 20310–20320.
[37] Ze yu Y ang, Hongy e Y ang, Zijie Pan, Xiatian Zhu, and Li
Zhang. 2023. Real-time photor ealistic dynamic scene r epr e-
sentation and r endering with 4d gaussian splatting. arXiv
pr eprint arXiv:2310.10642  (2023).
[38] Ziyi Y ang, Xinyu Gao , W en Zhou, Shaohui Jiao , Y uqing
Zhang, and Xiaogang Jin. 2024. Deformable 3d gaussians
for high-fidelity mono cular dynamic scene r e construction. In
Pr o ce e dings of the IEEE/CVF Confer ence on Computer Vision
and Pattern Re cognition , 2024. 20331–20341.
[39] Anlan Zhang, Chendong W ang, Bo Han, and Feng Qian. 2021.
Efficient v olumetric vide o str eaming thr ough sup er r esolu-
tion. In Pr o ce e dings of the 22nd International W orkshop on
Mobile Computing Systems and A pplications , 2021. 106–111.
[40] Pengyuan Zhou, Jinjing Zhu, Yiting W ang, Y unfan Lu, Zixi-
ang W ei, Haolin Shi, Y uchen Ding, Y u Gao , Qinglong Huang,
Y an Shi, and others. 2022. V etav erse: A sur v e y on the inter-se ction of Metav erse , v ehicles, and transp ortation systems.
arXiv pr eprint arXiv:2210.15109  (2022).
10

---

## Page 11

Supplementar y Material
Figur e 10: Rate distortion trade off for Cut Roaste d Be ef
Fr om left to right: 100%, 80%, 60%, 40%, 20% subsampling p er frame up date
Figur e 11: Rate distortion trade off for Co ok Spinach
Fr om left to right: 100%, 80%, 60%, 40%, 20% subsampling p er frame up date
Figur e 12: Rate distortion trade off for Sear Steak
Fr om left to right: 100%, 80%, 60%, 40%, 20% subsampling p er frame up date
7. Qualitativ e r esult for ABR
8. Implementation details
8.1. Setup
For mo del training in SwinGS , w e car efully tune d the hyp er-
parameters to achie v e optimal p erformance . The ke y parameters
w er e set as follo ws: scale_reg  at 1e-2, opacity_reg  at 2e-2, and
noise_lr  at 5e4. The degr e e of spher e harmonics function for
vie wp oint dep endent coloring is set to 1 to r e duce storage cost.
These values w er e determine d thr ough e xtensiv e e xp erimentation
to balance mo del accuracy and computational efficiency .
W e initialize our mo del using SfM p oints deriv e d fr om COLMAP
[ 28 ] . This appr oach pr o vides a str ong initial ge ometr y estimate ,
significantly impr o ving conv ergence sp e e d and final mo del quality .
W e also e xplor e d random initialization, but found it pr one to o v er-
fitting in our e xp erimental setup . By default, our training setup is
adapte d to the sp e cific characteristics of each dataset as follo ws:
Scene Genesis Iters Iters T otal Num Swin Size
D yNeRF [ 17 ] 30K 2K 200K 5
Geneiss iters  r efers to the training iterations for the v er y first
sliding windo w , i.e . [0, swin_size ) and Iters  r efers to trainingiterations for the follo wing sliding windo ws. Total Number  indi-
cates the maximum activ e Gaussians for a frame .
The choice of Gaussian counts was made to balance mo del
e xpr essiv eness with computational efficiency as w ell as storage
cost. D yNeRF scenes b enefit fr om a higher Gaussian count due
to the pr esence of b oth dynamic for egr ound elements and static
backgr ound details. The sliding windo w size of 5 frames was
sele cte d as an optimal trade-off b etw e en temp oral coher ence and
computational r esour ces.
9. Further discussion
9.1. 3DGS and Point Cloud-Base d Metho ds
While T able  1  pr esents 3D Gaussian Splatting (3DGS) and p oint
cloud metho ds as alternativ es, 3D Gaussians can actually b e
vie w e d as an e xtension of 3D p oints. In addition to the p osition
(xyz) and color (rgba) pr op erties inher ent to p oints, 3D Gaussians
incorp orate r otation (R) and scaling (S). This r elationship allo ws
for seamless adaptation of p oint cloud-base d v olumetric vide o
str eaming inno vations to the 3D Gaussian r epr esentation.
For instance , GROO T [ 15 ]  emplo ys o ctr e es for efficient ge om-
etr y data compr ession, a data structur e r e cently also has b e en
applie d to 3D Gaussians in Octr e e-GS [ 27 ]  to r e duce the total
numb er of Gaussians in a r endering scene . Similarly , RT GS [ 19 ]
11

---

## Page 12

adopts an appr oach analogous to ViV o [ 9 ] , utilizing user camera
p oses to optimize r endering r esour ce allo cation acr oss differ ent
scene se ctions.
On the other hand, curr ent off-the-shelf p oint cloud co de c,
MPEG Point Cloud Compr ession (PCC) [ 11 ]  or a general data com-
pr essor like arithmetic entr op y enco ding could also b e integrate d
to further r e duce the transmission load during vide o str eaming, at
a cost of longer de co ding time on the client side .
W e anticipate further e xtensions of p oint cloud te chniques to
3DGS, for e xample p oint cloud sup er-r esolution [ 39 ] , which could
p otentially r e duce data str eaming traffic costs in futur e implemen-
tations.
9.2. Limitations and Futur e W ork
A s the first effort to adapt 3DGS fr om short 3D dynamic scenes
to long v olumetric vide os, SwinGS  faces se v eral challenges that
warrant further inv estigation.
9.2.1. Infle xible Maturation Sche dule .  Curr ent Gaussian mat-
uration pr o cess follo ws a fixe d sche dule , wher e Gaussians matur e
after e xactly swin_size  frames to facilitate efficient batch op era-
tions in GP U memor y . Although the Gaussian r elo cation me cha-
nism and “birth” field allo w some inter-slice migration, w e hav e
y et to fully optimize bandwidth usage with the most informativ e
content. An ideal scenario w ould inv olv e r eplacing the least useful
Gaussians with the most informativ e ne w ones, rather than up dat-
ing a pr eassigne d fixe d subset. For instance , within 40K Gaussians
p er frame , w e could prioritize up dates for Gaussians r epr esenting
human motion while less fr e quently up dating those depicting
static backgr ounds.
12