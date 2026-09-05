# Analysis: survey + three miscellaneous papers on ceramic fragment reassembly

Prepared for the museum 3D fragment reassembly tool. Target data: thick-walled terracotta,
coloured PLY meshes (350k–670k verts), non-axisymmetric sculptural object first, pots/plates later.
Constraints: CPU-only (M2 Pro, 10 cores, 16 GB), Python, no GPU, no training data, no ground truth,
minutes not hours, 4 to dozens of fragments, missing/foreign pieces possible.

---

# PART 1 — SURVEY: Eslami, Di Angelo, Di Stefano & Pane (2020)

*"Review of computer-based methods for archaeological ceramic sherds reconstruction",
Virtual Archaeology Review 11(23): 34–49. DOI 10.4995/var.2020.13134.*

## 1.0 What the survey is and how it is organised

53 English-language papers from Scopus up to end of 2019, restricted to **pottery fragments only**.
Split into three periods: before 2000 (7 papers), 2000–2009 (20), 2010–2019 (26). The survey's own
organising frame is a **six-step generic pipeline**, not a taxonomy of matching algorithms:

1. Data acquisition & pre-processing
2. Feature extraction
3. Orientation
4. Classification
5. Reconstruction (= matching + assembly)
6. Refinement (= filling missing parts)

The survey explicitly notes that **steps 3 (Orientation) and 6 (Refinement) are addressed in very
few papers**. Its Tables 1–3 tabulate, per paper: acquisition tool, features extracted,
classification method, matching technique. Those tables are the raw material for the taxonomy below.

**Critical caveat for us: the survey NEVER states code availability for ANY method.** A full-text
grep for "source code", "open source", "publicly available", "github", "implementation available"
returns exactly one hit in the whole document — the Blender URL in the reference list. The only
software artefacts named anywhere are *consumer tools*: Blender, MeshLab, CloudCompare, 3ds Max,
Geomagic Studio, ScanStudio, SolidWorks, Agisoft PhotoScan, and one closed commercial plugin called
"Fragments Reassembler" (Kotoula et al. 2016). **Every "code available?" cell in the taxonomy below
therefore reads "not stated" — that is a fact about the survey, not about the underlying works.**
Availability must be checked independently at the primary sources.

## 1.1 Taxonomy of reassembly method families

I reconstructed seven families from Tables 1–3 and the narrative. The survey does not name them this
way; the grouping is mine, the contents are its.

---

### FAMILY A — Axis / profile-based (rotational-symmetry prior)

The single largest family in the survey; roughly half of all reassembly papers.

| | |
|---|---|
| **Key works** | Halir & Menard (1996); Halir & Flusser (1997); Halíř (1999); Sablatnig & Menard (1998); Melero et al. (2003); Cao & Mumford (2002); Willis, Orriols & Cooper (2003); Kampel & Sablatnig (2003a, 2003b, 2004); Kampel et al. (2005); Maiza & Gaildrat (2005); Mara & Sablatnig (2006); Zhou et al. (2010); Karasik & Smilansky (2011); Han & Hahn (2014); Di Angelo, Di Stefano & Pane (2017, 2018); Banterle et al. (2017); Kalasarinis & Koutsoudis (2019) |
| **Input assumed** | Wheel-thrown, **rotationally symmetric** vessel. Thin-to-medium wall. Colour not used. Fragment must be large enough and curved enough for the axis to be observable. |
| **Matching primitives** | Rotation axis (3 DOF after normalisation) + 2D profile curve: radius, tangent and curvature as a function of height. Kampel et al. (2005) additionally uses **rills** (throwing grooves) on the inner surface as an orientation cue when the profile is degenerate. |
| **Global assembly strategy** | Place every fragment in a shared cylindrical frame. This collapses the 6-DOF pose problem to ~2 DOF (height along axis + azimuth). Then match profile sections / break curves within that reduced space. |
| **Reported robustness** | Halir & Flusser: diameter error 2–3 mm on synthetic + real. Kampel & Sablatnig (2003a): 40 fragments, **50 % success**. Mara et al. (2002) classification: 62/70 fragments correct. Karasik & Smilansky (2011) classification: 94.8 % on 358 fragments. Kampel et al. (2005) tested on 35 small, low-curvature fragments. |
| **Stated failure modes** | The survey states this plainly and repeatedly: *"The computation of the rotational axis is ambiguous when the surface of the fragment is too flat or too small"*; *"sometimes the fragments are too small; hence estimating an accurate axis/profile-curve of a fragment is not obvious and may not even be possible"*. Han & Hahn (2014) exists solely to patch this. |
| **Code stated?** | No. |

**Verdict for us: structurally inapplicable to the current dataset.** Structure-from-Sherds++ sits
squarely in this family; its failure on our sculptural, non-axisymmetric object is the *expected*
outcome, not a bug. Keep this family in reserve for the later pot/plate sets only.

---

### FAMILY B — Break-curve / breaking-curve (1D contour) matching

| | |
|---|---|
| **Key works** | Üçoluk & Toroslu (1999); Cooper et al. (2001); Andrews & Laidlaw (2002); Willis & Cooper (2004); Zhou et al. (2007); Son, Almeida & Cooper (2013); Rasheed & Nordin (2014, 2018). Leitão & Stolfi (2005) is cited from Rasheed's own review, not the survey. |
| **Input assumed** | Break curve must be extractable, i.e. a clean ridge where the intact surface meets the fracture surface. Üçoluk & Toroslu assume *nothing* about symmetry — the only fully symmetry-free member. Zhou et al. (2007) explicitly targets **fragments with thickness**, using internal *and* external contours of a solid object. |
| **Matching primitives** | The 3D boundary space curve, described by **rigid-invariant differential signatures: curvature κ(s) and torsion τ(s)** sampled along arclength (Üçoluk & Toroslu). Zhou et al. use polygonal arcs and **junction vertices**. Cooper/Willis fit mathematical models to break curves + outer surface. |
| **Global assembly strategy** | Pairwise candidate generation → greedy or bottom-up merge. Andrews & Laidlaw: quasi-Newton optimisation of an *ensemble likelihood*, ranked, then greedy pair selection. Zhou et al. 2007: **binary tree** with fragments as nodes. Willis & Cooper: MLE / Bayesian over break curves and outer surface **simultaneously**. |
| **Reported robustness** | Üçoluk & Toroslu: explicitly a *"Noise Tolerant algorithm"*, validated on simulated broken objects, designed for the missing-pieces case. Andrews & Laidlaw: 8 groups / 16 fragments → **only 13 valid pairwise matches** recovered. Willis & Cooper (2004): **10 of 13 fragments** of one vase assembled. Son et al. (2013): robust to noise, bumps and erosion. |
| **Stated failure modes** | The survey names the killer directly: *"the break-curves … may be eroded and chipped, so that the search space for reconstruction of these fragments can become huge."* |
| **Code stated?** | No. |

**Verdict for us: high relevance.** Break curves are cheap to extract on a watertight mesh
(fracture-surface boundary loops), are 1D so matching is O(n²) in curve samples rather than in
vertices, and κ/τ signatures are rigid-invariant so no pose search is needed to score a candidate.
The erosion problem is real but our fracture surfaces are described as clearly visible.

---

### FAMILY C — Fracture-surface (2D area) matching — **the key family for us**

| | |
|---|---|
| **Key works** | Papaioannou, Karabassi & Theoharis (2000, and 2002 in IEEE TPAMI 24(1):114–124); **Huang, Flöry, Gelfand, Hofer & Pottmann (2006), SIGGRAPH**; Zhou et al. (2007); Belenguer & Vidal (2012); Vendrell-Vidal & Sánchez-Belenguer (2014), JOCCH 7(3). |
| **Input assumed** | **Solid / thick objects with genuine fracture surfaces. No symmetry assumption. No colour required.** Papaioannou's "material and structural constraints" imply volumetric fragments. Zhou et al. 2007 is explicitly *"a solution to deal with the problems of reassembling fragments with thickness"*. |
| **Matching primitives** | Papaioannou 2000: **surface bumpiness** computed from a depth buffer, used to *identify which sides are fracture sides* — "the system chooses the least irregular sides for correct matching". Papaioannou 2002: pointwise **distance between mutually visible faces** of a fragment pair, evaluated over the whole surface, via a z-buffer. Huang et al. 2006: multi-scale edge extraction → **graph-cut partition into original vs. fracture faces** → cluster/patch descriptors on fracture surfaces → **forward search** to select a robust feature subset. Belenguer/Vendrell-Vidal: discrete uniform sampling of fracture surfaces into projective depth maps; distance between edge-of-fracture point sets. |
| **Global assembly strategy** | Papaioannou 2002: **global optimisation with the pairwise matching error as cost function and material-axis/surface overlap as hard constraints** (Table 2 lists the 2000 variant as simulated annealing / genetic-like). Huang et al. 2006: pairwise matching produces a candidate set, then **global multi-piece matching with simultaneous local multi-piece registration**. Vendrell-Vidal: cost function over discrete samples plus a **hierarchical (coarse-to-fine) search** guaranteeing convergence to the global solution. |
| **Reported robustness** | Papaioannou et al. (2000), real + synthetic data: **50 % of fragments correctly assembled with no constraints and no user intervention; 90 % with material/structural constraints plus user-enforced selection.** Belenguer & Vidal (2012): hierarchical search is **250× faster than exhaustive search**. Vendrell-Vidal & Sánchez-Belenguer (2014): "great performances", not quantified in the survey. |
| **Code stated?** | No. |
| **Compute caveat** | Belenguer & Vidal and Vendrell-Vidal explicitly push "all heavy calculations" onto the **GPU** (projective depth maps). We have no GPU. The *hierarchical search* idea transfers to CPU; the depth-map rasteriser must be replaced by KD-tree nearest-neighbour queries or a CPU rasteriser. |

**Verdict for us: this is the family to build on.** It makes exactly the assumptions our data
satisfies — thick, watertight-ish, clear fracture surfaces, arbitrary non-symmetric shape — and
makes none that it violates.

---

### FAMILY D — Appearance / feature-descriptor based (colour, texture, normal maps)

| | |
|---|---|
| **Key works** | Kampel & Sablatnig (2000); Brown et al. (2008) ACM TOG 27(3), the Theran wall-paintings system; Shin et al. (2010); Toler-Franklin, Funkhouser, Rusinkiewicz, Brown & Weyrich (2010) ACM TOG 29(6); Funkhouser et al. (2011) JOCCH 4(2); Cohen, Zhang & Jeppson (2010, 2016); Rasheed & Nordin (2015, 2018); Rasheed et al. (2017); Smith et al. (2010). |
| **Input assumed** | **Thin, flat fragments — frescoes and wall paintings, not vessels.** Colour and a decorated/painted surface are mandatory. Brown et al. capture "shape on the side of the fragments, colour, plaster surface texture and surface roughness". |
| **Matching primitives** | Colour maps, normal maps, texture statistics (GLCM contrast/correlation/entropy/homogeneity; LBP histograms; HSV block descriptors), contour + ribbon + junction angle, convex hulls of surface markings, affine moment invariants. |
| **Global assembly strategy** | Brown et al.: exhaustive **matching error at all possible orientations** for every pair. Funkhouser et al.: train a classifier (M5P regression trees) on many computable properties, then **rank** predicted pairwise matches by precision. Cohen et al. (2016): align each fragment to a **generic vessel model**, mend the rest by border anchor points, pairwise only, no global consideration. |
| **Reported robustness** | Toler-Franklin et al. (2010), three fresco datasets: **90 % correct on match features, 78 % on non-match**. Brown et al. (2008): "high precision". Funkhouser et al. (2011): a classifier trained on one dataset transfers to another. |
| **Code stated?** | No. |

**Verdict for us: partial relevance, low priority.** The geometry assumption (thin flat frescoes) is
wrong for us. But the *ranking* idea from Funkhouser and the *surface roughness / bumpiness* cue are
directly reusable, and our meshes do carry vertex colour, which can serve as a cheap **rejection**
filter (terracotta fabric colour differs between objects) for the "pieces from other objects mixed
in" requirement.

---

### FAMILY E — Template / generic-model based

| | |
|---|---|
| **Key works** | Cohen et al. (2016); Banterle et al. (2017); Kalasarinis & Koutsoudis (2019); Fragkos et al. (2018). |
| **Input assumed** | A prior typology or CAD/profile catalogue for the object class exists. |
| **Primitives / strategy** | Cohen: generic vessel models built from expert historical knowledge + provenance; align fragments to model using weighted moments; fall back to border anchor points. Banterle: extract structured geometric description from **paper profile catalogues**, generate a large set of **synthetic sherds**, train a classifier on them. Kalasarinis / Fragkos: reverse-model the missing regions, then FDM/3D-print them. |
| **Reported robustness** | Not quantified in the survey. Applied to Roman amphorae, terra sigillata, medieval pottery; two Hellenistic vessels. |
| **Code stated?** | No. |

**Verdict for us: 1/5.** We have no template for a sculptural terracotta object and no ground truth.

---

### FAMILY F — Thickness-profile based

| | |
|---|---|
| **Key work** | Stamatopoulos & Anagnostopoulos (2016), arXiv:1601.05824. |
| **Rationale (its own)** | *All* other families read the **external** surface, which is exactly what erosion, wear and encrustation destroy. Wall thickness does not change. |
| **Pipeline** | Photograph each fragment on a stable base from all sides (30 cameras) → build 3D model → extract the **optimal Thickness Profile (TP)** of each fragment → iteratively maximise TP matching score between candidate neighbours. |
| **Reported robustness** | Successful on deliberately broken test vessels; "seems not to be affected" by wear/erosion. **Requires human interaction** (Table 3 lists "Thickness Profile Matching Method / Human Interaction" as the matching technique). |
| **Code stated?** | No. |

**Verdict for us: 3/5 as a *pruning* signal, not as a matcher.** Our brief states wall thickness is
5–10 % of fragment extent, so it is directly measurable. Thickness is a cheap, rotation-invariant,
1D scalar that prunes candidate pairs before any expensive surface matching, and it separates
fragments from *different* objects.

---

### FAMILY G — Learning-based

| | |
|---|---|
| **Key works** | Igwe & Knopf (2006) SOFM; Toler-Franklin et al. (2010); Funkhouser et al. (2011) M5P regression trees; Rasheed et al. (2017) SOM; Rasheed & Nordin (2018) backprop NN; Banterle et al. (2017). |
| **Input assumed** | Labelled or at least large data. Funkhouser and Banterle need a training set; Igwe/Rasheed 2017 use *unsupervised* SOM/SOFM and need none. |
| **Survey's own recommendation** | The conclusion pushes hard for this: *"Data-mining techniques, artificial neural networks, and deep learning are some of the tools that researchers can use to classify and reconstruct, even when the original patterns are unknown"*, conditional on *"If the data is large enough"*. |
| **Code stated?** | No. |

**Verdict for us: 0–1/5 for supervised methods** (no training data, no GT, no GPU). The unsupervised
SOM/SOFM clustering idea is marginally reusable for grouping fragments by fabric before matching.

---

### FAMILY H — Metaheuristic pose search (cross-cutting; the *global* half of several families)

| | |
|---|---|
| **Key works** | Papaioannou et al. (2000) simulated annealing / genetic-like; Melero et al. (2003) GA for orientation; Maiza & Gaildrat (2005) GA; Kashihara (2012, 2017) **GA + hill-climbing**. |
| **Strategy** | Treat the whole reassembly as a continuous global optimisation over fragment poses. Kashihara: real-coded GA for a coarse global solution, then hill-climbing to fine-tune 3D positions; the objective compares **silhouettes** from ~30 cameras at different angles (2012) or AKAZE image features (2017). |
| **Reported robustness** | Kashihara (2012) validated on **one vase of five fragments**. |
| **Survey's own summary** | Figure 6: the two most-used reconstruction methods across the whole corpus are **meta-heuristic optimisation** and **similarity analysis** (~9–10 papers each), ahead of Bayesian, hierarchical search, machine learning and cost-function approaches. |
| **Code stated?** | No. |

**Verdict for us: 2/5.** GA over 6·N DOF will not run in minutes on CPU for dozens of fragments.
The *coarse-global-then-local-refine* two-stage structure is worth keeping; the GA is not.

---

## 1.2 Which methods the survey supports for THICK fragments with fracture surfaces, NON-axisymmetric

The survey never issues an explicit "best method" ranking. Reading its evidence, four works are the
only ones that assume thickness *and* drop the symmetry prior:

1. **Huang et al. (2006), SIGGRAPH** — the most complete non-symmetric pipeline in the corpus:
   segmentation into original vs. fracture faces by graph cut, patch-based fracture descriptors,
   robust feature selection by forward search, then global multi-piece matching with simultaneous
   local multi-piece registration. It is the only entry that solves *both* pairwise and multi-piece
   without a symmetry or template crutch.
2. **Papaioannou et al. (2000, 2002)** — surface-morphology matching by pointwise distance between
   mutually visible faces, with **material overlap as a hard constraint**. Only work in the survey
   with clean quantitative numbers on real fragments: **50 % unconstrained / 90 % with constraints
   and user intervention**.
3. **Zhou et al. (2007)** — the only paper the survey introduces with the words *"a solution to deal
   with the problems of reassembling fragments with thickness"*: internal + external contour
   extraction on solid fractured objects, junction-vertex matching, binary-tree assembly, plus a
   repair step for seams and holes.
4. **Belenguer & Vidal (2012) / Vendrell-Vidal & Sánchez-Belenguer (2014)** — discrete pairwise
   fracture-surface matching with a cost function on uniformly sampled fracture points and a
   hierarchical search that guarantees the global optimum, 250× faster than exhaustive. Their
   original targets are flat fresco fragments and their speed comes from the GPU, but the
   discretisation + hierarchy is generic to any fracture surface.

Also relevant but *not* an assembly method: **Di Angelo, Di Stefano & Pane (2018)** is the survey's
only work explicitly about *non-axially-symmetric* pottery — it recognises "detail features of
constant radius" (DFCR) on decorated vessels via fuzzy-sensitivity segmentation and robust fitting.
It extracts features, it does not reassemble. For a terracotta-warrior-like object with rectangular
relief plates, the analogous idea (segment planar/constant-curvature relief patches and use them as
strong alignment anchors) is worth stealing.

## 1.3 Which methods the survey supports for pots (axisymmetric) — our later sets

Ranked by the evidence the survey reports:

1. **Son, Almeida & Cooper (2013), CVPR** — estimates an **Axis-of-symmetry Profile Curve (APC)**
   per fragment using **circle templates**, then reassembles by **break-curve matching**, evaluating
   both local and global solutions. Explicitly designed for the fragments that break other methods:
   *"almost flat, chipped, and represented by very noisy data"*. Reported **robust to noise, bumps
   and erosion**; assembled **three vases / 48 fragments in 10.56 hours**. This is the direct
   ancestor of Structure-from-Sherds.
2. **Willis & Cooper (2004)** — Bayesian assembly. Fit mathematical models to the outer surface and
   the break curves, then optimise alignment by **Maximum Likelihood Estimation over both
   simultaneously**. Survey lists three advantages: combines heterogeneous evidence types, the
   search skips unlikely configurations, and it is computationally reasonable. Result: **10 of 13
   fragments** of one vase.
3. **Cao & Mumford (2002)** — axis estimation from the geometric property that the symmetry axis
   contains the centre of the sphere of curvature of every parallel circle; least-squares line fit;
   cubic-spline profile; **bootstrap confidence bounds** on axis and profile used as the matching
   features. Robust to noisy 3D data by construction.
4. **Kampel & Sablatnig (2003a/b, 2004)** — profile-section classification and matching by
   point-by-point distance between facing outlines. 40 fragments, 50 % success.
5. **Kampel et al. (2005)** — the fallback when the axis is degenerate: orient the fragment from the
   **rills on the inner surface** (the wheel-throwing grooves), mimicking the manual archaeological
   method. Tested on 35 small, low-curvature fragments.

## 1.4 Open problems the survey states

Verbatim in substance, from Section 4:

1. **Wear, erosion, encrustation and chipping are essentially unaddressed.** *"very little research
   has taken this problem into account, and often this is not considered. Instead, to implement
   systems that can be used in the field, it would be useful to develop a system that is robust in
   the analysis of ceramic sherds noised, worn, encrusted, and chips."*
2. **Data acquisition is the bottleneck.** *"the data acquisition step remains the one that requires
   greater intervention by the operator. For this reason, it represents the bottleneck in the
   development of a fully automatic analysis method."* Calls for automated bulk scanning.
3. **Machine learning is under-exploited**, conditional on data volume; data mining, ANNs and deep
   learning are named as the way to classify and reconstruct "even when the original patterns are
   unknown".
4. **Break-curve representation is wrong.** Polynomial fitting of fragment edges is criticised
   because *"the edges of fragments are almost always irregular"*; the survey recommends
   **non-parametric methods such as wavelet transformation** instead.
5. **Missing-part completion (the "Refinement" step) is barely studied**, and the few methods that
   exist *"require user interaction during all steps of the process"*. Calls for automatic
   identification and 3D printing of missing fragments.
6. Implicitly: **Orientation and Refinement are addressed in only a handful of the 53 papers.**

## 1.5 Relevance verdict for the survey — **4 / 5**

Not an algorithm, but the highest-value item in this batch. It gives (a) the field map, (b) the one
piece of hard evidence that our failed approach was structurally doomed rather than misconfigured,
and (c) three quantitative baselines (Papaioannou 50/90 %, Willis & Cooper 10/13, Son 48 fragments
in 10.56 h) against which we can calibrate expectations. Loses a point because it reports no code
availability whatsoever and never benchmarks methods against each other on common data.

---

# PART 2 — Rasheed & Nordin (2020)

*"Classification and reconstruction algorithms for the archaeological fragments",
Journal of King Saud University – Computer and Information Sciences 32(8): 883–894. Open access,
CC BY-NC-ND. DOI 10.1016/j.jksuci.2018.09.019.*

## 2.1 What it does

Two loosely coupled subsystems. **CAF** (Classification of Archaeological Fragments) groups 2D
photographs of sherds into per-object clusters using RGB colour-set intersection plus Local Binary
Pattern texture. **RAO** (Reconstruction of Ancient Objects) then takes the 3D scans of one cluster,
extracts each fragment's boundary contour, splits it into short sub-contours, computes a 13-D
geometric descriptor per sub-contour (dominated by 3D slope), and trains a small backpropagation
neural network to decide *which sub-contour of fragment A mates with which sub-contour of fragment
B*. Once a mating sub-contour pair is identified, a rigid transform is computed and applied. The
authors' stated selling point is that assembly needs **no prior knowledge of which fragment to start
from**, which they argue makes it tolerant of gaps from missing pieces.

## 2.2 Pipeline, precisely enough to implement

**Stage 1 — CAF (2D, colour + texture).**

1. Load six photographs per fragment set (Nikon DSLR).
2. Segment fragment from background (their prior method, Rasheed & Nados 2018).
3. **Colour feature — set intersection.** For image *k*, build the set
   `C_k = { (R_ij, G_ij, B_ij) : i=1..n, j=1..m }` of all pixel triples. For each ordered pair of
   images (A, B) compute `S = A ∩ B` and record `|S|`. This yields an all-pairs intersection-count
   matrix. Reported counts in their example: image 1 intersects images 3 and 5 at 8110 and 8411
   points; image 2 intersects images 4 and 6 at 2724 and 1783.
4. **Colour grouping algorithm.** Sort each image's intersection column descending. Take image A's
   maximum-intersection partner B. If B's own maximum is also A, group them. Otherwise compare B's
   maximum partner C: if C's value dominates, do *not* group A with B; else group. Move to the next
   highest value, repeat until all columns are exhausted.
5. **Texture feature — LBP.** Compute the regular LBP histogram (Ojala et al. 1996) per fragment,
   bins 1..255 plus a separate bin 256.
6. **Fusion.** Independently group by texture using Euclidean distance between LBP histograms. Keep
   only groups where the colour grouping and the texture grouping **agree**.

**Stage 2 — RAO (3D, contour + neural network).**

7. Acquire meshes with a **Primesense Carmine 1.09** structured-light scanner. Denoise.
8. **Boundary extraction.** (a) enumerate all mesh edges, internal ones appearing twice; (b) find
   unique edges; (c) count occurrences of each; (d) **keep edges that occur exactly once** — the
   standard open-boundary test; (e) plot. Note this yields the *whole* open boundary; there is no
   fracture-vs-intact discrimination anywhere in the paper.
9. **Contour partition.** Split the boundary into **4 parts of as-equal-as-possible size** (their
   example: 75, 75, 75, 72 points), each treated as an independent object. Then split each part into
   **sub-contours of exactly 5 consecutive points**, discarding the remainder (their example: 59
   sub-contours, 2 points dropped; a second fragment gives 55 sub-contours, 3 dropped; a third 59
   with 4 dropped).
10. **Descriptor, 13 dimensions per sub-contour.** 3D slope (computed by the Maidment & Tarboton
    2011 GIS slope formula), plus min, max, mean and variance of x, y and z over the 5 points.
    Normalise before training.
11. **Network.** Feed-forward MLP, **13 input nodes → 30 hidden nodes → 8 output nodes**, one output
    per candidate sub-contour part of the reference fragments, targets in {−1, 1}. Backpropagation,
    **learning rate 0.05, momentum 0.9**, converged in **335 of a budgeted 1000 epochs**. Input
    matrix in their example is 13 × 114.
12. **Inference.** Feed each sub-contour of the unknown fragment A forward; the winning mate is the
    output node whose activation is closest to 1.
13. **Alignment.** Centre fragment A: `Center_A = (1/N) Σ p_i`. Accumulate
    `H = Σ_i (p_A^i − Center_A)`. Take `[U, S, V] = svd(H, 0)`, set `R = V·Uᵀ` (V is the direction of
    maximum variance), translate `T = R·Center_A`, slide so all x ≥ 0. Then rotate fragment B onto
    A: recover the angle by **inverse cosine of a dot product**, and apply **`Rz` only** —
    `Rz = [cosθ, −sinθ, 0; sinθ, cosθ, 0; 0, 0, 1]`, `C = P_B · Rz + T`. Finally slide along z so all
    points are positive.
14. **Scoring.** Optimal match by minimum Euclidean distance
    `d(b, A) = min_{i∈1..n} d(b, a_i)` between edge points.

## 2.3 Input assumptions

- Six **2D photographs** per fragment exist for the classification stage, plus separate 3D scans for
  the reconstruction stage. Two disjoint acquisition modalities.
- Fragment surfaces retain **discriminative colour and texture** — assumes minimal wear.
- Test data is tiny: **two vessels of three fragments each** for RAO; the CAF benchmark is 80
  fragments in 8 classes from the public Ceramic Sherd Database (Drexel / NEC Labs, 2010).
- The whole open boundary is treated as matchable; **thick fragments with distinct fracture
  surfaces are not modelled at all**.
- The alignment derivation assumes a single **rotation about z** suffices.

## 2.4 Results and limitations

| Metric | Value |
|---|---|
| CAF accuracy, 80 fragments, 8 classes | **96.1 %** |
| Baseline SIFT (Smith et al. 2010), same data | 76 % |
| Baseline TVG (Smith et al. 2010), same data | 75 % |
| Worst CAF class (Class B, 9 pieces) | 89 % |
| RAO test set size | 2 vessels × 3 fragments, plus a 7-fragment Nabataean vessel |
| Claimed RAO precision | "100 % precision" |
| NN training | 335 epochs |

Limitations, mostly unacknowledged by the authors:

- **The 100 % reconstruction claim rests on 2 vessels of 3 fragments plus one 7-fragment vessel.**
  It is not a meaningful number.
- **Restricting rotation to `Rz` is geometrically wrong.** A general fragment pairing needs full
  SO(3); the paper justifies z-only rotation as "through a number of experiments, the best result
  can be achieved". This will not generalise.
- **The 5-point sub-contour is far too short** to be a discriminative shape descriptor on a mesh with
  hundreds of thousands of vertices, and the descriptor is dominated by raw coordinate statistics
  (min/max/mean/var of x, y, z), which are **not rigid-invariant**. Any change of fragment pose
  changes the descriptor.
- **The network's output layer is sized to the number of candidate parts (8),** so it must be
  retrained for every new fragment set. It is not a reusable model.
- No fracture-surface segmentation, no handling of foreign fragments, no global assembly — matching
  is strictly pairwise and greedy.
- The paper's own literature review is sloppy (garbled citations, e.g. "Guoguang et al." for a paper
  whose first author's given name is Guoguang).

## 2.5 Relevance verdict — **2 / 5**

The overall architecture (cluster first, then match within cluster) matches our two-tier problem, and
one component is directly reusable. But the matcher requires per-set NN training, the descriptor is
not rigid-invariant, the rotation model is wrong, and the validation is too small to trust. Do not
re-implement the pipeline.

## 2.6 Concrete reusable ideas

1. **Cluster before matching.** For the "dozens of fragments, possibly from other objects" case,
   run a cheap grouping pass first so the O(N²) fracture-surface matcher only ever sees within-group
   pairs. Their colour-intersection statistic is crude, but our PLY files carry per-vertex colour, so
   a **fabric-colour histogram distance plus a wall-thickness statistic** would do the same job far
   more robustly and at negligible cost.
2. **The open-boundary edge test (step 8) is exactly right and trivially implementable**: build the
   edge→face incidence map, keep edges incident to exactly one face. On a watertight-ish Geomagic
   mesh this is one `trimesh` call and it gives the boundary loops for free.
3. **Sub-contour segmentation of the break curve into overlapping windows** is the right shape for a
   descriptor-based break-curve matcher — but the window should be metric (e.g. 5–15 mm of
   arclength) rather than a fixed 5 points, and the descriptor should be **rigid-invariant (κ, τ)**
   rather than coordinate statistics.
4. **PCA/SVD canonical frame per fragment** (`H = Σ (p_i − centroid)`, `svd`, align to principal
   axes) is a cheap, deterministic pose normalisation that shrinks the pairwise search space before
   any expensive alignment. Worth keeping, with the caveat that their formulation of `H` as a sum of
   vectors rather than a scatter matrix `Σ (p−c)(p−c)ᵀ` is almost certainly a typo in the paper —
   implement the scatter matrix.
5. **LBP over the fabric texture** is a plausible secondary discriminator for separating fragments of
   different objects, computable on rendered surface patches, CPU-cheap.

---

# PART 3 — Barreau et al. (2014), *Photogrammetry Based Study of Ceramics Fragments*

*International Journal of Heritage in the Digital Era 3(4): 643–656. DOI 10.1260/2047-4970.3.4.643.
HAL: hal-01394971.*

## 3.1 What it does

A field-practice / workflow paper, not an algorithms paper. Two French sites (Iron Age Rezé,
Bronze Age Lannion Penn An Alé) supplied eight vessels' worth of fragments. The team built a
low-cost photogrammetry protocol (consumer DSLR, two light tables, one halogen lamp), reconstructed
each fragment as a textured mesh in Agisoft PhotoScan, then **reassembled and completed the vessels
entirely by hand in Blender**, computed volumes in MeshLab, and 3D-printed a physical display stand
that holds the surviving fragments in their correct relative positions. The contribution is that the
whole chain is affordable and reproducible by non-specialists; there is **no matching algorithm
anywhere in the paper**.

## 3.2 Pipeline with parameters

1. **Capture.** Nikon D60 DSLR, 10 MP CCD, 18–55 mm AF-S at 40 mm focal length, Multi-Cam 530 AF.
   Two light tables (light from below and above) plus a halogen lamp, white background. **Five
   orbital series per fragment: top, 45° above, horizontal, 45° below, bottom.** Each series is a
   semicircle; the object is then rotated 180° and the series repeated. Photo counts per vessel
   ranged **36 to 111**.
2. **Alignment.** PhotoScan 0.8.3 beta 64-bit. **Mask each photo** to clip the object from the
   background before matching. Accuracy setting **"strong"**. Maximum common points raised from the
   default **40 000 to 60 000**. Sparse clouds came out at **18 354 to 59 623 points**.
3. **Mesh.** Object type **"arbitrary"**, quality **"high"**, geometric precision **"sharp"**, target
   face count **250 000**. Vertex colour from the point cloud; texture from image patches with
   **"generic" mapping** and **"average" blending**. Resulting meshes: **17 911 to 250 000 faces**,
   **11 036 to 149 449 vertices**. Wall-clock **39 minutes to 1 h 15 min per fragment**.
4. **Scaling.** Export OBJ → MeshLab → measure a known real-world distance with the ruler tool →
   scale factor = real / measured → `Transform: scale` on x, y, z. Done as a separate step because
   the lighting was too uneven for PhotoScan's marker-based scaling to work.
5. **Reconstruction, Rezé (single-fragment vessels).** Manual, in Blender. If a fragment retains part
   of the base, lay it flat on the XY plane. Fit a **circle to the lip curvature** to recover the
   theoretical circumference. Repeat with further circles displaced along Z, each fitted to a
   different height of the surviving profile. The stack of circles is the vessel **"skeleton"**.
   Export to MeshLab, run **convex hull** (Remeshing/Simplification/Reconstruction) over the circle
   stack to close the shape, then **Compute Geometric Measures** for volume. Volumes obtained:
   **88.9 to 1938.7 cm³**.
6. **Reconstruction, Lannion (multi-fragment vessel).** Assemble lip-bearing fragments first onto a
   fitted circle, then import and place the remaining pieces **one by one, manually**. Delete
   handles, then run **Poisson surface reconstruction** in MeshLab to extrapolate the full vessel
   envelope. Volume **2742.9 cm³**.
7. **Display.** Print the envelope on a MakerBot Replicator 2x (ABS). The vessel was too large for
   the print bed and had to be split in two; the Poisson envelope did not match the fragments closely
   enough, so a designer corrected the mesh in 3ds Max.

## 3.3 Input assumptions

- Fragments are **large enough and thick enough to photograph reliably**. The paper is explicit that
  several small pieces failed: *"only the inner or outer part of the fragment was reconstituted"*,
  attributed to poor light, insufficient image resolution, or **"a too thin thickness of the ceramic
  fragments that corrupted the detection of common points on the edges"**. Their workaround was to
  rescan the inner and outer faces separately and merge them manually along Z.
- Vessels are **roughly rotationally symmetric** — the entire Rezé completion method is circle
  fitting about a Z axis.
- An archaeologist is available to do the assembly by hand.

## 3.4 Results and limitations

Seven Rezé vessels reconstructed from a single fragment each (Gr8B had three), one Lannion vessel
assembled from many fragments, one 3D-printed display. Volumes reported to 0.1 cm³, a precision the
method cannot possibly support. Limitations the authors themselves list: uneven illumination broke
the built-in scaling; thin fragment edges broke feature matching; Poisson reconstruction introduced
enough error at the fragment/envelope boundary that the printed stand did not fit; the printer was
too small. Their own stated next step is *"to extend these methods to other kinds of less symmetrical
archaeological fragmented artifacts"* — i.e. our exact case is future work for them.

## 3.5 Relevance verdict — **1 / 5**

No algorithm, no automation, no code, and the one completion technique it does describe (circle
stacks about a Z axis) assumes the symmetry our object lacks. Included only for the acquisition
lessons and the Poisson-envelope idea.

## 3.6 Concrete reusable ideas

1. **Poisson surface reconstruction over the union of already-placed fragments** as an *envelope
   prior*. Once a partial assembly exists, a Poisson/ball-pivoting envelope over the placed intact
   surfaces gives a smooth global shape that can be used to **score or veto candidate placements**
   of the remaining fragments — a symmetry-free substitute for the axis constraint. `open3d` has
   `create_from_point_cloud_poisson` and runs on CPU.
2. **Convex hull + `compute geometric measures`** as a fast sanity metric on a proposed assembly:
   the assembled hull volume and surface area should behave monotonically as correct fragments are
   added, and jump implausibly when a wrong one is placed.
3. **The thin-edge failure warning is a live risk for us.** Their common-point detection collapsed on
   thin fragment edges. Our fragments are thick (5–10 % of extent), so we are on the right side of
   this, but the same reasoning predicts that any *thin* chip in a later set will be badly scanned
   near its fracture rim and should be flagged, not trusted.
4. **The physical display stand** is a genuine museum deliverable worth mentioning to the client:
   once transforms are computed, printing a support that holds the real fragments in their computed
   relative poses is a small extra step with clear exhibition value.

---

# PART 4 — Barreau et al., *Ceramics Fragments Digitization by Photogrammetry, Reconstructions and Applications*

## 4.1 What it does

This is the **earlier/shorter preprint of the same study by the same eight authors**. Same two sites,
same seven Rezé vessels plus the Lannion vessel, same Nikon D60, same PhotoScan 0.8.3 settings, same
Table 1 with **byte-identical numbers** (photo counts, cloud sizes, face/vertex counts, calculation
times, volumes 536.4 / 1722.8 / 88.9 / 138.8 / 338.7 / 211.1 / 1938.7 cm³, and 2742.9 cm³ for
Lannion), same MakerBot display.

## 4.2 Differences from the 2014 IJHDE version

Only four, all minor:

- The camera-series description is compressed and the 2014 paper's explicit five-series list
  (top / 45° above / horizontal / 45° below / bottom) is replaced by a generic "other series from
  different angles".
- No dedicated "Process Overview" section and no process diagram.
- The archaeological context section is one paragraph instead of two.
- **One factual inconsistency:** its conclusion credits the 3D-printed display to the **Rezé**
  ceramics, whereas the body of both papers and the 2014 conclusion attribute it to **Lannion Penn
  An Alé**. The 2014 version is the one to cite.

## 4.3 Input assumptions, results, limitations

Identical to Part 3 in every respect. Same thin-edge scanning failure, same Poisson approximation
error at the fragment/envelope boundary, same oversized-print workaround, same closing wish to
extend the work to "less symmetrical archaeological fragmented artifacts".

## 4.4 Relevance verdict — **1 / 5**

No independent content. **Cite the 2014 IJHDE version and discard this one** to avoid
double-counting the evidence.

## 4.5 Concrete reusable ideas

None beyond Part 3. The only marginal addition is its Figure 2, which shows the camera-position
geometry as both a front view and a top view — slightly clearer than the 2014 figure if we ever
document a rescanning protocol for the museum.

---

# CROSS-PAPER SYNTHESIS

**Recommended family: fracture-surface (area) matching — Family C — with break-curve matching
(Family B) as the candidate generator and thickness (Family F) plus vertex colour as pruning
filters.**

Family C is the only family in the survey whose stated assumptions are ours: thick solid fragments,
genuine fracture surfaces, arbitrary non-symmetric geometry, no colour required, no training data.
Families A and E are ruled out by the object's asymmetry and the absence of a template — Family A is
where Structure-from-Sherds lives, and the survey states outright that axis estimation is ambiguous
for flat or small fragments and impossible for some. Family D targets thin frescoes. Family G needs
training data and a GPU we do not have.

The practical architecture the papers converge on: **segment intact vs. fracture surface → extract
break curves as the fracture-surface boundary → generate pairwise candidates cheaply from
rigid-invariant curve signatures → verify and refine each candidate by dense fracture-surface
alignment under a no-interpenetration constraint → assemble globally, greedily, best-score-first.**
Prune the O(N²) pair space first with wall thickness and fabric colour, which also solves the
foreign-fragment requirement. Papaioannou's 50 %-unconstrained / 90 %-constrained split is the
number to beat, and it says plainly that the material-overlap constraint is worth 40 points.

## Ranked list: 5 classical algorithms to consider re-implementing

1. **Huang, Flöry, Gelfand, Hofer & Pottmann (2006), SIGGRAPH — "Reassembling fractured objects by
   geometric matching."** Segment each fragment into original and fracture faces by multi-scale edge
   extraction followed by a graph cut, then describe fracture patches by clustered surface features
   selected via a forward search for robustness. Pairwise matches feed a global multi-piece matching
   stage with simultaneous local multi-piece registration, so pose errors do not accumulate along a
   greedy chain.

2. **Papaioannou, Karabassi & Theoharis (2002), IEEE TPAMI 24(1) — "Reconstruction of 3D objects
   through the matching of their parts."** Score a candidate relative pose by the pointwise distance
   between the *mutually visible* faces of the two fragments, integrated over the whole facing
   surface, and drive a global optimiser with that error as its cost function. Its earlier 2000
   variant contributes the second half: use **depth-buffer surface bumpiness to decide which sides
   are fracture sides**, pick the least irregular ones, and add material overlap as a hard
   constraint — the constraint that raised its success from 50 % to 90 %.

3. **Üçoluk & Toroslu (1999) — noise-tolerant break-curve matching.** Represent each fragment's
   boundary as a discrete 3D curve and compute **curvature and torsion at every sample point** as a
   rigid-invariant feature vector, so two mating curves have matching signature sequences regardless
   of pose. A noise-tolerant sequence-matching algorithm then finds the corresponding arcs and the
   rigid transform follows directly from the correspondence, with no symmetry assumption and no pose
   search.

4. **Vendrell-Vidal & Sánchez-Belenguer (2014), JOCCH 7(3), with Belenguer & Vidal (2012).**
   Discretise each fracture surface by uniform sampling into a fixed representation, define an
   alignment cost over those discrete samples, and search the pose space **hierarchically,
   coarse-to-fine**, which both guarantees convergence to the global optimum and reported a **250×
   speedup over exhaustive search**. Their speed came from GPU depth maps we cannot use, but the
   coarse-to-fine pose hierarchy is the transferable part and maps cleanly onto CPU KD-tree
   nearest-neighbour queries.

5. **Son, Almeida & Cooper (2013), CVPR — APC + break curve** (for the later pot and plate sets
   only). Estimate each fragment's axis-of-symmetry profile curve with **circle templates**, which
   stays stable on fragments that are almost flat, chipped or noisy, then reassemble by break-curve
   matching evaluated both locally and globally. Reported robust to noise, bumps and erosion, and
   assembled 48 fragments across three vases in 10.56 hours — slow, and it is the family that already
   failed us on the sculptural object, so hold it strictly in reserve. **Willis & Cooper (2004)** is
   the alternative in this slot if a probabilistic formulation is preferred: fit models to the outer
   surface and break curves and maximise the joint likelihood over both simultaneously, which the
   survey credits with skipping unlikely configurations cheaply.

**Honourable mention:** *Zhou et al. (2007)* is the only work the survey introduces as explicitly
solving "fragments with thickness" — internal plus external contour extraction on solid fractured
objects, junction-vertex matching, binary-tree assembly, plus a seam/hole repair step. Poorly
documented in the survey, but conceptually the closest fit to our thick-walled terracotta and worth
retrieving in full.

**Do not re-implement:** Rasheed & Nordin's neural sub-contour matcher (needs per-set training, uses
non-rigid-invariant descriptors, restricts rotation to the z axis, validated on 3-fragment vessels)
or anything from Barreau et al. (manual assembly, no algorithm).
