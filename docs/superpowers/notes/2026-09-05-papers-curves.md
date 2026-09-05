# Curve / Contour-Based Reassembly — Technical Analysis

Scope: three papers on breaking-curve and contour-curve driven fragment reassembly, analysed against
our target task (thick-walled terracotta fragments, coloured PLY, ~350k–670k vertices, CPU-only Mac,
no training data, 4 to dozens of fragments, possible missing pieces and intruders from other objects).

Conventions used below:
- **[paper]** = value or formula stated explicitly in the paper.
- **[derived]** = my inference from the paper's text/formulas.
- **[ours]** = a parameter value I recommend for our data; the paper gives none.

---

# PAPER 1 — Alagrami, Palmieri, Aslan, Pelillo, Vascon (2023), "Reassembling Broken Objects using Breaking Curves" (arXiv 2306.02782)

## 1. What it does

The paper attacks **pairwise** assembly of two 3D point clouds that are pieces of one broken object,
using only geometry, with no prior on object class, no template, no symmetry assumption and no
learning. Its core claim is a division of labour: the hard part is not registration but
**segmentation**, i.e. correctly identifying which subset of each fragment is the fracture surface.
It solves that by detecting *breaking curves* (sets of connected points lying on a sharp 3D edge),
using those curves as barriers for a region-growing segmentation, and then running plain off-the-shelf
ICP between candidate region pairs, picking the pair with the lowest Chamfer distance. The message is
that once fracture regions are isolated, a "dumb" registration method suffices, whereas ICP on the raw
fragments and a learned method (DGL) both fail badly.

## 2. Pipeline, step by step

### 2.1 Graph construction over the point cloud

Point cloud `P`, represented as an unweighted directed graph `G = (V, E)`; `V` = points, `E` =
neighbour relations. Because point density is non-uniform they use a **mixed ε-graph + kNN** rule
**[paper]**: the radius ε is the mean k-nearest-neighbour distance over the *entire* cloud:

```
ε = (1/|P|) * (1/k) * Σ_{p∈P} Σ_{q∈N_k(p)} |p − q|
```

where `N_k(p)` is the set of k nearest neighbours of `p`. So ε is a single global scalar equal to the
average kNN edge length. **[derived]** Each point then connects to neighbours within ε (with kNN as
the fallback where the cloud is sparse). `k` is not given numerically **[paper gives no value]**.

### 2.2 Breaking-curve detection: the actual criterion

**Important correction to a common assumption: this paper does NOT use a dihedral-angle criterion and
does not touch mesh faces at all.** It works purely on points, via the Gumhold et al. (2001)
**corner penalty**:

```
ω_co(p) = ( λ2(p) − λ0(p) ) / λ2(p)          [paper, eq. unnumbered]
```

where `λ0 ≤ λ1 ≤ λ2` are the eigenvalues of the correlation (covariance) matrix of the neighbours of
`p`. Equivalently `ω_co = 1 − λ0/λ2`.

Interpretation (note the paper's own prose is garbled here — it prints "λ2 ≈ λ1 and λ2 ≈ 0" for the
flat case, which is a typo; the surrounding sentences and the formula make the correct reading
unambiguous) **[derived]**:

| point type | eigenvalue pattern | ω_co |
|---|---|---|
| flat surface | λ0 ≈ 0, λ1 ≈ λ2 | ≈ 1 |
| edge / crease | λ0 small but nonzero | intermediate |
| corner | λ0 ≈ λ1 ≈ λ2 | ≈ 0 |

**Selection rule [paper]: keep all nodes whose corner penalty is *less than* a threshold.** Low ω_co
= edge/corner candidate. The threshold value is not given **[paper gives no value]** and the paper
states they used *different parameter sets for synthetic vs real objects*.

**Refinement [paper]:** the raw thresholded set is a noisy breaking curve. It is cleaned with an
operation analogous to **morphological opening on the graph**: a **pruning** step followed by a
**dilation** step, to remove small isolated branches and to promote *closed* curves. No iteration
counts or structuring-element sizes are given.

Result: `B^P ⊂ P`, the set of breaking-curve points of cloud `P`.

### 2.3 Region segmentation

Region growing **constrained by the breaking curves [paper]**:

1. Pick a seed `p ∉ B^P`; create region `R_i^P`, assign `p`.
2. For each neighbour `q ∈ N(p)`: if `q ∉ B^P`, add `q` to `R_i^P`.
3. Iterate until all points not on a breaking curve are assigned.

The curves therefore act as walls; each connected non-curve component becomes a region. For a
fracture surface bounded by a closed breaking curve, that region *is* the fracture surface.

4. **Breaking-curve points themselves** are then assigned by **k-NN majority voting [paper]**: a curve
   point joins the region owning the majority of its neighbours. The paper explicitly notes the curve
   shape itself is a useful matching cue, which is why they fold curve points back into regions rather
   than discarding them.

### 2.4 Region matching and registration

1. **Size filter [paper]:** discard regions whose node count is below a threshold (value not given).
   Stated purpose: cut computation and suppress noisy regions.
2. **Exhaustive search [paper]:** for two segmented clouds `P`, `Q`, evaluate every pair
   `(R_i^P, R_j^Q)` of surviving regions.
3. **Registration [paper]:** run standard **ICP** (Besl & McKay 1992; Open3D implementation used for
   the baseline) on each region pair.
4. **Score [paper]: Chamfer Distance** between the registered regions.
5. **Selection [paper]:** the pair with the best (lowest) Chamfer distance wins; its rigid transform
   is applied to the whole fragment as the final alignment.

There is **no** additional verification test (no penetration test, no normal-opposition test, no area
or perimeter compatibility test, no RANSAC). The Chamfer distance is the only rejection criterion, and
it is used only to rank, never to reject outright.

### 2.5 Multi-fragment assembly

**None.** The method is strictly pairwise. The conclusion says extension to multiple parts "following a
greedy approach is under exploration", and lists "designing more principled ways of selecting the best
registration among many pairs" and "detecting non-matching surfaces" as future work.

### 2.6 Code and datasets

- Code: **`https://github.com/RePAIRProject/AAFR`** — stated as "the code will be released" **[paper]**.
- **Breaking Bad** dataset (Sellán, Chen, Wu, Garg, Jacobson, NeurIPS 2022 Datasets & Benchmarks) —
  synthetic, physically realistic fracture; categories used: BeerBottle, WineBottle, DrinkBottle,
  Bottle, Mug, Cookie, Mirror, ToyFigure, Statue, Vase.
- **TU-Wien dataset** (Huang et al. 2006) — one sample, a scanned *Brick*.
- **RePAIR project** (EU H2020 grant 964854, `repairproject.eu`) — in-house scanned **Pompeii fresco
  fragments**.

## 3. Input assumptions

| assumption | status |
|---|---|
| thin vs thick | **Neither is assumed.** Works on solids (Statue, ToyFigure, Brick) and on slab-like fresco fragments. Requires only that a fracture surface exists and is bounded by detectable sharp edges. |
| axial symmetry | **Not required.** Explicitly "agnostic on the type of object", no shape prior. |
| texture / colour | **Not used at all.** Geometry only. |
| training data | **None.** Fully unsupervised/geometric. |
| templates / complete object | **Not needed.** Explicitly assembles incomplete broken parts "with no need for the complete object reconstruction". |
| manual interaction | **None** in the pipeline, but **parameter tuning per dataset is required** (their own stated limitation). |
| scan resolution | Not stated. The ε-graph auto-adapts to point spacing, so the method is scale-free in that respect; but the corner-penalty threshold and the min-region-size threshold are resolution-dependent. |
| scale | Fragments assumed in a common metric scale (both clouds same units). No scale estimation. |
| input format | **Point clouds**, not meshes. Faces, normals and connectivity are never used. |

## 4. Results, datasets, runtime, limitations

Metric **[paper]**: relative rotation RMSE (degrees) and translation RMSE, following Sellán et al.
Table 1, "ours" column:

| category | rot RMSE | trans RMSE |
|---|---|---|
| BeerBottle | 1.62 | 0.020 |
| WineBottle | 1.58 | 0.020 |
| DrinkBottle | 1.89 | 0.033 |
| Bottle | 1.983 | 0.077 |
| Mug | 1.12 | 0.025 |
| Cookie | 1.96 | 0.043 |
| Mirror | 0.111 | 0.001 |
| ToyFigure | 1.98 | 0.079 |
| Statue | 0.66 | 0.003 |
| Vase | 0.592 | 0.002 |
| Brick (real scan, TU-Wien) | 3.064 | 0.626 |
| RePAIR fresco (real scan) | 3.466 | 0.695 |

Baselines were far worse: ICP rotation RMSE 0.593–208.3, DGL 62.8–89.6. Note the two **real scanned**
rows are the worst for the proposed method (3.06° and 3.47° rotation, and 0.63/0.70 translation), i.e.
real scan noise costs roughly 2× the rotation error of synthetic data.

- **Dataset size:** pairs only. No count of pairs per category is given. One TU-Wien sample. A small
  in-house RePAIR fresco set.
- **Runtime: not reported anywhere.** No complexity analysis, no hardware description.
- **Ground truth** for the two real datasets came from *manual* alignment.
- **Stated limitations [paper]:**
  1. "The proposed pipeline is sensitive to the choice of the parameters" — different parameter sets
     were needed for synthetic vs real objects.
  2. Pairwise only; multi-part is future work.
  3. Non-matching surfaces are not detected (no reject option) — a critical gap for our large sets
     with intruder fragments.
  4. Low ICP translation error can coincide with a completely wrong solution: for Mirror, Cup and
     RePAIR, ICP achieved low translation error simply because the parts fully overlap in a nonsense
     configuration (Figure 1e). **Translation error alone is not a valid quality signal.**

## 5. Relevance verdict: **4 / 5**

Reasons for the high score: it is the only one of the three papers that is (a) fully automatic,
(b) free of any shape/symmetry prior, (c) free of training data, (d) built from primitives we already
have on CPU (kNN graph, per-point covariance eigendecomposition, region growing, ICP, Chamfer
distance) and (e) accompanied by public code. Its central insight — segment the fracture surface
first, then registration is easy — maps exactly onto our thick-walled terracotta, where the breaking
curve is precisely the rim where the fracture surface meets the intact outer/inner surface. Thick
walls actually help: they give a wide, geometrically rich fracture region, unlike the paper's fresco
fragments.

Reasons it is not 5/5: it stops at two fragments, has no rejection mechanism for non-matching
surfaces, reports no runtime, and admits parameter sensitivity. All three of those are exactly the
things our large mixed sets need. It gives us the front half of the pipeline, not the back half.

## 6. Concrete reusable ideas

**Borrow:**

1. **Corner penalty `ω_co = 1 − λ0/λ2` from per-point covariance eigenvalues.** Cheap, vectorisable
   over all points in one `numpy` einsum after an Open3D kNN query. This is our breaking-curve
   detector. It needs no mesh connectivity, so it survives the Geomagic remeshing artefacts.
2. **Global adaptive ε = mean kNN distance over the whole cloud.** A single scale parameter derived
   from the data, not hand-set. Reuse it as the unit for every downstream distance threshold (ICP
   correspondence distance, inlier radius, dilation radius), which makes the pipeline scale-free.
3. **Breaking curves as barriers for region growing, not as the matching primitive itself.** This is
   the paper's real contribution and it is more robust than matching curves directly.
4. **k-NN majority vote to reassign curve points to regions.** Prevents a one-point-wide gap of
   unlabelled points at every region boundary, and keeps the curve geometry inside the region that
   ICP will use.
5. **Morphological opening (prune then dilate) on the thresholded edge point set** to kill isolated
   branches and close curves. On our data this is the difference between a closed fracture rim and a
   fragmented dotted line.
6. **Region-size threshold before matching.** Both a speed and a robustness measure.
7. **Chamfer distance as the pair score after ICP.** Symmetric, cheap with a KD-tree, and directly
   comparable across region pairs of different sizes if normalised.

**Recommended concrete values [ours, the paper gives none]:**
- Voxel-downsample each fragment to ~40k–80k points before the graph step (from 350k–670k). At
  ~0.5 mm voxel size this preserves fracture-surface relief while making per-point eigendecomposition
  a few seconds.
- `k = 25` (range 20–30) for the kNN covariance.
- Corner-penalty threshold: **do not hard-code it**. Take the lowest **5–10th percentile** of `ω_co`
  per fragment as edge candidates. This auto-adapts across fragments and directly addresses the
  paper's own parameter-sensitivity limitation.
- Min region size: max(500 points, 1.5% of fragment points).

**Avoid:**

- **Do not use their exhaustive all-region-pairs ICP as-is at our scale.** For N fragments with R
  regions each the cost is `C(N,2)·R²` ICP runs. With N = 30 and R = 5 that is 10,875 ICP calls; even
  at 0.3 s each that is nearly an hour. We must prefilter pairs (see synthesis).
- **Do not trust translation RMSE or a single Chamfer number as an accept/reject test.** The paper's
  own Figure 1e case shows a fully-overlapping degenerate solution scoring well. We need
  interpenetration and normal-opposition checks on top.
- **Do not adopt their per-dataset manual parameter retuning.** Replace fixed thresholds with
  percentile/data-derived ones.
- Do not discard mesh connectivity just because they did. We have watertight-ish meshes, so we can
  additionally use face-normal dihedral angles and geodesic connectivity to make the breaking-curve
  extraction far more robust than their point-only version.

---

# PAPER 2 — Zheng, Huang, Li, Wang (2014), "Reassembling 3D Thin Fragments of Unknown Geometry in Cultural Heritage" (ISPRS Annals II-5, 393–399)

## 1. What it does

This paper targets **thin** shards (a broken bowl) where there is essentially no fracture *surface* to
match, only the fracture *curve*: the outer boundary contour of the shard. It solves three stated
problems in prior art: (i) numerical instability of curvature/torsion-based curve descriptors, which
need up to third-order derivatives; (ii) the geometric prior of axial symmetry imposed by pot-oriented
methods; (iii) error accumulation from purely pairwise assembly. Its answer is to attach a **local
Cartesian frame** to every contour point (built from the surface normal and the curve tangent), so
that a single hypothesised point correspondence directly yields a full rigid transform with no
derivatives beyond first order; to score hypotheses with an arc-length-over-residual likelihood; to
assemble all fragments by a **maximum-weight spanning tree** over a pairwise-match graph; and finally
to remove accumulated error with a **bundle-adjustment-style global least-squares refinement** over all
fragments simultaneously.

## 2. Pipeline, step by step

### 2.1 Data acquisition

Structured-light scanning (Zheng et al. 2012 process) → point cloud → triangulated mesh → **contour
curves extracted from the outer contours of the mesh** (i.e. mesh boundary edges) **[paper]**. Two
explicit remarks: they keep the **original contour points** as the curve representation rather than
curvature/torsion strings; and although only contours are matched, the **surrounding surface points are
still needed** to estimate the normals at contour points and to render the final assembly.

### 2.2 Local Cartesian coordinate frame at each contour point **[paper, Figure 3]**

For contour point `p`:
- **Origin** = `p`.
- **z axis** = the surface **normal** at `p`, obtained during mesh reconstruction.
- **x axis** = the curve **tangent** at `p`, estimated by **polynomial fitting to neighbouring curve
  points**. Tangent directions along one contour are consistently oriented so that the contour runs
  **counterclockwise around the region of the corresponding point cloud** — this fixes the sign
  ambiguity and is what makes the frames comparable across fragments.
- **y axis** = `z × x` by the right-hand rule.

Neighbourhood size for the polynomial fit and the polynomial degree are **not given [paper gives no
value]**.

### 2.3 Pairwise contour matching **[paper, steps (1)–(5)]**

For a pair of fragments with contour curves `C_A`, `C_B`:

1. Hypothesise that contour point `a ∈ C_A` corresponds to `b ∈ C_B`. The rigid transform is read
   directly off the two local frames: `T = F_b ∘ F_a^{-1}` **[derived from paper's step (1)]**. Each
   hypothesis costs O(1); there are `|C_A| · |C_B|` of them.
2. Apply `T` to bring both curves into the same coordinate system.
3. For each point `P_i` on one transformed curve, find the nearest point `Q_i` on the other, compute
   `d_i = |P_i − Q_i|`. A **fixed distance threshold** splits the points into inliers `I` and
   outliers `O`. Threshold value **not given [paper gives no value]**.
4. Compute `L_i`, the **arc length along the trajectory of the inliers** (i.e. how much contiguous
   contour actually mates, not merely how many points).
5. Similarity score **[paper, eq. (1)]**:

```
              L_i                          L_i
G_i = ─────────────────────  =  ───────────────────────────────
       ( Σ_{Pj∈I} d_j² )/N + c    ( Σ_{Pj∈I} ‖P_j − Q_j‖² )/N + c
```

with **`c = 0.3` (a constant, explicitly stated) [paper]** and `N` = number of inliers.
Reading: reward long contiguous contact, penalise mean squared residual, and `c` both prevents
division by zero and stops a tiny near-perfect match from scoring infinitely.

6. **Matching degree [paper, eq. (2)]:** `D = max_i G_i` over all hypothesised correspondences `i`.
   The argmax hypothesis simultaneously yields the **maximum-likelihood alignment** for that pair —
   no separate initial-alignment stage is needed.

### 2.4 Multi-fragment assembly: maximum-weight spanning tree **[paper, step (6)]**

- Build a graph: **nodes = individual contour curves / fragments**, **every pair of nodes joined by an
  edge weighted by `D`**.
- Observe that **exactly `n − 1` pairwise matches** suffice to place `n` fragments into one common
  coordinate system.
- Compute the **maximum-weight spanning tree** of the graph. Compose the pairwise transforms along the
  tree to get each fragment's transform `(R_i, T_i)` into the common frame.

This is the paper's answer to "how do pairwise results become a global assembly", and it is the single
most directly reusable piece of the paper for us.

### 2.5 Global refinement: bundle adjustment over all fragments **[paper, Section 4]**

Model **[paper, eqs. (3),(4)]**: `P_g = R_i P_l + T_i`, `n_g = R_i n_l` (normals rotate only).

Cross-fragment lookup **[paper, eq. (5)]**: to find, for a point of fragment i, its nearest point in
fragment j's own coordinate system,

```
P_l^(j) = R_j^T ( R_i P_l^(i) + T_i − T_j )
```

Objective **[paper, eq. (6)]** — minimise over all `{R_i, T_i}` jointly:

```
C({R_i, T_i}) = Σ_{i≠j} [  w ‖P_g^(i) − P_g^(j)‖²  +  (1 − w) ‖n_g^(i) − n_g^(j)‖²  ]

              = Σ_{i≠j} [  w ‖R_i P_l^(i) − R_j P_l^(j) + T_i − T_j‖²
                         + (1 − w) ‖R_i n_l^(i) − R_j n_l^(j)‖²  ]
```

with **`w ∈ (0, 1]`** **[paper; no numeric value given]**. Note the objective mixes a **point-to-point
term** and a **normal-agreement term**; the normal term is what stops fragments sliding along the
contact and is why it beats plain ICP composition.

Iteration **[paper, steps (1)–(3)]**:
1. **Update the matching**: for every point `P_l^(i)` on one contour, search nearest points on all
   other contours; **discard nearest pairs whose distance exceeds a given threshold** (value not
   given). This is the outlier rejection.
2. **Error adjustment**: least-squares minimisation of eq. (6) with the updated correspondences.
3. **Convergence check**: if all adjusted rigid transforms changed by less than a tolerance versus the
   previous iteration, stop; otherwise go to 1.

This is essentially a **multi-way, normal-aware ICP solved as one global least-squares problem**,
i.e. pose-graph/bundle adjustment rather than sequential pairwise ICP.

## 3. Input assumptions

| assumption | status |
|---|---|
| thin vs thick | **Thin is assumed and is the entire premise.** "Only the complementary among the contour curves of the fragments can be utilized", because thin shards lack a usable matching surface. Explicitly contrasts thin vs thick as two separate research streams. |
| axial symmetry | **Explicitly rejected.** A stated goal is to be "free from any geometry assumption of original shape". Test object is a bowl, but symmetry is never exploited. |
| texture / colour | **Not used.** Listed as *future work*: needed "in case the fragments are complete unorganized… to determine whether the fragments can be reassembled or not, and to adapt to the case of the fragments of more than one objects". |
| training data / templates | **None.** |
| manual interaction | **None described.** Fully automatic given the scans. |
| scan resolution | Structured-light scanner assumed, dense enough that (a) normals can be estimated from the mesh and (b) contour point spacing is fine relative to fracture detail. Final accuracy 0.47 mm implies sub-mm scanning. |
| scale | Common metric scale assumed; results reported in mm. No scale estimation. |
| other | Assumes all fragments belong to **one** object (spanning tree spans everything) and that a **connected** matching graph exists. Missing pieces are not discussed. |

## 4. Results, datasets, runtime, limitations

- **Datasets [paper]:** a broken bowl used for illustration; **Data I = 4 fragments**; **Data II = 12
  fragments**. Data II had a **pre-break 3D model available as ground truth**.
- **Accuracy [paper]:** reassembled model ICP-aligned to the pre-break model; point-to-surface
  distances; **RMS = 0.47 mm** for Data II. Error distribution is explicitly **non-normal**, and the
  paper attributes this to the fact that "the contour curves can provide very little information to
  the reassembling of the fragments".
- **Runtime: not reported.** No hardware, no timings.
- **Stated limitations [paper]:**
  1. "The computation will increase with the increasing number of the fragments" — they call for an
     improved pairwise initial matching strategy. **[derived]** the cost is O(n² · m²) in fragment
     count n and contour points m, since every point pair of every fragment pair is a hypothesis.
  2. Cannot handle fully unorganised fragment sets or **mixtures of more than one object** without
     additional cues (texture, colour).
  3. Contour curves alone are information-poor — this is the root cause of the non-normal error
     distribution.

## 5. Relevance verdict: **3 / 5**

Reasons for the middling score: the **back half** of this paper is exactly what our task needs and what
Paper 1 lacks. The maximum-weight spanning tree over a pairwise-score graph is a clean, cheap,
implementable answer to multi-fragment assembly with `n − 1` transforms; the global bundle-adjustment
refinement with a normal-agreement term is the right fix for drift in dozens-of-fragment sets; and both
run comfortably on CPU. The single-correspondence-to-rigid-transform trick via local frames is also
directly transplantable to our breaking curves.

Reasons it is not higher: the **front half is built on an assumption we do not satisfy**. Its matching
cue is the *outer boundary contour of a thin shard*, where the shard is effectively a surface patch and
the boundary is the whole story. Our fragments are thick-walled with genuine fracture surfaces, so
matching only a contour throws away most of our signal and would inherit the paper's own complaint that
"contour curves provide very little information". Its O(n²m²) hypothesis enumeration is also far too
slow for dozens of fragments at our point counts. And it explicitly cannot handle missing pieces or
intruder fragments from other objects, which our large sets have.

## 6. Concrete reusable ideas

**Borrow:**

1. **Local Cartesian frame from (surface normal, curve tangent) at each curve point**, with the tangent
   consistently oriented so the curve runs counterclockwise about its own region. One correspondence →
   one full rigid transform, no RANSAC triples needed. For us: build these frames on **breaking-curve
   points** (Paper 1's `B^P`), not on outer mesh boundaries.
2. **Avoid curvature/torsion descriptors.** Their argument is sound and empirical: third-order
   derivatives on scan data are numerically unstable. Use point positions plus first-order frames.
3. **The similarity score `G = L / (mean squared residual + c)` with `c = 0.3`.** The key design idea
   is the **numerator being contiguous inlier arc length, not inlier count** — it rewards a long
   continuous mating seam and penalises scattered accidental agreement. Rescale `c` to our units.
4. **`D = max over hypotheses` as the pair score, and the argmax hypothesis doubles as the initial
   alignment.** No separate coarse-alignment stage.
5. **Maximum-weight spanning tree over the fragment graph for global assembly.** `scipy.sparse.csgraph.
   minimum_spanning_tree` on negated weights; then compose transforms along the tree from a root.
   O(n²) edges, trivial cost.
6. **Global bundle-adjustment refinement, eq. (6), with joint point + normal residuals**, iterating
   correspondence update → least squares → convergence check on transform deltas. This is the correct
   replacement for chaining pairwise ICPs and is what took them from visibly wrong to 0.47 mm.
7. **Reject correspondences beyond a distance threshold at every refinement iteration** — a trimmed/
   robust ICP, essential when pieces are missing.

**Recommended concrete values [ours]:**
- `w ≈ 0.7`–`0.9` for the point term in eq. (6), with **normals scaled by a length constant** (e.g.
  0.5 × mean point spacing) so the two residual terms are dimensionally commensurate. The paper gives
  no value and the two terms have different units as written.
- Inlier / correspondence-rejection threshold ≈ **1.5–3 × mean point spacing**, tightened over
  iterations (e.g. start 5×, end 1.5×).
- Subsample candidate correspondences: use every ~10th breaking-curve point as a hypothesis anchor
  rather than all pairs.

**Avoid:**

- **Do not enumerate all `|C_A| · |C_B|` point-pair hypotheses.** At our resolution a fracture rim can
  have thousands of points; the full product across dozens of fragments is intractable on CPU. Prune
  with descriptor prefilters (see synthesis).
- **Do not use a spanning tree that must span every fragment.** It forces every fragment, including
  intruders from other objects and pieces whose true neighbours are missing, into the assembly. Use a
  **thresholded maximum-weight forest** instead: drop edges below an absolute score floor, then take
  the MST of each connected component and report multiple clusters.
- **Do not match on outer mesh boundary contours.** For our thick fragments that boundary is a scan
  artefact of unscanned/occluded regions, not the fracture. Use breaking curves instead.
- Do not rely on contour matching alone; the paper itself blames contour information-poverty for its
  error distribution.

---

# PAPER 3 — Kotoula (2016), "Semiautomatic Fragments Matching and Virtual Reconstruction: A Case Study on Ceramics" (Int. J. Conservation Science 7(1), 71–86)

## 1. What it does

This is a **conservation-science methodology paper, not an algorithms paper.** It compares manual
physical refitting against three existing pieces of *semi-automatic* software (MeshLab, Fragments
Reassembler from VCG ISTI-CNR, and 3ds Max) on real ceramic material, and recommends a combined
workflow: use a 3D DCC package for project management, metrics and initial categorisation, then use
Fragments Reassembler for the actual alignment. It also quantifies, from video transcripts of a
conservator at work, **which cues actually produce successful matches** — a genuinely useful empirical
result — and demonstrates virtual restoration (modelling missing parts, non-photorealistic rendering to
mark additions).

## 2. Pipeline, step by step

There is **no proposed algorithm**. What the paper documents is a human-in-the-loop workflow plus
descriptions of the three tools' mechanics.

**Digitisation [paper]:** CT scanning for the maiolica sherds (chosen because it also captures internal
structure and voids); **photogrammetry with Agisoft PhotoScan** for the amphora fragment and vessels.

**The recommended semi-automatic workflow [paper]:**
1. Import all fragments into 3ds Max; use scene explorer / summary statistics for management (object
   count, vertex and face counts, per-object naming and grouping).
2. **Compute each fragment's volume** with the measure utility; sum them; **compare against the
   estimated volume of a complete pot of the same type** to quantify how much material is missing.
   Result for Jug 367: **~30% of the vessel's volume is missing** — known *before* any matching is
   attempted, which sets expectations for the reconstruction.
3. **Sort fragments by volume** and **group them by shape into part categories** via MaxScript.
   Jug 367's 28 fragments were categorised as: 3 base, 2 rim, 2 handle, 3 upper body, 1 body (for the
   larger fragments; smaller ones were left uncategorised because shape-based typing is only reliable
   for large pieces). Volumes ranged from **39,209.72 down to 875.53** (arbitrary metric units), Table 2.
4. **Align within groups**, forming 5 sub-assemblies (body, base, rim, handle, upper body), using
   Fragments Reassembler.
5. **Align the groups to each other**, producing the full pot.
6. Remaining small fragments are placed using **curvature and profile** as the cue.

**Alignment mechanics of each tool [paper]:**
- **MeshLab align tool:** user picks **at least 4 point pairs** between a fixed mesh and a moving mesh;
  the algorithm solves for the best rigid transform through those correspondences. Criticisms: points
  cannot be edited (mistake ⇒ restart), fails when there are voids between fragments (i.e. worn edges,
  "almost always the case with archaeological fragments"), materials/colour are not visualised during
  alignment, and it degrades with many fragments or high-resolution meshes.
- **Fragments Reassembler** (Palmas, Pietroni, Cignoni, Scopigno, Digital Heritage 2013): finds the best
  match between two fragments **subject to user-supplied constraints**, using a **global energy
  minimisation that considers all pieces at once**, organised **hierarchically — two matched fragments
  form a group to which a third fragment or group can be attached**. The user can move the initial
  points, which lets it align severely damaged or eroded fragments. Judged **the most efficient for
  alignment**; weakest for project management; beta-quality.
- **3ds Max:** best for management, **inefficient for alignment**; precise relative positioning is hard
  and the needed matching scripts do not exist.

**Empirical cue statistics from manual matching (Jug 366, 17 sherds) [paper, Figs. 5–6]:**
- Actions: observation → selection of candidates → **testing joins (about half of all actions)** →
  matching → temporarily securing joins.
- Criteria used: shape; painted design/colour; texture; texture+shape.
- **60% of successful matches came from shape.**
- **20% came from painted design/colour.**
- The remainder came from texture and texture+shape, and those were mainly fallbacks after repeated
  failures on shape.
- 1 of the 17 sherds definitively belonged to **another pot** — i.e. intruder fragments are the norm,
  not an edge case.

## 3. Input assumptions

| assumption | status |
|---|---|
| thin vs thick | Thin-walled ceramic sherds (maiolica jug, skyphos, amphora). Not analysed as a variable. |
| axial symmetry | Implicitly assumed by the workflow: fragments are typed as base / rim / handle / body, and **"the curvature and profile assisted the correct positioning"**. This is wheel-thrown-pot reasoning. |
| texture / colour | **Yes, used and important.** Painted design accounts for 20% of manual matches. Also "ceramic wheel marks" were used as an alignment cue. The paper criticises MeshLab precisely for not showing materials during alignment. |
| training data | None, but **prior typological knowledge of the vessel form** is assumed (a complete pot of similar type is needed to estimate missing volume). |
| manual interaction | **Required throughout.** Every method is semi-automatic at best: MeshLab needs ≥4 hand-picked point pairs; Fragments Reassembler needs user constraints. |
| scan resolution | CT and photogrammetry; explicitly notes that **high-resolution meshes must be decimated** to be workable, and recommends flattening matched groups into a single exported mesh to reduce the piece count. |
| scale | Metric scale needed for the volume bookkeeping. |

## 4. Results, datasets, runtime, limitations

- **Datasets:** Jug No 366 — **17 sherds**, matched manually, **1 unassigned (belongs to a different
  pot)**, with lower-body fragments and small base/rim fragments **missing**. Jug No 367 — **28
  fragments**, matched digitally, **~30% of volume missing**, all but **2** placed; the 2 failures are
  attributed to **severely damaged edges**. Also a black-glazed chous, a Gnathian skyphos and a neopunic
  amphora fragment for the virtual-restoration demonstration.
- **Validation:** the virtual reconstruction of Jug 367 was afterwards **checked physically** by
  traditional manual refitting and gave **exactly the same fragment identifications** — a real,
  if small-n, end-to-end validation.
- **Runtime:** no numbers. Qualitative only: manual matching is "painstaking, time- and
  space-consuming"; MeshLab is "much more time consuming"; digital refitting of high-resolution meshes
  with many fragments "slows down the process considerably" and forces resolution reduction.
- **Stated limitations:** every tool needs an expert operator; MeshLab cannot cope with voids/worn
  edges; Fragments Reassembler is beta and weak at project management; 3ds Max cannot align; the
  computational cost of high-resolution multi-fragment refitting is the main bottleneck; the level of
  expertise required "goes beyond the field of conservation".

## 5. Relevance verdict: **1 / 5**

Reasons: it contributes **no implementable algorithm, no formula, no threshold, and no automatic
method** — every workflow it evaluates requires a human picking correspondences or constraints in a
GUI, which our GUI-less, autonomous requirement rules out entirely. Its case material is thin
wheel-thrown pottery and its placement logic leans on rim/base/handle typology and profile curvature,
which does not transfer to a non-axially-symmetric sculptural object.

What keeps it above 0: it supplies well-grounded *engineering expectations and workflow structure* that
we would otherwise have to guess at. The 60%-shape / 20%-colour split tells us how to weight cues; the
hierarchical group-then-merge strategy is the right control flow for dozens of fragments; the
missing-volume estimate is a cheap and genuinely useful pre-flight check; and the observation that 1
sherd in 17 came from another pot confirms intruder rejection is a first-class requirement, not a
nicety.

## 6. Concrete reusable ideas

**Borrow:**

1. **Hierarchical group-then-merge assembly control flow.** Match pairs, freeze confident pairs into a
   rigid group, then treat the group as a single unit for subsequent matching. This is both the
   Fragments Reassembler design and the manual conservator's strategy, and it keeps the pairwise search
   space from exploding as the assembly grows. Combine with Paper 2's spanning tree: build the tree,
   then merge greedily along edges in descending score order, re-running global refinement per merge.
2. **Global energy minimisation over all pieces at once rather than sequential pairwise gluing.**
   Independently corroborates Paper 2's bundle adjustment.
3. **Volume bookkeeping as a cheap completeness check.** `trimesh` gives us watertight mesh volume for
   free. Sum of fragment volumes vs an estimate of the intact object gives a "% missing" figure that
   tells the operator whether a gap in the assembly is a failure or an genuinely absent piece.
4. **Sort and process fragments by descending volume/area.** Large fragments carry more reliable
   geometry and should seed the assembly; small ones are placed last against an already-rigid group.
   Directly mirrors both the manual and the digital case study.
5. **Shape first, colour second, as a cue hierarchy with roughly a 3:1 weighting.** Use colour/texture
   as a **tie-breaker and as a rejection test** (does the fragment's fabric colour match the group?),
   not as a primary matching signal. This is exactly how we should use our coloured PLYs, and it is the
   cheapest available filter against intruder fragments.
6. **Expect worn/eroded fracture edges and voids.** Both the tool criticisms and the 2 unplaced
   fragments trace to damaged edges. Our scoring must tolerate a gap at the seam rather than demanding
   exact contact — argues for a soft, robust residual and against a hard contact constraint.
7. **Decimate before matching, restore full resolution only for the final transform application.**
   Confirms our plan to work on downsampled clouds.

**Avoid:**

- Everything requiring a GUI or hand-picked correspondences (MeshLab's ≥4-point align, Fragments
  Reassembler's user constraints, MaxScript).
- **Profile/curvature-based placement and rim/base/handle typology.** Sound for wheel-thrown pots,
  useless for a non-axially-symmetric sculptural object with relief panels, which is our first test set.
- The 4-points-then-rigid-transform approach as a matching primitive; Paper 2's single-point-plus-frame
  formulation is strictly better because it is automatic.

---

# CROSS-PAPER SYNTHESIS

The papers divide the problem cleanly: Paper 1 owns segmentation, Paper 2 owns global assembly,
Paper 3 owns control flow.

For thick fragments the breaking curve should segment and prefilter, not match. Paper 2 admits contour
curves "provide very little information", which explains its non-normal error distribution; it matches
curves only because thin shards leave no alternative. Our thick walls do leave one. Follow Paper 1:
detect breaking curves via the corner penalty (1 minus the ratio of smallest to largest covariance
eigenvalue), use them as barriers for region growing to carve out the fracture surfaces, then match
those surfaces, which carry far more constraint than their rims.

The curve earns its keep three other ways.

- **Pairwise prefilter.** Closed rims have a length, enclosed area, bounding-box aspect and coarse
  turning-angle signature, and mating rims must agree on all of them. Screening on those cuts Paper 1's
  all-region-pairs ICP and Paper 2's quadratic-in-points enumeration to a shortlist.
- **Pose generator.** Paper 2's local frame from surface normal plus curve tangent turns one
  hypothesised curve-point correspondence into a full rigid transform, seeding ICP instead of letting
  it start blind, the exact failure Paper 1 documents.
- **Verification.** A correct fit puts the rims in contact along a long contiguous arc, so Paper 2's
  score of contiguous inlier arc length over mean squared residual is a seam-quality test that
  Paper 1's single Chamfer number cannot provide.

Assembly follows Paper 2's graph: score every pair, build a maximum-weight spanning forest with an
absolute score floor so intruders and orphans split into separate components, then finish with the
joint point-plus-normal bundle adjustment rather than chained pairwise ICP. Paper 3 supplies the
ordering: seed with the largest fragments, freeze confident pairs into groups, refine after each merge,
and use colour only to reject.
