# Web scan: state of the art in 3D fractured-object reassembly

Scope: what is *practical* for a CPU-only, no-training, no-GT pipeline on an Apple M2 Pro
(10 cores, 16 GB) in Python 3.12 with numpy/scipy/open3d 0.19/trimesh/scikit-learn.
Input: coloured PLY triangle meshes of thick-walled terracotta fragments,
~350k-670k vertices, ~1M faces, watertight, non-axially-symmetric.

Compiled 2026-09-05. All URLs verified live at that date unless noted.

---

## 0. Executive verdict

**Nothing off the shelf will solve this.** There is no maintained, CPU-runnable,
open-source library that takes a folder of fragment meshes and returns an assembly.
The two things that come closest are:

* **AAFR** (RePAIR project, ICPR 2024) — real Python/Open3D code, CPU-capable, but
  pairwise-only (2 fragments) and only 12 stars, effectively a research prototype.
* **GARF** (ICCV 2025) — genuinely strong on real scans, but hard-requires CUDA 12
  (`flash-attn`) and HDF5-formatted Breaking-Bad-style input.

The realistic path is to **reimplement the Huang et al. 2006 pipeline structure**
on top of Open3D primitives, substituting cheaper modern equivalents for the
expensive 2006 components. That is spelled out in §5.

---

## 1. Classical geometric methods

### 1.1 Huang, Flöry, Gelfand, Hofer, Pottmann (SIGGRAPH 2006) — the canonical reference

"Reassembling fractured objects by geometric matching", ACM TOG 25(3):569–578.

* Paper PDF: https://geometry.stanford.edu/lgl_2024/papers/hfghp-rfogm-06/hfghp-rfogm-06.pdf
* ACM DL: https://dl.acm.org/doi/10.1145/1141911.1141925
* Project page: https://www.dmg.tuwien.ac.at/geom/ig/pottmann/oldpub/2006/hfghp_fracture_06/hfghp_fracture_06.html
* **No source code was ever released.** The scanned fragment datasets were at
  `http://www.geometrie.tuwien.ac.at/ig/3dpuzzles.html` — that URL now 301-redirects
  and the data appears gone.

This is the single most relevant paper for our data: thick solid fragments, real
laser scans, non-symmetric objects, no training, no ground truth. It ran on a
**1.4 GHz machine with 512 MB RAM** in 2006, so its compute budget is far below an M2 Pro.

#### Full pipeline, stage by stage

Input is a set of point-set surfaces with oriented normals (they scanned the physical
fragments). The whole pipeline is run **2–3 times**, with matched fragments merged into
a "virtual fragment" between passes.

**Stage A — Integral invariants (§2).** Everything downstream is built on two
volumetric descriptors computed on a ball `B_r(p)`:

* Volume descriptor `V^r(p)` = fraction of `B_r(p)` inside the solid.
  `V^r(p) = 1/2 − (3/16)·H·r + O(r²)`, i.e. it encodes **mean curvature** `H`.
* Volume-distance descriptor `VD^r(p)` = weighted integral of squared distance to the
  surface over the ball. `VD^r(p) = 1 − (κ₁−κ₂)²·r²/28 + O(r³)`, i.e. it encodes the
  **principal-curvature difference**.
* For 3D curves (fracture edges), a deviation descriptor `D^r(p)` encoding curvature:
  `D^r(p) = 1 − κ²r²/16 + O(r³)`.

Computed at **N = 6 to 8 equidistant scales** `r_i = r_min + i·(r_max − r_min)/(N−1)`.
Critically, **`r_max = 0.1 × average fragment size`, and `r_min = r_max / 2`.**
Everything is relative to fragment size — no absolute units. This is the parameter
choice to copy.

**Stage B — Surface sharpness and roughness (§2.2).**

* *Sharpness* `s_vol(p) = [ (1/N)·Σᵢ (V^{r_i}(p) − 1/2)² ]^{1/2}`. Vanishes on planar
  neighbourhoods, high on the break curves between faces. Drives edge extraction.
* *Roughness*: they explicitly **rejected integral invariants** here — "at the small
  scales needed to compute the surface roughness, the integral invariants become
  expensive to compute and unstable." Instead they use a **local bending energy**
  over k-nearest neighbours:
  `e_k(p) = (1/k) Σᵢ ‖n_p − n_{q_i}‖² / ‖p − q_i‖²`,
  then average it over a ball neighbourhood: `e_{k,r}(p) = mean over N_r(p) of e_k(q)`.
  `k` and `r` are fitted **per object** by supervised learning on two manually selected
  point groups (one original-surface, one fracture-surface), picking `(k₀, r₀)` that
  minimises classification error. Output is a **binary label** `ρ(p) ∈ {0,1}`:
  1 = original (as-made) surface, 0 = fracture surface.

  → For us this is the "is this triangle a break face or the original terracotta
  surface" classifier, and it is the one supervised component in the whole paper.
  On our data the original surface is smooth/slipped and the break is rough, so a
  simple unsupervised 2-means or Otsu split on `e_{k,r}` should substitute fine.

**Stage C — Segmentation into faces (§3).**

1. *Multi-scale edge extraction*: a modified Pauly, Keiser & Gross (2003) point-cloud
   feature extractor. Two modifications: (a) replace Pauly's "surface variation" with
   the integral invariants `V^r`, `VD^r` plus the variance of `ρ` in the neighbourhood;
   (b) from the minimum spanning graph of edge points, **extract long closed cycles**
   — the cycles *are* the goal, since they bound the faces.
2. That yields an initial (over-)segmentation into faces `F_i`.
3. Build a weighted graph `G(F, E)`, nodes = faces, edges = adjacent face pairs, with
   two edge weights:
   * `w_r = |ρ(F_i) − ρ(F_j)|` (roughness contrast — high across an original/fracture boundary)
   * `w_s = mean of s_vol(p) over the shared border` (sharpness — high across a crease)
4. **First series of normalized cuts** (Shi & Malik 2000) using `w_r`, splitting into
   `P₁` = original faces, `P₂` = fracture faces. Stop when the roughness variance
   `σ²_ρ(P_k) < 0.3` for both parts.
5. **Second series of normalized cuts** on `P₂` using `w_s`, iteratively cutting along
   the highest-total-`w_s` cut to merge over-segmented fracture patches into meaningful
   faces. **Stop when the graph-cut threshold `w_s` falls below 0.1.**

**Stage D — Feature clusters (§4).** Not point features — *patch* features, and they
deliberately **overlap**.

* Take a descriptor `g` with range `[0, b]`, quantise into **32 equal intervals**.
  For each level pair `l_i < l_j`, take `S_ij = {p : g(p) ∈ [l_i, l_j]}`, and split it
  into connected components by depth-first search over K-nearest neighbours → clusters `C^k_ij`.
* Discard redundant clusters where `min g > l_{i+1}` or `max g < l_{j−1}`.
* Descriptors used: for **fracture surfaces**, four — `g₁ = V^{r_max}`, `g₂ = VD^{r_max}`,
  `g₃ = V^{r_min}`, `g₄ = VD^{r_min}`. For **fracture edges**, two — `D^{r_max}`, `D^{r_min}`.
* **Cluster topology**: `C` and `D` are *neighbours* if they share any point; `C_{g_j}`
  is a *parent* of `C_{g_i}` if its kernel radius is larger. This parent/child structure
  is what later verifies correspondences.
* Each cluster is stored compactly as `C = {b(C), n_j(C), p^±_k(C), R(C), µ_i(C)}`:
  barycentre `b`; PCA principal directions `n₁,n₂,n₃` (right-handed); auxiliary points
  `p^±_k = b ± l_d·n_k` at **`l_d = r_max/2`**, used for rough registration;
  a representative point set `R(C) = B_{r_max}(b) ∩ Φ` used for fine registration; and
  four signatures — the two level values `l_i, l_j`, the **size signature**
  `sig_S = (λ₁+λ₂+λ₃)^{1/2}`, and the **anisotropy signature** `sig_A = |λ₂/λ₃|^{1/2}`
  (for edges, `|λ₁/λ₃|^{1/2}`). Edge clusters that lie on an original/fracture break curve
  additionally carry an **angle signature** `sig_a = ∠(n₁, n_o)` against the original-surface normal.

  They explicitly note this low-dimensional representation "suits better for fracture
  surfaces (with rather similar local neighborhoods) than other, richer descriptors
  such as shape contexts or spin images."

**Stage E — Pairwise matching (§5).** Between two *faces* `S` and `T` (not whole fragments).

1. *Initial correspondences*: `C_{g_i}` ↔ `D_{g_i}` if same descriptor and same
   descriptor signatures. Edge features additionally need `|sig_a(C) + sig_a(D)| < ε_θ`
   (note the **sum**, not difference — complementary surfaces have opposite angles).
2. *Shape pruning*: with `SD(p) = (sig_S(C) − sig_S(D)) / (sig_S(C) + sig_S(D))` and
   `AD(p)` likewise for anisotropy, keep only `SD ≤ ε_s` and `AD ≤ ε_a`.
   **Both thresholds = 0.1.**
3. *Topological pruning*, two steps:
   (a) A correspondence at a *large* radius is kept only if a child (smaller-radius,
   overlapping) correspondence also exists — "correspondences whose features are not
   verified by their children are likely to be incorrect."
   (b) Among overlapping same-descriptor pairs that are geometrically consistent, drop
   the one with the larger average size signature.
4. *Geometric consistency* on pairs `p₁=(C¹,D¹)`, `p₂=(C²,D²)`: displacement vectors
   `b(C¹)−b(C²)` and `b(D¹)−b(D²)` similar in length, and all pairwise angles
   `∠(n_k(C¹), n_l(C²))` vs `∠(n_k(D¹), n_l(D²))` within **`ε_θ = 0.2` rad**.
5. *Registration consistency*: Horn's quaternion method on the 8 points
   `(b(C), p^±_k(C))` gives an initial transform; refined by local registration on the
   full `R(C)` point sets. Keep only if it converges with mean deviation below `ε_dev`
   — and `ε_dev` is **not** a user constant, it is derived from the estimated surface
   noise level à la Fleishman et al. 2005.
6. **Forward search** (Atkinson et al. 2004 — this is their key robustness trick, in
   place of RANSAC). Order correspondences by increasing `AD·SD`. Seed `E¹` with a
   consistent pair whose barycentres satisfy `|b(C^i) − b(C^j)| > 2·l_d`. At step `m`,
   fit rigid `α^m` from `E^m`, compute residual `‖α^m(b(C^l)) − b(D^l)‖` for every
   candidate, and take the `m+1` smallest into `E^{m+1}`. **The critical index `m*` is
   where the largest residual first exceeds `2·ε_dev`** — that is where outliers start
   entering, and `E^{m*}` is emitted as one match hypothesis. Mark those pairs, repeat
   until exhausted. So one face pair can emit *several* competing hypotheses; they are
   all kept and resolved globally later.
7. *Match quality*: `w_d` = mean deviation over overlapping fracture-face region,
   `w_e` = same for fracture edges, `w_f = ∫_{S∩T} s_vol(p) dp` = total sharpness in the
   overlap (a proxy for "how much distinctive geometry actually agrees"). Overlap
   regions come from a bidirectional closest-point search (Pauly et al. 2005). Final score:

   **`w = log(w_f) − log(w_e) − log(w_d)`**

   High feature content, low deviation. If the matching faces carry original/fracture
   break curves, a *surface consistency* constraint additionally requires those curves
   to be within a distance threshold — this is the "the outer skin must line up" check.

**Stage F — Global multi-piece matching (§6.1).** Graph `G(F, E)`: nodes = fragments,
edges = candidate matches with weight `w(e)` and transform `T(e)`. Multi-piece matching
is NP-hard (Huber 2002). Their greedy scheme:

* A second graph `M(Γ, L)` over *sub-graphs* (already-merged groups). Edge weight
  between groups is `w(G₁,G₂) = Σ w(e)` over all connecting edges.
* **They do not merge on the single best edge** (Huber's strategy) — Fig. 9 shows that
  fails. Instead they use **cycle consistency**: relative transforms around a loop in
  `M` must compose to the identity. Concretely they minimise
  `Q = Σ_e w(e)·( ‖α_i − α_j∘T(e)‖²_{F_i} + ‖α_j − α_i∘T(e)‖²_{F_j} )`
  over the fragment motions `A = {α₁ … α_n}` in an affine relaxation (Hofer et al. 2004
  metric), then **project back onto the rigid-motion manifold** (Krishnan et al. 2005).
* A **second forward search**, now over graph edges rather than feature correspondences:
  order unclassified edges by residual `‖α_i − T(e)∘α_j‖²`, grow the set, stop at `m*`.
  The resulting edge subset `E_s^{k,l}` is one consistent bundle.
* Bundles are ranked **by size first** (prefer many mutually consistent edges over one
  strong edge), then by total weight.
* **Penetration test**: collision detection (Lin & Manocha 2004) between all fragment
  pairs in the merged group. **Any match with penetration exceeding `r_min` is discarded.**
  This is the workhorse rejection test — "incorrect matches lead to heavy penetration
  effects and thus can easily be detected."

**Stage G — Multi-piece constrained local registration (§6.2).** A non-penetrating
simultaneous ICP. Fix `Σ₀`; for every other fragment use the linearised (helical) motion
`v_{i0}(x) = c̄_i + c_i × x`. For sampled points `x_i` with closest target point `y_i`,
unit residual `r_i` and distance `d_i`, minimise

`F(c₁,c̄₁,…,c_n,c̄_n) = Σ_i [ d_i + r_iᵀ·v_{jk}(x_i) ]²`

subject to the **non-penetration linear constraints** `n_iᵀ·(x_i − y_i + v_{jk}(x_i)) ≥ 0`
(all points of one fragment stay on the same side of its neighbour's surface). This is a
QP with linear constraints, solved with an **active set method**. Because the linearised
motion is affine not Euclidean, each step is applied as the underlying helical motion.
**Converges in 5–10 iterations.**

**Stage H — Merge.** Delete matched fracture-face points via bidirectional closest-point
search, hole-fill (Amenta & Kil 2004), producing a single closed "virtual fragment" that
re-enters the pipeline on the next pass.

#### Reported performance

| Example | Material | Fragments | Points | Result |
|---|---|---|---|---|
| Gargoyle | stone | 30 | 3.54 M | 28/30 assembled |
| Cake | mortar | 11 | 1.45 M | full |
| Brick | stone | 6 | 1.49 M | full |
| Venus | clay | 7 | 1.84 M | only 6 largest |
| Sculpture | clay | 15 | 1.66 M | full |
| Head (thin shell) | clay | 12 | 1.15 M | 10/12 |
| Forma Urbis Romae | marble | 20 | 9.45 M | eroded faces, correct matches found |

Timings on a **1.4 GHz / 512 MB** machine:
* ~**1 minute per fragment at 400k points** for descriptor setup (integral invariants at 8 scales).
* Segmentation + feature selection + pruning: **~4 s per fragment**, linear in point count.
* Brick example (6 fragments): potential matches built in **2 s**; multi-piece matching
  including the dominating constrained registration **5 s**; two full passes **15 s total**.

So the expensive part is the integral-invariant precomputation, and it is embarrassingly
parallel over points. On an M2 Pro this is minutes, not hours.

#### Threshold summary (copy these)

| Symbol | Value | Meaning |
|---|---|---|
| `r_max` | 0.1 × mean fragment size | largest integral-invariant kernel radius |
| `r_min` | `r_max / 2` | smallest kernel radius |
| `N` scales | 6–8 | equidistant radii between the two |
| descriptor bins | 32 | level-set quantisation |
| `l_d` | `r_max / 2` | offset of auxiliary points from cluster barycentre |
| `ε_θ` | 0.2 | angle-signature and geometric-consistency tolerance |
| `ε_s`, `ε_a` | 0.1 | size / anisotropy deviation |
| `ε_dev` | noise-adaptive | registration mean-deviation tolerance |
| roughness cut stop | `σ²_ρ < 0.3` | original vs fracture split |
| sharpness cut stop | `w_s < 0.1` | fracture-face merging |
| penetration reject | `> r_min` | global-match validity |

#### Honest caveats for our use

* Their fragments came from **deliberately fracturing modern objects** (dropping, hammer
  and chisel) and scanning immediately. Our terracotta is archaeological: eroded fracture
  faces, possible missing material at the joins. The FUR example is their only heavily
  eroded case and there they only verified known matches rather than discovering new ones.
* The roughness classifier is **supervised per object** (two hand-picked point groups).
* The thin-shell "head" example is the closest analogue to a plate or a thin pot wall,
  and it is where they lost fragments (10/12) — small fracture surfaces are the hard case.

### 1.2 Papaioannou, Karabassi, Theoharis (2001 / 2003 / 2017)

* Papaioannou, Karabassi, Theoharis, "Virtual archaeologist: Assembling the past",
  IEEE CG&A 21(2):53–59, 2001.
* Papaioannou & Karabassi, "On the automatic assemblage of arbitrary broken solid
  artefacts", Image and Vision Computing 21:401–412, 2003.
* Papaioannou, Schreck, Andreadis et al., "From Reassembly to Object Completion: A
  Complete Systems Pipeline", ACM JOCCH 10(2), 2017.
  https://dl.acm.org/doi/10.1145/3009905

**Key idea.** Region-grow the fracture faces, project each face along its average normal
into a **depth map (z-buffer)**, and score a candidate pose by the *complementarity* of
the two depth maps — how well the peaks of one fill the valleys of the other. Pose error
is minimised by **simulated annealing over a 7-D pose space** relative to a separating
plane.

**Input.** Solid fragment meshes with normals. **Assumes fracture faces are near-planar
and match each other completely** — Huang explicitly cites this as the limitation their
method removes.

**Code.** None public. The 2017 pipeline paper describes a full system (from the
PRESIOUS EU project) but no repository.

**Relevance to us: moderate.** The z-buffer complementarity score is *cheap*, trivially
implemented with numpy, and is an excellent **verification/scoring function** even if not
used as the search method. Worth stealing as a rejection test alongside penetration.

### 1.3 Winkelbach & Wahl — RANSAM family

* Winkelbach, Rilk, Schönfelder, Wahl, "Fast Random Sample Matching of 3D Fragments",
  DAGM 2004. https://link.springer.com/chapter/10.1007/978-3-540-28649-3_16
* Winkelbach & Wahl, "Pairwise Matching of 3D Fragments Using Cluster Trees",
  IJCV 78(1):1–13, 2008. https://dblp.org/rec/journals/ijcv/WinkelbachW08.html

**Key idea.** A RANSAC variant ("RANSAM") over **point pairs with normals**. Picking two
oriented points on each surface fixes almost all 6 DoF, so the search space collapses
dramatically. The 2008 version decomposes each point set into a **binary cluster tree**
and descends both trees simultaneously depth-first, which prunes the pair sampling.

**Input.** Any surface data with normals — range images, meshes, point clouds. No
segmentation needed, no training. Explicitly "robust, very time and memory efficient,
easy to implement".

**Code.** None public from the authors.

**Relevance to us: high, conceptually.** This is essentially what
Open3D's `registration_ransac_based_on_feature_matching` does when you feed it FPFH
plus normals, and what 4PCS/Super4PCS do. The "two oriented points fix the pose" insight
is the reason a FPFH+RANSAC baseline is viable at all on fracture surfaces.

### 1.4 Altantsetseg, Matsuyama, Konno (2014) — FFT curve matching

"Pairwise matching of 3D fragments using fast Fourier transform", The Visual Computer 30:1–11.
https://link.springer.com/article/10.1007/s00371-014-0959-9

**Key idea.** Extract feature points on unorganised point clouds by curvature, cluster
them, and build a descriptor consisting of the cluster **plus curves along its principal
directions**. Approximate each curve by a **Fourier series**, then compare clusters by
FFT coefficients and total curve energy.

**Input.** Unorganised point clouds. No training. **Code: none public.**

**Relevance: moderate.** The idea of reducing a fracture patch to a small set of 1-D
profile curves and comparing them in the frequency domain is cheap and rotation-friendly.
Note this is the *Tokyo Denki / Iwate* line of work, not Kyushu.

### 1.5 ElNaghy & Dorst — morphological scale space (the best fit for *thick* eroded fragments)

* "Complementarity-Preserving Fracture Morphology for Archaeological Fragments",
  ISMM 2019. https://arxiv.org/abs/1901.05726
* "Pairwise Alignment of Archaeological Fragments Through Morphological Characterization
  of Fracture Surfaces", IJCV 130:2924–2949, 2022.
  https://link.springer.com/article/10.1007/s11263-022-01635-3

**Key idea.** Use **mathematical-morphology scale spaces** (simultaneous closing and
opening via distance transforms, exploiting the Lipschitz property of fracture geometry)
to hierarchically simplify a fracture surface while **preserving complementarity** — i.e.
the simplification of a fracture surface and of its counterpart remain mutual negatives.
Explicitly designed to be robust to **abrasion and erosion**, where the two faces no
longer fit exactly.

**Input.** Voxelised / distance-transformed solid fragments. Came out of the **GRAVITATE**
EU project (grant 665155), working on real thick Cypriot terracotta and marble artefacts.

**Code.** Not found public. Searched author profiles, Springer supplementary, GitHub — nothing.

**Relevance to us: high on paper.** This is the only line of work explicitly built for
*eroded, thick, non-exactly-fitting* archaeological fracture surfaces, which is exactly
our data. But with no code and a voxel/distance-transform formulation, reimplementing is
a substantial project. Worth reading for the erosion-tolerance idea (match at a coarse
morphological scale first, then refine) even if not implemented.

### 1.6 Others worth knowing

| Work | Year | Key idea | Code |
|---|---|---|---|
| Kong & Kimia, "On solving 2D and 3D puzzles using curve matching", CVPR | 2001 | curve matching on break curves | no |
| Willis & Cooper, axially symmetric sherds | 2004 | Bayesian axis+profile estimation; **our S-f-S++ failure mode** | no |
| Huber, PhD thesis (CMU) | 2002 | multi-piece = best spanning tree in match graph; proves NP-hard | no |
| Koller & Levoy, Forma Urbis Romae | 2005 | heavily eroded marble; matches **incisions on top surfaces**, not fracture geometry | data at http://formaurbis.stanford.edu/ |
| Son, Lee, Lim, Lee, "Reassembly of fractured objects using surface signature", Vis. Comput. 34:1371–1381 | 2018 | "surface signature" = convex/concave labelling of the fracture face; feature curves from signature boundaries; matched via spin images | no |
| Zhao et al., "Rigid blocks matching based on contour curves and feature regions", IET CV | 2018 | contour + region hybrid | no |
| Alzaid et al., "Reassembly of fractured object using fragment topology", ICPRS | 2021 | topology graph of faces | no — https://eprints.whiterose.ac.uk/id/eprint/181957/ |
| Lu, Huang et al., **Survey**, Computer Graphics Forum | 2025 | the field's only survey; taxonomy single-piece / multi-piece / template-based | https://onlinelibrary.wiley.com/doi/10.1111/cgf.70081 and https://arxiv.org/abs/2410.14770 |

**The survey is the single best entry point** and explicitly flags the gap we are hitting:
"real fragments often exhibit erosion, weathering, or missing pieces — conditions that are
difficult to fully capture in synthetic training data", and it calls for more open-source
release because so little classical code exists.

---

## 2. Deep-learning methods and datasets

### 2.1 Datasets

* **Breaking Bad** (Sellán, Chen, Wu, Garg, Jacobson, NeurIPS 2022 D&B).
  1M+ fractured objects from 10k base models, fractured by a physically based
  fracture-mode simulation. Everything downstream trains on this.
  - Paper: https://arxiv.org/abs/2210.11463
  - Site: https://breaking-bad-dataset.github.io/
  - Fracture simulator code: https://github.com/sgsellan/fracture-modes (115★)
  - Baselines: https://github.com/Wuziyi616/multi_part_assembly (82★)
  - **Caveat: entirely synthetic, clean, watertight, no erosion, no missing material.**

* **RePAIR** (Tsesmelis, Palmieri, Khoroshiltseva et al., NeurIPS 2024 D&B).
  Real Pompeii fresco fragments, broken by the AD 79 eruption and a WWII bombing.
  Multi-modal: high-res images + 3D scans + archaeologist metadata. Eroded, missing pieces.
  - Paper: https://arxiv.org/abs/2410.24010
  - Site: https://repairproject.github.io/RePAIR_dataset/
  - **These are fresco fragments — thin, plate-like, and the pictorial surface carries
    most of the signal. Less like our thick armour relief than the name suggests.**

* **Fractura** (with GARF, 2025). Real-world fracture types across **ceramics**, bones,
  eggshells, lithics. Closest public analogue to our data.
  https://ai4ce.github.io/GARF/

* **PRESIOUS / Presious** — EU project datasets of eroded stone artefacts, tied to the
  Papaioannou 2017 pipeline.

### 2.2 Methods

| Method | Year/venue | Code | GPU/training | CPU inference? |
|---|---|---|---|---|
| **Jigsaw** (Lu, Sun, Huang) | NeurIPS 2023 | https://github.com/Jiaxin-Lu/Jigsaw (50★, MIT, last push 2024-03) | trained on Breaking Bad; PyTorch | possible in principle, undocumented, unvalidated |
| **PuzzleFusion++** (Wang et al.) | ICLR 2025 | https://github.com/eric-zqwang/puzzlefusion-plusplus (73★) | diffusion denoiser + transformer verifier, Breaking Bad | not documented |
| **DiffAssemble** (Scarpellini et al., IIT-PAVIS) | CVPR 2024 | https://github.com/IIT-PAVIS/DiffAssemble (97★, no license) | graph-diffusion, 2D+3D | not documented |
| **PHFormer** (Cui et al.) | AAAI 2024 | https://github.com/521piglet/PHFormer (8★) | proxy-level hybrid transformer | no |
| **FragmentDiff** (Xu et al.) | SIGGRAPH Asia 2024 | https://github.com/xuqunce/FragDiff (3★) | transformer diffusion over poses | no |
| **GARF** (Li et al., ai4ce/NYU) | ICCV 2025 | https://github.com/ai4ce/GARF (101★, GPL-3.0) | see below | **blocked by flash-attn** |
| **Jigsaw++** | 2024 | — | shape-prior completion | no |
| **SE(3)-equivariant assembly**, BITR, PMTR, GeoAssemble, SARe | 2023–2026 | mostly none/minimal | all Breaking Bad trained | no |

**GARF in detail** (the only one with a serious real-scan story):

* Backbone: **Point Transformer V3** encoder, **12.7 M** trainable params for
  fracture-aware pretraining; **43.5 M** for the flow-matching denoiser.
* Trained on **1.9 M fragments** (three Breaking Bad subsets, 14× prior work) on
  **4× NVIDIA H100**, ~2 days pretraining + ~3 days flow matching.
* Inference: **5000 points per fragment** (Poisson-disk sampled), one-step preassembly
  then **20 flow-matching refinement steps**, **190.77 ms per assembly** on GPU.
* Does **not** need the fragment count known; trained on 2–20 fragments, generalises past 20.
* Pretrained checkpoints **are** released (GARF-mini-E-FM, GARF-mini-E-Diff, GARF-EAO-FM).
* Reported 82.87 % lower rotation error and 25.15 % higher part accuracy than prior SOTA;
  handles ceramics with up to three missing fragments.
* **Compares only against learning baselines — no classical geometric method is benchmarked
  anywhere in the paper.** That is a real gap in the literature and means we have no
  published evidence that GARF beats a well-tuned Huang-style pipeline on real terracotta.

**Feasibility of GARF on our Mac — honest assessment.**
The *arithmetic* is not the blocker: 43.5 M params over 5000 points × N fragments for 21
steps is seconds-to-a-minute of CPU work, well within 16 GB. The blockers are engineering:

1. `flash-attn` requires **CUDA ≥ 12.0** and has no CPU path. It would have to be replaced
   by `torch.nn.functional.scaled_dot_product_attention`. Mechanical, but it is code surgery
   in someone else's repo.
2. `pytorch3d` "may need GPU available when installing"; CPU-only builds on macOS arm64 are
   painful and usually require building from source.
3. **Point Transformer V3 depends on serialised sparse-voxel attention** and typically pulls
   in `spconv` / custom CUDA kernels. This is the deeper problem.
4. Input pipeline is **HDF5 in Breaking Bad layout**, produced by their fork of Breaking Good.
   Feeding raw scanned PLYs means writing a converter.
5. Scale/units: the model is trained on unit-normalised synthetic fractures. Our
   350k–670k-vertex Geomagic meshes must be decimated to 5000 Poisson-disk points and
   normalised the same way, and colour is ignored.

There is a **hosted browser demo at https://garf-demo.pages.dev** — worth trying with two
or three of our fragments as a **zero-cost sanity check on whether learned methods see any
signal in our data at all**, before investing in a local port. (Uploading museum scans to a
third-party demo is a data-governance decision for the museum, not a technical one.)

**Bottom line on deep learning:** every method is trained on synthetic Breaking Bad
fractures; none is CPU-documented; none ships a CPU inference path; and none benchmarks
against classical geometry. GARF is the only one with credible real-scan evidence, and it
is the one with the heaviest CUDA dependency chain. **Not a viable primary approach under
our constraints.** Reasonable as a later "second opinion" if a GPU box ever appears.

### 2.3 AAFR — "Reassembling Broken Objects using Breaking Curves"

Alagrami, Palmieri, Aslan, Pelillo, Vascon (Ca' Foscari / RePAIR), ICPR 2024.

* Paper: https://arxiv.org/abs/2306.02782
* Code: **https://github.com/RePAIRProject/AAFR** (12★, CC0-1.0, last push 2025-03-03)

**Pipeline** (three stages, all classical, no network):
1. Build **dual graphs** over the whole point cloud and over its borders; compute a
   *corner penalty* per point to detect points on the **breaking curves** (3D edges).
2. **Segment** the point cloud along the detected breaking curves into regions.
3. **Register every viable region pair** and select the best alignment.

**Practicalities.** Python ≥ 3.9, Open3D, NumPy pinned to `1.26.4`, and it *recommends
TEASER++* for robust registration. Input `.ply` or `.obj`. Run with
`python assemble_fragments.py --cfg assemble_cfg`. **No GPU required — this is the one
CPU-runnable public codebase in the whole scan.**

**The catch, stated in their own README: it handles two fragments, and multi-fragment
"will be extended".** Runtime is undocumented. 12 stars means essentially no external
users. Treat it as **a reference implementation of breaking-curve detection to read and
borrow from**, not as a tool to depend on.

Note we already have this paper locally at
`/Users/vaceslaveliseev/@dev/ceramic-reassembling/papers/Reassembling_Broken_Objects_using_Breaking_Curves.pdf`.

---

## 3. Other open-source tooling surveyed

**GitHub search results, sorted by stars** (queries: "fracture reassembly", "fragment
reassembly", "fractured object assembly", "sherd reassembly", "3d puzzle point cloud"):

| Stars | Repo | Note |
|---|---|---|
| 115 | sgsellan/fracture-modes | fracture *simulation*, not reassembly |
| 101 | ai4ce/GARF | CUDA-bound |
| 97 | IIT-PAVIS/DiffAssemble | no license file |
| 82 | Wuziyi616/multi_part_assembly | Breaking Bad learning baselines |
| 73 | eric-zqwang/puzzlefusion-plusplus | learning |
| 50 | Jiaxin-Lu/Jigsaw | learning |
| 12 | RePAIRProject/AAFR | **the only CPU-classical option** |
| 12 | SeongJong-Yoo/structure-from-sherds | C++, axially symmetric pots — **already tried, failed on our data** |
| 8 | 521piglet/PHFormer | learning |
| 6 | alexandrumeterez/3d-fracture-reassembly | student project, unmaintained |
| 3 | xuqunce/FragDiff | learning |

**Nothing in the "fragment reassembly" keyword space above 20 stars is both classical and
CPU-runnable except AAFR (12★).** The `reassembler` (35★) hit is network packet
reassembly, unrelated.

The **RePAIRProject org** (https://github.com/RePAIRProject) is the richest single source
of adjacent code — 23 repos. Relevant ones: `AAFR` (12★), `3D-baselines` (2★, all five
baselines are neural: Global/PointNet, LSTM, DGL, SE(3)-Equiv, DiffAssemble — **no
classical CPU baseline**), `segment_3Dfragment` (2★), `RL_puzzle_solver` (5★, 2D),
`fragment-restoration` (4★, 2D semantic segmentation of fresco motifs).

**CGAL / PCL.** Neither has a fragment-reassembly module. What they do offer that Open3D
lacks: CGAL has robust **shape detection (efficient RANSAC for planes/cylinders)**,
**mesh segmentation via the shape diameter function**, and exact predicates for
penetration tests; PCL has **Super4PCS/Super4PCS-style global alignment**,
`SampleConsensusPrerejective`, and `NormalDistributionsTransform`. Both are C++ with
awkward Python bindings on macOS arm64 (`pycgal`/`CGAL-bindings` are stale; `python-pcl`
is effectively dead). **Not recommended — Open3D covers the needed primitives natively.**

---

## 4. Robust registration building blocks — verified on this machine

Verified against the project venv at `/Users/vaceslaveliseev/@dev/ceramic-reassembling/.venv`:
**Python 3.12.9, Open3D 0.19.0, NumPy 2.5.2, trimesh present.**

### 4.1 pip availability on macOS arm64 / Python 3.12

| Package | On PyPI | arm64 wheel for cp312 | Verdict |
|---|---|---|---|
| **open3d 0.19.0** | yes | **yes** — `open3d-0.19.0-cp312-cp312-macosx_10_15_universal2.whl` | **installed and working** |
| trimesh | yes | pure-python wheel | fine |
| scikit-learn, scipy, numpy | yes | yes | fine |
| **teaserpp-python** | **NO — 404 on PyPI** | n/a | **must build from source** (CMake + Eigen3 + Boost + pybind11). Open issues on the repo: "[BUG] tests not passing on macOS", "PyPi package available and/or planned?" (still open), a closed "Add instruction for compiling on macOS". Repo itself is healthy (2343★, MIT, active Dec 2025). **Treat as a stretch dependency, not a baseline.** https://github.com/MIT-SPARK/TEASER-plusplus |
| **probreg** | yes, **sdist only, zero wheels** | none | needs a C++ build; last release 0.3.8. CPD/GMMReg/FilterReg. **Skip unless CPD is specifically wanted.** 990★, MIT. https://github.com/neka-nat/probreg |
| pycpd | yes, pure python | n/a | pure-numpy CPD, trivially installable, but O(N·M) per iteration — only for tiny point sets |

### 4.2 Open3D 0.19 API surface — all confirmed present

```
registration_ransac_based_on_feature_matching   OK
registration_fgr_based_on_feature_matching      OK
registration_icp                                OK
registration_generalized_icp                    OK
compute_fpfh_feature                            OK
TukeyLoss / GMLoss / HuberLoss / CauchyLoss     OK   (robust kernels for ICP)
TransformationEstimationPointToPlane            OK
TransformationEstimationForGeneralizedICP       OK
global_optimization                             OK   (pose-graph / multiway registration)
```

`global_optimization` deserves emphasis: it is a **direct, built-in substitute for
Huang §6.1's global cycle-consistency optimisation.** Build a `PoseGraph` with fragments
as nodes, candidate matches as edges (with `uncertain=True` for loop closures and an
information matrix from the match quality), run
`GlobalOptimizationLevenbergMarquardt` with `GlobalOptimizationLineProcessModel`, and the
line process **automatically prunes inconsistent edges** — exactly the outlier-edge
rejection Huang implemented by hand with a forward search. This is the highest-leverage
single API in the whole scan.

### 4.3 Measured timings on this M2 Pro

Synthetic fracture-like patches (noisy displaced sinusoidal surface), voxel size 0.02 on a
2-unit-wide patch, normals at 3× voxel radius, FPFH at 5× voxel radius, RANSAC with
`n=3`, 100k iterations / 0.999 confidence:

| Input pts | After voxel downsample | normals | FPFH | RANSAC | FGR | ICP refine |
|---|---|---|---|---|---|---|
| 20 000 | 15 516 | 0.02 s | 0.15 s | **0.91 s** | 1.35 s | 0.01 s |
| 50 000 | 28 522 | 0.04 s | 0.28 s | **4.18 s** | 6.97 s | 0.02 s |
| 100 000 | 38 860 | 0.07 s | 0.55 s | **11.55 s** | 20.17 s | 0.06 s |

All recovered the correct transform (fitness 0.97–1.00). Script:
`.../scratchpad/bench.py`.

**Reading of these numbers.**
* FPFH and ICP are **free** at our scale. The cost is entirely in the global search.
* RANSAC cost grows roughly linearly in retained points here; **keep each fracture face
  under ~20–30k points** (voxel downsample) and a pair costs ~1–4 s.
* **FGR is consistently slower than RANSAC** at these sizes in Open3D 0.19 and was the
  only method to drop below fitness 1.0. Prefer RANSAC; keep FGR as a fallback.
* Budget: 4 fragments × ~3 fracture faces each ⇒ 6 fragment pairs × ~9 face pairs ≈ 54
  registrations ≈ **1–4 minutes total**, comfortably inside the "minutes for 4 fragments"
  requirement. Dozens of fragments scales quadratically — 20 fragments ⇒ 190 pairs ⇒
  ~1700 face-pair registrations ⇒ **~30–90 min**, which is where cheap pre-filtering
  (area, bounding box, curvature histogram) stops being optional.

**A warning about FPFH on fracture surfaces.** FPFH is a histogram of normal-angle
relations in a neighbourhood. Fracture faces are *rough but statistically self-similar* —
patches from different parts of the same break can look alike, and the true match is a
**negative/complementary** surface, not a copy. FPFH+RANSAC as used for scan alignment
assumes the two surfaces are *the same surface seen twice*. For fracture matching you
must either (a) mirror/negate one side's normals and match, or (b) rely on the fact that
the *geometry* (not the normals) of two complementary faces coincides where they touch,
so point-to-point FPFH still fires if normals are flipped consistently. **This is the
single biggest technical risk in a FPFH-based approach and needs an early experiment.**

---

## 5. Recommended architecture for our constraints

A Huang-2006-shaped pipeline with modern CPU primitives. Every stage below has a concrete
Open3D/scipy implementation and a fallback.

**S1 — Preprocess.** Load PLY with Open3D (keeps vertex colours). Compute vertex normals.
Estimate a per-fragment scale = mean bounding-box extent; **set `r_max = 0.1 × mean
fragment size`, `r_min = r_max/2`** exactly as Huang. Voxel-downsample to a working
resolution of about `r_min/4`.

**S2 — Fracture/original classification.** Replace Huang's supervised roughness with the
same **local bending energy** `e_k(p) = (1/k)Σ‖n_p − n_q‖²/‖p−q‖²` (k≈20 via
`scipy.spatial.cKDTree`), smoothed over a ball of radius `r_min`, then split **unsupervised**
by Otsu or 2-means. Terracotta original surfaces are smooth and often slipped; break faces
are grainy. **Colour is a free extra signal here** — the fracture interior of terracotta is
usually a different hue from the weathered exterior, and we have coloured meshes. Huang had
no colour; we do. Use it.

**S3 — Face segmentation.** Two options, cheapest first:
* `TriangleMesh.cluster_connected_triangles` after cutting at high-`e_k,r` / high-curvature
  edges, or `PointCloud.cluster_dbscan` on the fracture-labelled points.
* Faithful route: normalized cuts on the face-adjacency graph with Huang's `w_r`/`w_s`
  weights (`scipy.sparse.linalg.eigsh` on the normalised Laplacian), stopping at
  `σ²_ρ < 0.3` then `w_s < 0.1`.

**S4 — Pairwise face matching.** FPFH + RANSAC per fracture-face pair, downsampled to
≤ 20k points per face (measured ~1–4 s/pair). **Pre-filter face pairs by area ratio and
by the histogram of `e_k,r`** to avoid the full quadratic blowup. Refine with
point-to-plane ICP under a `TukeyLoss` robust kernel.

**S5 — Verification.** Three independent rejection tests, all cheap:
1. **Penetration** — Huang's decisive test. Reject if interpenetration exceeds `r_min`.
   `trimesh` + `fcl` or a signed-distance query against the watertight meshes (our meshes
   *are* watertight, which makes this easy — that is a genuine advantage of this dataset).
2. **Complementarity** — Papaioannou's z-buffer score, ~20 lines of numpy: project both
   faces onto the mean-normal plane and check that the depth maps sum to a constant.
3. **Original-surface continuity** — Huang's surface consistency: the break curves where
   fracture meets original surface must line up. On armour relief this is strong signal.

**S6 — Global assembly.** `o3d.pipelines.registration.global_optimization` with a pose
graph, `GlobalOptimizationLevenbergMarquardt`, and `GlobalOptimizationLineProcessModel`
to auto-prune inconsistent edges. Score edges with Huang's
`w = log(w_f) − log(w_e) − log(w_d)`. Greedy merge with penetration re-testing after
each merge, and re-run the pipeline on merged virtual fragments (Huang ran 2–3 passes).

**S7 — Final polish.** Multiway ICP over all fragments simultaneously. Huang's
non-penetration QP is the principled version; a practical substitute is ICP with a
penetration penalty term, or simply stopping ICP short of penetration.

**Order of experiments, riskiest first:**
1. Does the fracture/original split work on one fragment? (S2) — 1 hour.
2. Does FPFH+RANSAC find the true match between two known-adjacent fracture faces,
   with correct normal handling? (S4) — this is the make-or-break test.
3. Does penetration+complementarity reject the false matches? (S5)
4. Only then build the global stage.

---

## 6. Sources

Classical:
- https://geometry.stanford.edu/lgl_2024/papers/hfghp-rfogm-06/hfghp-rfogm-06.pdf
- https://dl.acm.org/doi/10.1145/1141911.1141925
- https://www.dmg.tuwien.ac.at/geom/ig/pottmann/oldpub/2006/hfghp_fracture_06/hfghp_fracture_06.html
- https://link.springer.com/chapter/10.1007/978-3-540-28649-3_16 (Winkelbach RANSAM)
- https://dblp.org/rec/journals/ijcv/WinkelbachW08.html (cluster trees)
- https://link.springer.com/article/10.1007/s00371-014-0959-9 (Altantsetseg FFT)
- https://arxiv.org/abs/1901.05726 (ElNaghy & Dorst, morphological scale space)
- https://link.springer.com/article/10.1007/s11263-022-01635-3 (ElNaghy & Dorst IJCV 2022)
- https://link.springer.com/article/10.1007/s00371-017-1419-0 (Son, surface signature)
- https://dl.acm.org/doi/10.1145/3009905 (Papaioannou 2017 pipeline)
- https://onlinelibrary.wiley.com/doi/10.1111/cgf.70081 and https://arxiv.org/abs/2410.14770 (survey)

Learning + datasets:
- https://arxiv.org/abs/2210.11463 , https://breaking-bad-dataset.github.io/
- https://github.com/sgsellan/fracture-modes
- https://github.com/Wuziyi616/multi_part_assembly
- https://github.com/Jiaxin-Lu/Jigsaw
- https://github.com/eric-zqwang/puzzlefusion-plusplus , https://arxiv.org/abs/2406.00259
- https://github.com/IIT-PAVIS/DiffAssemble
- https://github.com/521piglet/PHFormer
- https://github.com/xuqunce/FragDiff
- https://github.com/ai4ce/GARF , https://arxiv.org/abs/2504.05400 , https://garf-demo.pages.dev
- https://arxiv.org/abs/2410.24010 , https://repairproject.github.io/RePAIR_dataset/
- https://github.com/RePAIRProject/AAFR , https://arxiv.org/abs/2306.02782
- https://github.com/RePAIRProject (org, 23 repos)
- https://github.com/SeongJong-Yoo/structure-from-sherds , https://arxiv.org/abs/2502.13986

Tooling:
- https://www.open3d.org/docs/release/arm.html
- https://github.com/MIT-SPARK/TEASER-plusplus , https://teaser.readthedocs.io/en/latest/installation.html
- https://github.com/neka-nat/probreg
