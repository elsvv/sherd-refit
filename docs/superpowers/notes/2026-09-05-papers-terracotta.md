# Terracotta Warrior Papers: Technical Analysis for a Fracture-Surface Reassembly Tool

Analysed against our target: CPU-only Python (numpy/scipy/open3d/trimesh/sklearn), Apple M2 Pro, 16 GB, no GPU, no training data, no ground truth, no GUI. Input is coloured PLY triangle meshes, 350k–670k vertices, ~1M faces, thick-walled terracotta (wall thickness ≈ 5–10 % of fragment extent), clearly visible fracture surfaces, not necessarily axially symmetric. Output is rigid transforms. Sets of 4 up to dozens of fragments, with missing pieces and intruders from other objects.

---

# PAPER 1 — SPPD: A Novel Reassembly Method for 3D Terracotta Warrior Fragments Based on Fracture Surface Information

Yao, Chu, Tang, Wang, Cao, Zhao, Li, Geng, Zhou. *ISPRS Int. J. Geo-Inf.* 2021, 10(8), 525. Northwest University, Xi'an + Beijing Normal University.

## 1. What it does

SPPD is an end-to-end automatic reassembly pipeline for 3D scanned terracotta warrior fragments that works purely from **fracture surface** geometry, i.e. the freshly broken interior faces, not the decorated exterior. It solves the two classic sub-problems together: (a) deciding **which pairs of fragments actually adjoin** (a binary match/no-match decision over all fracture-surface pairs), and (b) computing the **rigid transform** that seats one against the other. The pair decision is made by a learned Siamese embedding of the fracture surface point clouds (SiamesePointNet); the transform is computed by a coarse-to-fine cascade of PCA principal-axis alignment followed by Deep Closest Point (DCP). The two stages are wrapped in a **greedy agglomerative loop** so that multiple fragments are progressively merged into one object. The stated motivation is that manual reassembly of excavated warrior fragments is slow and risks secondary physical damage.

## 2. Full algorithm pipeline

### 2.0 Acquisition and preprocessing

- Scanner: **Artec Eva** handheld structured-light scanner. Data from the Emperor Qinshihuang's Mausoleum Site Museum, digitised by the Visualization Laboratory of Northwest University.
- Denoising: **Artec Studio Professional** then **Geomagic Wrap 2017**. (Identical to our own Geomagic-produced input, so the noise characteristics should transfer directly.)
- Representation: point cloud (they discard connectivity after segmentation).

### 2.1 Fracture surface extraction

- The paper **delegates** this: "This work follows Li's approach [11] to accomplish region segmentation and to strip fracture surfaces from the entire fragment." Reference [11] is Li, Geng, Zhou, *Pairwise Matching for 3D Fragment Reassembly Based on Boundary Curves and Concave-Convex Patches*, IEEE Access 8:6153–6161, 2020. **No formulas, thresholds, or curvature criteria are given in SPPD itself.** This is the single biggest implementation gap in the paper — the step our project most needs is the one they cite away.
- Rationale given for choosing the fracture surface over the intact surface: larger area, richer information, continuous shape.
- Post-extraction: **random sampling** to a fixed point count, plus normalisation, to feed the network.

### 2.2 Matching stage — SiamesePointNet

Two weight-shared PointNet branches, then an energy/scoring head.

- Inputs: two point sets `A = {x_1 … x_N} ⊂ R^3`, `B = {y_1 … y_N} ⊂ R^3`.
- Feature extractor: `F : R^{N×3} → R^K`, implemented as PointNet (per-point convolutions, input/feature transformation nets, max-pooling to a global vector). **K = 1024** throughout all experiments.
- Weight sharing between the two branches; weights adjusted by match/no-match labels so PointNet migrates from classification/segmentation to a similarity task.
- Scoring: `E(A,B) = ‖F(A) − F(B)‖_2`. The output similarity score `fp` is *negatively* correlated with `E` and lies roughly in [0, 1]; it is read as a matching probability.
- Loss (contrastive), as printed in the paper:

  `loss = (1/n) Σ_i [ fg_i · √(fp_i) + (1 − fg_i) · max(1 − √(fp_i), 0) ]`

  where `fg_i ∈ {0,1}` is the ground-truth label (1 = match) and `fp_i` is the predicted score. The OCR of the exponents is unreliable; this is the standard Chopra/Hadsell/LeCun contrastive form with a margin of 1.
- Invariance claim: PointNet's feature transform + pooling makes the embedding insensitive to pose and coordinate system, which matters because two matching fracture surfaces are scanned in arbitrary independent frames.

**Pair selection:** for a set of fragments, every fracture surface of every fragment is scored against every fracture surface of every other fragment. The single **highest-scoring pair** is chosen for registration (Figure 4 shows a 2-fragment example where the "2–2" surface pair wins).

### 2.3 Registration stage — PCA coarse alignment

The idea (credited to Makovetskii et al. [38]) is to align not the surfaces but their **Main Axes System (MAS)**: the three mutually orthogonal unit PCA eigenvectors of the fracture surface point set. Because the fracture surface is a subset of the fragment and shares its coordinate frame, the transform that aligns the two MASs can be applied to the whole fragment mesh.

- `MA = [a_1 a_2 a_3]`, `MB = [b_1 b_2 b_3]` — 3×3 matrices whose columns are the unit eigenvectors, ordered by descending eigenvalue.
- **Sign ambiguity:** each eigenvector has an arbitrary sign, giving 2³ = 8 pairings. Requiring a right-handed frame (`det = +1`) removes half, leaving **exactly 4 hypotheses**:

  | # | MA variant | MB |
  |---|---|---|
  | 1 | `[ a1,  a2,  a3]` | `[b1, b2, b3]` |
  | 2 | `[−a1, −a2,  a3]` | `[b1, b2, b3]` |
  | 3 | `[−a1,  a2, −a3]` | `[b1, b2, b3]` |
  | 4 | `[ a1, −a2, −a3]` | `[b1, b2, b3]` |

- Rotation: **`R = MB · MA^{-1}`** (eq. 2). Since MA is orthonormal, `MA^{-1} = MA^T`, so `R = MB · MA^T` — cheap and exact.
- Centroids (eqs 3, 4): `Ā = (1/N) Σ x_i`, `B̄ = (1/N) Σ y_j`.
- Translation (eq. 5 as printed): `T = B̄ − Ā`, and `A* = A · R + T`.

  ⚠ **This is algebraically wrong as printed.** Rotating then adding `B̄ − Ā` only lands the centroid correctly if `R ≈ I`. The correct form, which is clearly what they implemented, is `A* = (A − Ā) · R + B̄`. Implement the corrected version.

- **Hypothesis selection by symmetric nearest-neighbour distance** (eqs 6–9). For candidate transform result `A*` and target `B`:

  `D_AB = (1/n) Σ_j ‖p*_{y_j} − p_{x_i}‖`, where `p*_{y_j} = argmin_i ‖p_{y_j} − p_{x_i}‖`
  `D_BA = (1/n) Σ_j ‖p*_{x_j} − p_{y_i}‖`, where `p*_{x_j} = argmin_i ‖p_{x_j} − p_{y_i}‖`

  i.e. mean one-sided closest-point distance in each direction. **The scheme with the smallest D wins.** Small D = high overlap = good registration. This doubles as a generic, training-free quality/rejection score.

### 2.4 Registration stage — DCP fine alignment

- Deep Closest Point (Wang & Solomon, ICCV 2019): point cloud embedding network + attention module + differentiable SVD layer producing `R, T` in a **single forward pass**, no iteration.
- Sold explicitly as avoiding ICP's iteration, point-level optimisation cost, and non-convergence risk.
- DCP is initialised by the PCA result, because DCP alone diverges badly (see results).

### 2.5 Multi-fragment assembly — greedy agglomerative loop

1. Build a working set of all fragments to be reassembled.
2. If `|set| ≤ 1`, stop.
3. Extract fracture surfaces; compute the similarity score for **every** cross-fragment fracture-surface pair.
4. Take the argmax pair. If `max score ≤ ζ`, stop (no matching relationship remains). **The numeric value of ζ is never given in the paper.**
5. Register that pair (PCA → DCP).
6. **Update the set**: delete the two source fragments, insert the merged result as a single new fragment. The two fracture surfaces that were just consumed are **blocked** and excluded from all later rounds.
7. Go to 2.

The authors call this greedy strategy globally optimising and mismatch-avoiding; in reality it is a locally greedy merge with no backtracking, no global consistency enforcement, and no loop closure.

## 3. Input assumptions

| Assumption | Value / stance |
|---|---|
| **Thick vs thin** | Thick. Terracotta warrior body wall. Fracture surfaces are treated as substantial 2-manifold patches with real area, rich enough to embed and PCA. Matches our 5–10 % wall thickness exactly. |
| **Axial symmetry** | **Not required.** Nothing in the method uses a rotation axis, profile, or surface of revolution. Test objects are a sculptural arm and a whole warrior. |
| **Texture / colour** | **Not used at all.** Pure geometry, `R^3` point coordinates only. |
| **Training data** | **Required, and substantial**: 1800 labelled matched pairs + 1560 non-matched pairs for SiamesePointNet, plus DCP's own pretraining. This is the fatal blocker for us. |
| **Templates / reference model** | Not needed. |
| **Manual interaction** | Not needed for the automatic portion, but explicitly required to finish real objects: the 49-fragment warrior needed expert manual assembly for the residual 9 fragments and to join the 7 auto-assembled groups. |
| **Scan resolution** | Artec Eva class (roughly 0.2–0.5 mm point spacing). Network input downsampled to 2048 points regardless, so the scoring stage is resolution-agnostic. |
| **Scale** | Absolute millimetres. Robustness sweeps use 10–100 mm displacements; reported MAE 0.6–3.1 mm. Fragment volumes ~200–2000 cm³. |
| **Fracture surface integrity** | **Strongly assumed intact.** Explicitly identified as the failure mode when violated. |
| **Fracture surface count** | Implicitly, each pairing consumes one whole contiguous fracture surface per fragment, and the two surfaces have near-identical extent. Not stated, but PCA-of-the-whole-patch only works under that condition. |

## 4. Reported results, datasets, runtime, limitations

**Matching accuracy (SiamesePointNet), effect of input point count (Table 1):**

| Points | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| Acc (avg class) | 92.85 % | 93.10 % | 95.23 % | **96.53 %** | 95.60 % |
| Acc (overall) | 92.86 % | 93.15 % | 95.39 % | **96.58 %** | 95.83 % |

Saturation at 2048; 2048 chosen by Occam's razor. At 1024 points, per-class accuracy was 96.77 % on matched pairs and 93.69 % on non-matched.

**Matching, versus hand-crafted descriptors (Table 2, 2048 points):**

| Method | FPFH | SHOT | SPPD |
|---|---|---|---|
| Acc (avg class) | 77.93 % | 77.04 % | **95.60 %** |
| Acc (overall) | 77.38 % | 77.08 % | **95.83 %** |

**Registration, 360 known-matching fracture-surface pairs (Table 3):**

| Method | Success rate | MSE | RMSE | MAE | Mean iterations |
|---|---|---|---|---|---|
| ICP | 20.56 % | 9.48 | 3.08 | 1.73 | 23.74 |
| RANSAC (on FPFH) | 23.33 % | 1.17 | 1.08 | **0.64** | — |
| NDT | 4.72 % | 13.83 | 3.72 | 1.95 | — |
| PCA only | 51.67 % | 5.28 | 2.30 | 1.18 | — |
| DCP only | 21.38 % | 4.26 | 2.06 | 1.01 | — |
| PCA + ICP | 35.83 % | 4.77 | 2.19 | 1.21 | 9.03 |
| **PCA + DCP** | **53.89 %** | 3.11 | 1.76 | 0.92 | — |

"Failure" = direction or distance differs grossly from ground truth; failures are excluded before computing MSE/RMSE/MAE, so RANSAC's low error is an artefact of its 23 % success rate. Non-convergence for ICP methods declared at **200 iterations**.

**Robustness sweeps** (mean MAE over all samples): initial rotation capped at 15°, 30°, 45°, 60°, 75°, 90°, 105°, 120°; initial translation 10–100 mm in 10 mm steps; Gaussian noise starting at SNR 100 dB attenuated by 0–70 %. PCA+DCP was the most stable across all three. RANSAC's mean MAE exceeded 30 (stable but uniformly bad); NDT frequently inverted the registration direction, so its MAE was omitted from the plots.

**Multi-fragment demonstrations:**
- A terracotta **arm, 5 fragments**. Largest fragment ≈ 2000 cm³ (printed as cm², an obvious typo), smallest under a tenth of that. Fully and correctly reassembled through 4 greedy merge steps. Note their own observation: after fracture-surface extraction the *area* disparity between a huge and a tiny fragment largely disappears, which is why size imbalance did not hurt.
- A **whole warrior, 49 fragments**. Result: **7 partially assembled groups**, **9 fragments left with no match**, remainder joined manually by experts. Head and feet were never recovered (presumed destroyed).

**Hardware:** Intel Core i7-9700F @ 3.0 GHz plus **two NVIDIA RTX 2080 Ti GPUs**. **No wall-clock runtime is reported anywhere in the paper** — neither training nor inference nor per-pair registration.

**Stated limitations:**
1. Severely eroded/damaged fracture surfaces produce **false negatives** — Figure 14 shows a genuinely matching pair rejected by SPPD.
2. When two fracture surfaces are only *partially* similar (one broke further after the other), registration is visibly wrong in detail (Figure 15).
3. The method only handles the "high integrity" subset of a real object; the hard remainder is left to human experts.
4. Cost grows with fragment count. They argue the practical cap is ~100 fragments because excavation has weak spatial continuity (co-located fragments belong to the same warrior), and note that a faster search is future work.

## 5. Relevance verdict for our task: **4 / 5**

**Why high.** This is the closest published match to our exact problem: same object class, same wall thickness regime, same scanner-plus-Geomagic pipeline, same fracture-surface-only premise, no reliance on colour, no reliance on axial symmetry, and an explicit multi-fragment orchestration loop rather than a pairwise-only demo. The skeleton — extract fracture surfaces, score all cross-fragment surface pairs, coarse-align by principal axes with 4 sign hypotheses, disambiguate by symmetric nearest-neighbour distance, refine, merge greedily, block consumed surfaces, terminate on a score threshold — is directly implementable in numpy/scipy/open3d and runs comfortably on CPU. The empirical ranking of coarse registration methods is a gift: it tells us in advance that ICP-from-scratch (20.6 %), FPFH+RANSAC (23.3 %) and NDT (4.7 %) will all disappoint on this data, and that a global coarse initialisation is worth more than a better refiner.

**Why not 5.** Two of the four named components are unusable for us. SiamesePointNet needs 3360 labelled fragment pairs and a GPU; we have no ground truth and no training data. DCP needs a pretrained model and a GPU. We must substitute both. The fracture-surface extraction step — arguably the hardest and most important part for us — is cited away to another paper with no detail. The whole-object result (49 fragments to 7 groups plus 9 orphans plus manual finishing) sets a sober expectation for what "success" looks like on a large mixed set. And no runtime is reported, so we get no timing guidance.

## 6. Concrete reusable ideas

### Borrow

1. **PCA main-axes coarse registration with exactly 4 right-handed sign hypotheses.** `R = MB · MA^T` (MA, MB orthonormal eigenvector matrices sorted by descending eigenvalue), tried in the 4 sign variants `[+,+,+]`, `[−,−,+]`, `[−,+,−]`, `[+,−,−]` applied to MA. Cost is trivial: four 3×3 matrix products per candidate pair. **But fix the translation to `A* = (A − Ā)·R + B̄`.**

2. **Symmetric mean nearest-neighbour distance as the universal scoring and rejection function** (eqs 6–9): `D = ½(D_AB + D_BA)` with each term a mean one-sided closest-point distance via a KD-tree. Training-free, cheap, and reusable at three different places in our pipeline: choosing among the 4 PCA hypotheses, scoring a candidate pair, and gating the final acceptance. Also use it for a **penetration check**, which the paper omits and which we need.

3. **Match on the fracture surface, transform the whole fragment.** Because the fracture patch is a subset of the fragment mesh and shares its frame, everything computed on the patch applies verbatim to the parent mesh. This is what makes the whole approach affordable: we do all expensive geometry on a few thousand fracture-surface points instead of on 350k–670k vertices.

4. **2048 points per fracture surface as the working resolution.** Their ablation shows discrimination saturating there (96.5 %) with 1024 nearly as good (95.2 %) and 4096 slightly *worse* (95.6 %). Downsample fracture patches to ~2048 points for scoring and coarse alignment; keep full resolution only for the final refinement pass.

5. **Their own observation that fracture-surface extraction equalises fragment size.** A 2000 cm³ fragment and a 200 cm³ fragment have comparable fracture-surface areas. This means our pairwise scoring does not need to be size-normalised in any special way, and small fragments are not intrinsically disadvantaged.

6. **The greedy agglomerative loop with surface blocking.** Merge the best-scoring pair, replace both fragments with the merged assembly, mark the two consumed fracture surfaces as unavailable, recompute, repeat until the working set has one element or the best score falls below ζ. The blocking rule is what prevents a surface being reused by two different neighbours. For our "pieces missing / intruders present" requirement, the ζ-termination is exactly the right mechanism — it naturally leaves orphans unassembled instead of forcing bad joins.

7. **The empirical baseline table as a design prior.** Do not build a pipeline whose coarse stage is FPFH+RANSAC or NDT. Do not run ICP without a good initialisation. Budget ~50 % pairwise success as a realistic ceiling on real eroded material, and design the multi-fragment logic to tolerate that.

8. **Robustness test protocol.** Perturb a known-good alignment by rotations up to 120° and translations up to 100 mm, and add Gaussian noise, then measure mean MAE. We have no ground truth, but we can manufacture it: take one fragment, cut it synthetically, perturb, and check recovery. This gives us a regression test suite without museum ground truth.

9. **Millimetre-scale sanity anchors.** Their good registrations land at MAE 0.9–1.2 mm and RMSE 1.8–2.3 mm on warrior-scale fragments. That is a usable acceptance threshold for our own residuals, and a hint that sub-millimetre precision is neither achievable nor necessary.

### Avoid

- **SiamesePointNet.** 3360 labelled pairs and a GPU. Non-starter. Replace the pair-scoring function with a training-free geometric alternative.
- **DCP.** Pretrained + GPU. Replace with point-to-plane ICP seeded by the PCA hypothesis. Their PCA+ICP number (35.8 %) is worse than PCA+DCP (53.9 %), but PCA+ICP still beats raw ICP by 15 points, and open3d's point-to-plane ICP with a robust (Tukey/Huber) kernel is materially better than the plain point-to-point ICP they benchmarked.
- **The printed translation formula** `T = B̄ − Ā` with `A* = A·R + T`. Wrong. Centre before rotating.
- **Unguarded PCA on a fracture patch.** If the two largest eigenvalues are close, the first two axes are arbitrary and all 4 hypotheses are junk. Add an eigenvalue-gap guard (e.g. require `λ1/λ2 > 1.2` and `λ2/λ3 > 3`) and fall back to a different initialiser when it fails. Our thick-walled fragments help here: a fracture surface on a 5–10 % wall is an elongated curved band, so `λ1 ≫ λ2 ≫ λ3` is the normal case and the frame is well conditioned. Near-square or near-circular patches are the danger.
- **Assuming one fracture surface per fragment pair with matching extent.** A fragment in a dozens-of-pieces set touches several neighbours along one contiguous broken rim. We must sub-segment the fracture region into per-neighbour patches, or use a local/partial matching formulation, rather than PCA-ing one giant fracture region.
- **The claim that the greedy loop "guarantees global optimisation".** It does not. It is a locally greedy merge with no backtracking. Plan for a global consistency or cycle-consistency pass, or at minimum keep the top-k pair hypotheses instead of only the argmax.

---

# PAPER 2 — Classifying Fragments of Terracotta Warriors Using Template-Based Partial Matching

Du, Zhou, Yin, Wu, Shui. *Multimedia Tools and Applications* (2018) 77:19171–19191. Beijing Normal University.

## 1. What it does

This paper does **not** reassemble anything. It solves the upstream triage problem: given a pile of excavated fragments, sort them into body-part categories (head, torso, hand, skirt, leg) so that later reassembly only compares fragments within a category, avoiding "one-to-all" matching. It reframes classification as **part-in-whole partial shape matching**: an expert picks a handful of small, locally distinctive template regions (an ear, the nail on the armour suit, a fist, the skirt edge), each of which occurs in exactly one category; a fragment is assigned to a category if *any* sub-region of its surface is geometrically similar to that category's template. Crucially, and in direct opposition to SPPD, it works on the **intact decorated surface** and deliberately avoids the fracture surface. The stated advance over prior art is that it beats keypoint-descriptor methods (spin image, SHOT, HKS) on smooth, featureless fragments, where keypoint detectors have nothing to latch onto.

## 2. Full algorithm pipeline

### 2.1 Categories and templates

- Five categories: `C1` Head, `C2` Torso, `C3` Hand, `C4` Skirt, `C5` Leg.
- **Eight templates, selected manually by expert knowledge**: eyes, ears, mouth, nose (Head); the nail on the suit (Torso); fist ×2 (Hand); skirt edge (Skirt); a leg region (Leg). Selection criterion: the region must uniquely define one category.

### 2.2 Largest enclosed geodesic disk of a template

A template is hand-drawn and therefore has an arbitrary boundary, which makes it incomparable with regions on a fragment. They regularise it to a disk.

Definition: on a 3D manifold surface `S`, the geodesic disk `GD(p, r)` is the set of points whose **geodesic** (shortest polygonal path along the mesh) distance from `p` is ≤ `r`.

Algorithm for the largest enclosed geodesic disk of a template patch:
1. Take **all boundary points of the template patch as source points**.
2. For every non-boundary point, compute the shortest geodesic distance to the source set.
3. The **maximum** of those distances is the radius `r`; the corresponding point is the centre `p`.

This is the geodesic inradius, and it is elegant: it finds the deepest interior point of the hand-drawn patch and the largest disk that fits inside it, making template and fragment regions directly comparable.

Geodesics computed with the **MMP algorithm** (Mitchell, Mount, Papadimitriou 1987 — exact discrete geodesics).

Measured template radii (Table 2, units are presumably mm):

| Category | Template | Points in template | Points in disk | Disk radius |
|---|---|---|---|---|
| C4 | Skirt | 1687 | 1064 | 118.77 |
| C1 | Ears | 1450 | 1150 | 59.71 |
| C5 | Leg | 2339 | 1130 | 56.87 |
| C3 | Hand-1 | 2446 | 1746 | 47.69 |
| C1 | Nose | 842 | 409 | 41.08 |
| C3 | Hand-2 | 1115 | 715 | 33.99 |
| C1 | Mouth | 639 | 369 | 32.30 |
| C2 | Nail | 405 | 223 | 18.90 |

### 2.3 Coarse matching — Normal Distribution Descriptor (NDD)

NDD (Martinek, Grosso, Greiner 2014) is a 10-bin histogram of how neighbouring normals deviate from the reference point's normal.

1. **Sphere radius**: for the template, the NDD radius is the **shortest Euclidean distance from the disk centre `p` to the template's boundary points**. (Note this is a *different, smaller* radius than the geodesic disk radius, and it is Euclidean, not geodesic.)
2. Collect all points inside that Euclidean sphere around the reference point.
3. Compute `dot(n_q, n_p)` for each neighbour `q`. Nominal range `[−1, 1]`, but **restricted to `[0, 1]` in their experiments**, because a normal pointing into the interior of the surface is meaningless.
4. Uniformly divide the range into **10 bins**. Each bin holds the *percentage* of neighbours whose dot product falls in it. That 10-vector is the NDD.
5. Compute NDD for **every point of the fragment** using the **same radius**.
6. Similarity = **Pearson correlation coefficient** between the template NDD and each fragment-point NDD:

   `r = Σ(X_i − X̄)(Y_i − Ȳ) / [ √(Σ(X_i − X̄)²) · √(Σ(Y_i − Ȳ)²) ]`

   Range `[−1, 1]`; they use the absolute value in `[0, 1]`. They state that `> 0.6` counts as "strongly relevant".
7. **Candidate threshold: `r > 0.7`.** Deliberately loose, "to ensure the covering of actual matching points."
8. **Candidate cap: keep the top 30** most similar points. If fewer than 30 pass, keep all. (A worked example earlier in the paper shows 104 candidates for a torso fragment, before the cap rule is introduced in the experiments section.)

NDD is used first specifically because it is cheap and fast, to shrink the candidate set before the expensive descriptor runs.

### 2.4 Fine matching — modified Point Feature Histogram

Standard PFH (Rusu et al. 2008) builds a Darboux frame for every *pair* of points in a spherical neighbourhood. Two modifications are made.

**Modification 1 — geodesic disk instead of Euclidean sphere.** Neighbourhoods are `GD(p, r)` with `r` equal to the template's largest-enclosed-geodesic-disk radius. Rationale: Euclidean spheres on a triangle mesh ignore connectivity and can bridge across a fold or across the wall to the far side; geodesic disks preserve connectivity and encode more shape.

**Modification 2 — centre-to-neighbour pairs only.** Instead of all `O(k²)` pairs within the disk, only the `O(k)` pairs `(p, q)` from the centre to each disk point are used. Explicitly justified by the high point count of the scans.

Feature computation for a pair. Source `p_s` is the point of the pair whose normal makes the smaller angle with the connecting line; target is `p_t`. Darboux frame at `p_s`:

- `u = n_s`
- `v = (p_t − p_s) × u`
- `w = u × v`

Features (eq. 1):

- `α = v · n_t`
- `φ = u · (p_t − p_s) / d`
- `θ = arctan(w · n_t, u · n_t)`
- with `d = ‖p_t − p_s‖_2`

**Binning: `b = 2` subdivisions per feature ⇒ `b³ = 8` bins.** Each bin stores the percentage of pairs falling into that combination. Result: an **8-dimensional descriptor**.

Properties they demonstrate: **resolution invariant** (Figure 6 — subdividing the torso template leaves the descriptor essentially unchanged), invariant to triangulation, and invariant to rigid transformation.

**Fine match threshold: Pearson correlation between the two 8-vectors `> 0.6`** ⇒ the two disks are similar.

**Decision rule:** if **at least one** geodesic disk on the fragment correlates above 0.6 with a template, the fragment is assigned to that template's category. A fragment may legitimately receive several categories (their example G10–22-5 is both Skirt and Leg); any of them counts as correct.

### 2.5 The geodesic-disk speed optimisation

This is the most quantitatively valuable trick in the paper. Naively, computing `GD(p, r)` requires geodesic distances from `p` to all other mesh points. But:

- Geodesic distance between two surface points is **never smaller** than their Euclidean distance.
- Therefore every point of `GD(p, r)` lies inside the Euclidean ball `B(p, r)`.

So: extract the submesh inside the Euclidean ball of radius `r`, run the geodesic computation only on that submesh, and back-map the resulting disk onto the full model.

**Measured:** 104 geodesic disks took **13.637 s** with the optimisation versus **1347.155 s** without. A ~**99× speedup**.

### 2.6 Search-order pruning

- Sort **fragments** by approximate diameter, ascending.
- Sort **templates** by geodesic disk radius, ascending.
- Start with the smallest fragment against the smallest template; escalate to larger templates only on failure.
- **Skip any comparison where the fragment's approximate diameter is less than the diameter of the template's geodesic disk** — the region cannot possibly fit.

## 3. Input assumptions

| Assumption | Value / stance |
|---|---|
| **Thick vs thin** | Thick (warrior body), but thickness is irrelevant to the method — everything happens on the outer surface. |
| **Axial symmetry** | **Not required**, and the paper explicitly criticises prior pottery classifiers for depending on the rotational-axis profile, noting that worn fragments make profiles impossible to extract. |
| **Texture / colour** | **Not used.** Purely geometric (positions + normals). The paper explicitly argues *against* colour/texture classification because archaeological fragments share textures and colours. |
| **Training data** | None in a machine-learning sense, but it needs **expert-selected templates**, which is a human-supplied prior of comparable practical cost. |
| **Templates** | **Mandatory and central.** 8 hand-picked regions. The method cannot run without them. |
| **Manual interaction** | Required once per object class, for template selection. Classification itself is automatic. |
| **Scan resolution** | High-resolution scans assumed ("the fragments are scanned in a high resolution and have a large amount of points" — the stated motivation for the `O(k)` PFH modification). Descriptor shown to be resolution-invariant, so mixed resolutions across the set are fine. |
| **Scale** | Absolute, consistent, and **shared between template and fragment** — the radii are transferred verbatim from template to fragment, so the two must be in the same units at the same scale. |
| **Surface used** | **Intact surface only.** Fracture surfaces are treated as a nuisance that corrupts competing methods. |
| **Mesh quality** | Needs a manifold triangle mesh with reliable normals and connectivity for geodesics. Point clouds alone would not work. |

## 4. Reported results, datasets, runtime, limitations

**Dataset:** 113 terracotta warrior fragments scanned at real archaeological sites, pre-classified by archaeologists (used as ground truth), denoised beforehand. 8 templates.

**Accuracy — overall 89.3 %:**

| Category | Fragments | Correct | Ratio |
|---|---|---|---|
| C1 Head | 8 | 7 | 87.5 % |
| C2 Torso | 75 | 68 | 90.7 % |
| C3 Hand | 10 | 8 | 80 % |
| C4 Skirt | 10 | 9 | 90 % |
| C5 Leg | 10 | 9 | 90 % |

The dataset is heavily unbalanced — 75 of 113 fragments are torso — so the headline number is dominated by one class.

**Runtime.** Intel Xeon 2.13 GHz, 6 GB RAM, 64-bit. Average per fragment per category:

| Category | Coarse (NDD) | Fine (modified PFH) |
|---|---|---|
| C1 | 18.645 s | 15.583 s |
| C2 | 9.223 s | 13.607 s |
| C3 | 15.734 s | 6.345 s |
| C4 | 14.316 s | 25.589 s |
| C5 | 11.595 s | 10.869 s |

So roughly **16–40 s per fragment per category**, on 2010-era hardware. Cost scales with fragment vertex count and template radius.

**Comparisons.** Benchmarked against spin image, SHOT, and HKS partial matching (with ISS keypoint detection). Their method wins across all categories, with the largest margins on C4 (skirt) and C5 (leg). Three explanations given, all of which are useful diagnoses:

1. **ISS and HKS keypoints frequently land on the fracture surface.** Those features do not exist on the intact reference model, so correspondence fails outright.
2. **Keypoint descriptors have no discriminative power on smooth regions.** Skirt and leg fragments are smooth; region-based descriptors still work there because they integrate over an area.
3. Keypoint methods are **sensitive to triangulation**; the geodesic-disk PFH is not.

**Stated limitations.**
- Templates must be selected by an expert and must be **locally unique** to a category. No procedure is given for verifying uniqueness or for handling an object class where no such region exists.
- Fragments can match several categories; the evaluation counts any of them as correct, which is a generous metric.
- Generalisation is explicitly conditional: "Our algorithm can be expanded to any other archaeological artifacts **if they have unique regions** from which their categories can be identified."

## 5. Relevance verdict for our task: **2 / 5**

**Why not lower.** Several component techniques are directly transplantable and genuinely valuable to us: the geodesic-disk region descriptor, the Euclidean-ball bound that makes geodesic computation ~99× cheaper, the `O(k)` centre-to-neighbour PFH variant, the cheap-then-expensive descriptor cascade with concrete correlation thresholds, and the size-based pruning rule. Its negative results are as useful as its positive ones: it is direct evidence that ISS/HKS/spin-image/SHOT keypoint pipelines fail on smooth terracotta, which tells us not to build our fracture-surface matcher on a keypoint detector. And its observation that keypoint detectors preferentially fire on fracture surfaces is, for us, inverted into a feature — that same bias is a free fracture-surface saliency signal.

**Why not higher.** The paper's actual goal is orthogonal to ours. We need rigid transforms; it produces category labels. Its whole apparatus depends on **expert-selected templates that uniquely identify a category**, which we do not have, cannot obtain (no ground truth, no GUI, no expert in the loop), and could not define for the later sets that include pots and plates — a plate has no locally unique "ear" or "armour nail". It matches on the **intact surface**, the opposite of our signal. Its runtime, 16–40 s per fragment per category on old hardware, would be tolerable, but its dependence on exact MMP geodesics over 350k–670k vertex meshes would not be. And its classification stage only pays off at scale: for a 4-fragment set it is pure overhead, and even for dozens of fragments a cheaper unsupervised grouping (by wall thickness, curvature statistics, or colour) would serve the same triage purpose without templates.

## 6. Concrete reusable ideas

### Borrow

1. **The Euclidean-ball bound for geodesic neighbourhoods.** Because geodesic distance ≥ Euclidean distance, `GD(p, r) ⊆ B(p, r)`. Extract the submesh inside the Euclidean ball, compute geodesics only there, back-map. Measured **13.637 s vs 1347.155 s for 104 disks**. This makes any geodesic-based local descriptor affordable on our 350k–670k vertex meshes, and it is a handful of lines with trimesh plus a KD-tree.

2. **The `O(k)` centre-to-neighbour PFH variant.** Compute the three Darboux features only between the region centre and each neighbour, not between all `O(k²)` pairs. With `k` in the thousands this is the difference between milliseconds and minutes per descriptor. Use `u = n_s`, `v = (p_t − p_s) × u`, `w = u × v`, then `α = v·n_t`, `φ = u·(p_t − p_s)/d`, `θ = arctan2(w·n_t, u·n_t)`.

3. **Two-stage cheap-then-expensive descriptor cascade with concrete thresholds.** Stage 1: a 10-bin normal-deviation histogram (NDD), Pearson `r > 0.7`, keep the **top 30** candidates. Stage 2: the expensive region descriptor, Pearson `r > 0.6` for acceptance. This exact pattern maps onto our pairwise problem: prune the `O(n²)` fracture-surface pair space with a cheap global signature, then run the expensive geometric verification only on survivors.

4. **Pearson correlation as the histogram similarity measure**, rather than L2 or chi-squared. It is scale- and offset-invariant across histograms, which matters when two fracture patches are sampled at different densities. Their calibration is worth reusing directly: `> 0.6` = strongly relevant, `> 0.7` = candidate-worthy, `1.0` = identical.

5. **The Normal Distribution Descriptor itself**, as our cheap first-stage signature. Ten bins of `dot(n_q, n_p)` over a fixed-radius neighbourhood, **restricted to `[0, 1]`** because inward-pointing normals are meaningless. This is trivially vectorisable in numpy over an open3d KD-tree, costs almost nothing, and gives a rotation-invariant local roughness signature — very well suited to characterising the sandpaper-like texture of a fracture surface versus the smooth-or-relief exterior.

6. **The largest-enclosed-geodesic-disk construction** for regularising an arbitrary-boundary patch. All boundary vertices as multi-source; the interior vertex maximising the distance-to-boundary is the centre; that distance is the radius. Applied to our per-neighbour fracture patches, this gives a canonical, boundary-independent centre and scale for each patch — useful for anchoring a local descriptor and for measuring whether two patches are even comparable in size.

7. **Size-based pruning before any expensive comparison.** Sort by extent and skip any pair where one item is too small to contain the other. Our analogue: skip a fracture-surface pair when the two patch areas, bounding-box diagonals, or geodesic-disk radii differ by more than a factor (say 2–3×), since matching fracture surfaces must have comparable extent. On a set of dozens of fragments with many surfaces each, this alone removes a large fraction of the pair space for free.

8. **Region descriptors, not keypoint descriptors.** Their central empirical finding: on smooth terracotta, ISS/HKS keypoints are non-discriminative and land in the wrong places, while descriptors integrated over a region still work. Our fracture surfaces are rough but locally self-similar, so the same logic applies — describe patches, not points.

9. **The keypoint bias as an inverted signal.** They complain that ISS and HKS keypoints "lie on the fracture surface of the fragment". For us that is exactly where we want to look. A cheap keypoint-density or high-curvature-density map is a usable prior for fracture-surface segmentation, essentially for free.

10. **Resolution invariance as an explicit design requirement.** Their Figure 6 test — subdivide a template, recompute the descriptor, confirm it is unchanged — is a cheap unit test we should replicate on whatever descriptor we build, given our fragments span 350k–670k vertices and may be decimated inconsistently.

### Avoid

- **The entire template-based classification framing.** No templates, no expert, no full reference model, no GUI, and later object sets (pots, plates) have no locally unique category-defining region. Skip the classification stage; if we need triage on large sets, do it unsupervised (wall thickness, mean curvature statistics, colour histogram, fragment size) rather than by templates.
- **Exact MMP geodesics** on full-resolution meshes. Even with the Euclidean-ball trick, exact discrete geodesics on our mesh sizes are too slow. Use the heat method (`potpourri3d`), or Dijkstra on the edge graph of a decimated mesh, and accept the approximation.
- **Matching on the intact surface.** For reassembly, the fracture surface carries the signal; the exterior relief is decoration that repeats across the object (which is precisely why they could use it for *classification*, and precisely why it is useless for *pairing*).
- **`b = 2` bins per PFH feature (8-dim total)** for our purpose. That is enough to separate five body-part categories, but far too coarse to discriminate one candidate fracture pairing from another. If we adopt this descriptor for pairing, use `b = 5` (125 bins) or the standard FPFH 33-dim, and reserve the 8-dim version for cheap first-stage pruning only.
- **Their per-fragment runtimes as a budget.** 16–40 s per fragment per category, times dozens of fragments, times pairs, blows our minutes-not-hours requirement immediately. Any geodesic machinery must be restricted to a small number of candidate patches, never applied per-vertex over a whole fragment.
- **Accepting a match on a single passing region.** Their "at least one disk above 0.6" rule is appropriate for classification, where a false positive is cheap. For reassembly a false positive corrupts the entire assembly, so we need consensus across many correspondences plus a physical-plausibility check.

---

# CROSS-PAPER SYNTHESIS

Together these papers point to a five-stage, training-free, CPU-feasible pipeline.

**Segment first, and treat it as the hard part.** Both papers make the fracture surface the pivot — SPPD by matching on it, Du et al. by carefully avoiding it — yet neither gives an extraction algorithm. Since roughness is what separates a fracture face from a slip-finished exterior, segment by local normal variation. Du et al.'s 10-bin NDD histogram, restricted to `[0,1]`, is a ready-made per-vertex roughness signature; cluster it and take connected components. Their observation that generic keypoint detectors preferentially fire on fracture surfaces is a free second prior.

**Sub-segment per neighbour.** SPPD implicitly assumes one fracture surface per pairing with matching extent. Thick-walled fragments in a dozens-of-pieces set break along a rim contacting several neighbours, so split the fracture region into per-neighbour patches before PCA.

**Prune the `O(n²)` pair space cheaply.** Adopt Du et al.'s cascade wholesale: a cheap global signature per patch, Pearson correlation, keep roughly the top 30, and pre-filter by size — patch areas differing more than 2–3× cannot mate. Their Euclidean-ball bound on geodesic neighbourhoods (13.6 s versus 1347 s for 104 disks) and their `O(k)` centre-to-neighbour PFH make any region descriptor affordable at our mesh sizes.

**Align coarsely by principal axes, then refine.** SPPD's four right-handed sign hypotheses with `R = MB·MA^T`, disambiguated by symmetric mean nearest-neighbour distance, is the highest-value transplant: exact, four 3×3 products, no training. Fix the translation to `(A − Ā)R + B̄`, guard against eigenvalue degeneracy, and substitute point-to-plane ICP for DCP. Thick walls make fracture patches elongated bands, so PCA axes are well conditioned.

**Merge greedily with blocking and a rejection threshold.** Consume surfaces on merge, terminate on a score floor. That threshold is what leaves orphans and intruders unassembled instead of forcing bad joins. Expect roughly 50 % pairwise success and partial assemblies — SPPD's 49-fragment warrior yielded 7 groups plus 9 orphans.
