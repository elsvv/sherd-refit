# S1 — colour into the generator, and two coloured mixed collections with ground truth

**Date:** 2026-09-11. **Tree:** branch `rust-core`, `ff6a1fc` (task C8) → `S1.1`…`S1.4`. **Machine:**
Apple M2 Pro, 10 cores (6P + 4E), 16 GB, macOS 24.6.0, rustc 1.97.0, `--release`, Python 3.12 in
`.venv` (open3d 0.19.0, numpy 2.5.2, scipy 1.18.1, trimesh 5.1.0, pygltflib 1.16.5). This is the
first step of the colour workflow and it closes audit §C.4's fourth bullet: *"colour is a constant
terracotta for both … item 4's colour/texture consensus is therefore **untestable on every set that
has object ids**"*.

**The four sentences.** *The photograph is now on the sherd*: `tools/make_synthetic.py --texture`
reads the base-colour texture out of the source GLB's own bytes, samples it at the closest point of
the scan surface under each fragment vertex, and paints the fragment's skin with it while the
fracture faces take the vessel's clay-body colour — painted outside, plain break, which is what a
sherd looks like. *Two coloured mixed collections exist*: `input/synthetic_mix3_24` (three decorated
vessels, 8 cells each, 24 fragments, 40 adjacent pairs) and `input/synthetic_mix3_60` (the same
three at 20 cells each, 58 fragments, 127 pairs), both with `object_of` and adjacency in the format
`tools/evaluate.py` reads, both reproducible from `sh tools/make_mix3.sh` and its committed seeds.
*Colour is measurable for the first time on a set with object ids, and it is strong*: the CIE76
distance between two fragments' mean Lab separates same-object from different-object pairs with
**AUC 0.996** on `synthetic_mix3_24`, against **≤ 0.74** for every geometric object feature the
cache carries (M1 §4) — and **nothing in the pipeline uses it yet**, which is what makes the
baseline below a baseline. *Nothing else moved*: without `--texture` the generator reproduces
`input/synthetic_pingsdorf_20` **byte for byte** — all 20 PLYs and `ground_truth.json` — the parity
total is the frozen 23 804 / 0, and the gate's forty old rows are identical cell by cell to the
previous step's.

---

## 1. What the generator learned

`--texture` is off by default. With it off, `tools/make_synthetic.py` executes the code it executed
before this step, draw for draw — that is the point of §7 — and every existing set regenerates
unchanged. With it on, three things happen that did not before.

### 1.1 The photograph comes out of the GLB, not out of trimesh

`_glb_base_colour(path)` reads the glTF with pygltflib and returns, per material, the image decoded
from the file's own buffer. Two of the five scans make the naive reading wrong, and each of them
breaks a different naive rule:

| model | what the file actually says | the rule it breaks |
|---|---|---|
| `025_zylinderhalsgefaess` | no `pbrMetallicRoughness` block at all; the photograph hangs off `KHR_materials_pbrSpecularGlossiness.diffuseTexture` | "read `baseColorTexture`" — the material appears not to reference its own texture |
| `094_bemalte_schuessel` | two textures: `baseColorTexture` → image 0 (the photograph), `metallicRoughnessTexture` → image 1 (a 1024² palette PNG that renders magenta) | "take the only image in the file" — it would paint the bowl magenta |

The other three (`012`, `049`, `074`) carry one material with a plain `baseColorTexture`.

trimesh resolves both cases too, and it was the obvious shortcut, but it **converts** the
specular-glossiness diffuse map on the way in: measured on `025`, trimesh's `baseColorTexture`
differs from the file's image bytes by up to **4/255** per channel on **83 %** of texels. The whole
premise of this task is that the fragment carries what the scanner wrote, so geometry and texture
coordinates come from trimesh (which puts them in the frame `load_solid` builds the solid in, node
transforms and all, verified equal to `Scene.to_geometry()`'s bounds) and the *pixels* come from
pygltflib. trimesh flips the v axis on load; `load_textured_surface` flips it back, so
`sample_texture` works in the convention the file is written in.

**Colour management: none, deliberately.** glTF declares a base-colour texture to be sRGB and a
renderer linearises it before lighting. We do not. A museum scanner writes sRGB bytes, the fragment
files are the scanner's output rather than a render of it, and R §3.1's reader
(`Features::with_colour`) treats a PLY's `uchar` colour as exactly those bytes. Sampling is bilinear
with REPEAT wrap, which is what all five samplers declare.

### 1.2 Where the skin ends and the break begins

`ShellPaint` builds one Open3D raycasting scene over the scaled scan surface and answers, per
fragment vertex, the distance to that surface and the photograph's colour at the closest point —
barycentric over the hit triangle, so the resolution is the texture's (≈ 0.4 mm/texel at these
scales) and not the scan's vertex spacing (≈ 0.9 mm). Skin and break are then separated by that
distance alone, which is a measurement and not a guess. Measured on a four-cell test break of
`049_kelch` at `--voxel 0.6`:

| fragment | vertices | distance to the scan surface, quantiles (voxels) | on the skin | on the break |
|---|---:|---|---:|---:|
| `frag_001` | 315 220 | q01 0.00, q10 0.01, q50 0.06, q90 0.20, q99 **5.11** | 94.9 % | 4.0 % |
| `frag_003` | 162 620 | q01 0.00, q10 0.01, q50 0.06, q90 0.57, q99 **6.76** | 90.9 % | 7.6 % |

The two populations are separated by nearly an order of magnitude: marching cubes and Taubin leave
the skin within **0.2 voxels** of the scan, and a fracture face is **5–7 voxels** away from it —
half a wall, which is what a fracture face is. `SHELL_BAND_VOX = (1.0, 2.0)` puts the smoothstep in
the empty middle. On the same two fragments the break's measured mean colour is (104, 62, 43) and
(108, 64, 45) sRGB against a clay body of (106, 63, 44), which is the mottling and nothing else; the
skin's is (144, 97, 73) and (135, 88, 67).

The share of a fragment that is skin — 91–95 % here — is geometry, not a choice: a sherd is a thin
plate and its break is a narrow band around the rim of it. On the 8-cell breaks of the shipped sets
the fracture share is larger, and the smaller the cell the larger it gets.

### 1.3 The clay body

**Rule chosen: `dark-quantile` — the mean of the darkest 10 % of the photograph's sampled vertex
colours, by CIE luminance.** It is recorded in each set's `ground_truth.json` as `body_colour` and
`body_colour_rule` and printed in the set's README, so a later measurement can always say which
constant it was scored against. `--body-colour median` and `--body-colour r,g,b` are the other two
forms.

The alternative on the table was the median, and the measurement decides against it:

| vessel | scan vertices | **dark-10 % (used)** | median | what the median is |
|---|---:|---|---|---|
| `012_verziertes_gefaess` | 385 928 | **(65, 55, 44)** | (129, 107, 84) | the stamped, partly burnished outside |
| `049_kelch` | 378 036 | **(106, 63, 44)** | (144, 97, 78) | the terra sigillata slip |
| `094_bemalte_schuessel` | 656 046 | **(75, 54, 36)** | (122, 94, 68) | paint and slip mixed |
| `074_tongefaess` | 659 643 | (74, 58, 36) | (199, 174, 141) | the pale Pingsdorf skin |
| `025_zylinderhalsgefaess` | 72 689 | (26, 23, 19) | (111, 103, 94) | — |

The median is the *surface* — the slip, the paint, the burnish — and a break shows none of those.
The dark tail is the fabric in shadow, and it lands where a fired earthenware body actually is: a
dark grey-brown, a red-brown and a mid brown for the three vessels of these sets. It is not a
perfect estimator (on `012` the tail also contains dark stamped decoration, and on `025` it is
pulled to (26, 23, 19), too dark to use as-is), which is why the rule is recorded rather than
assumed and why `--body-colour r,g,b` exists.

Two consequences worth stating before anyone measures colour again. The body colours of `V012` and
`V094` differ by ΔE ≈ 10 while `V049` is far from both, so a fracture-face colour comparison on
these sets is *not* trivially separable — which is the honest case. And the clay body is a
**constant per vessel plus the generator's existing mottling**, so a fracture-face colour match is
easier here than on real sherds, where the fabric varies within one pot; the note says so rather
than letting a later AUC be read as a promise.

---

## 2. The two collections

```
sh tools/make_mix3.sh 24     # -> input/synthetic_mix3_24
sh tools/make_mix3.sh 60     # -> input/synthetic_mix3_60
sh tools/make_mix3.sh        # both
```

`tools/make_mix3.sh` is the definition of the sets: three invocations of `make_synthetic.py` at
fixed seeds followed by `tools/merge_collections.py`. Wall thickness, voxel, target edge, wear,
missing fraction and pose spread are the pingsdorf sets' own (`--wall 8.0 --voxel 0.6 --edge 0.6
--wear 0.25 --missing 0.0 --spread 500`), so the fragments are the size and resolution the gates
already work with. Each vessel is scaled to the same 8 mm wall, which is what stops wall thickness
from being an object label.

| collection | vessel | source | cells | seed | files | adjacent pairs | generation |
|---|---|---|---:|---:|---:|---:|---:|
| `synthetic_mix3_24` | `V012` | 012 Verziertes Gefäß, zenodo.10311275 | 8 | 0 | 8 | 14 | 13 s |
| | `V049` | 049 Kelch, zenodo.10354385 | 8 | 1 | 8 | 13 | 106 s |
| | `V094` | 094 Bemalte Schüssel, zenodo.10330624 | 8 | 2 | 8 | 13 | 432 s |
| `synthetic_mix3_60` | `V012` | as above | 20 | 100 | 19 | 43 | 12 s |
| | `V049` | as above | 20 | 101 | 19 | 40 | 109 s |
| | `V094` | as above | 20 | 102 | 20 | 44 | 131 s |

Both are gitignored like every other collection under `input/`; the script, the generator and the
merge tool are what is committed. The `_60` set holds **58** files and not 60 because two Voronoi
cells came out below the generator's minimum area — the same convention the existing sets already
follow (`synthetic_pingsdorf_60` holds 56 files, `_170` holds 164): the name is the number of cells
requested. `094`'s 432 s in the first row is the one occupancy grid that had no voxel cache yet
(947 × 875 × 508, 18.0 M solid voxels, 19 s to compute and then cached).

Sizes and resolutions, against the sets they are meant to sit beside:

| set | fragments | objects | adjacent pairs | file size kB, min / median / max | total |
|---|---:|---:|---:|---|---:|
| `synthetic_mix3_24` | 24 | 3 | 40 | 331 / **12 973** / 93 960 | 462 MB |
| `synthetic_mix3_60` | 58 | 3 | 127 | 86 / **6 198** / 24 427 | 496 MB |
| `synthetic_pingsdorf_20` | 20 | 1 | 50 | 2 055 / **15 457** / 27 630 | 312 MB |
| `synthetic_pingsdorf_60` | 56 | 1 | 143 | 133 / **4 777** / 17 544 | 314 MB |

The medians sit inside the pingsdorf range; the tails are wider, because three vessels of very
different size are broken into the same number of cells each. `V012` is a 165 mm vessel and `V094` a
564 mm bowl once both are scaled to an 8 mm wall, so `V094`'s cells are the 94 MB tail and `V012`'s
the 331 kB one. **This is a property of these sets a later measurement has to respect: fragment
size correlates with object identity here**, and any object-separation number measured on them must
be checked against a feature that cannot see size before it is believed.

`ground_truth.json` follows `tools/stage_sfspp.py`'s convention for `mixed_ABG`/`mixed_all`
exactly: every vessel keeps the matrices of its own assembled frame, there is no frame shared
between vessels, and none is needed — `evaluate.py` buckets a cross-object join before it looks at
any matrix. Added beside them: `objects`, with each vessel's source, licence, author, fragment
count, adjacency count and clay-body colour.

### Pictures

| file | what it shows |
|---|---|
| `input/synthetic_mix3_24/preview_V012_fragments.png` | six sherds of the stamped vessel in their stored poses — the rouletted bands on the skin, plain clay on every break |
| `input/synthetic_mix3_24/preview_V049_fragments.png` | six sherds of the terra sigillata goblet — the turned rings and the white base disc |
| `input/synthetic_mix3_24/preview_V094_fragments.png` | six sherds of the painted bowl — the blue-green and cream painted decoration |
| `input/synthetic_mix3_24/preview_V0{12,49,94}_assembled.png` | each vessel put back together by its ground truth |
| `input/synthetic_mix3_60/preview_*.png` | the same six pictures for the 60-cell set |
| `output/s1/mix3_24/preview_segmentation.png` | what `segment` writes — see §3 |

---

## 3. `segment` on `synthetic_mix3_24`, and one thing it does not do

```
target/release/sherd-refit-rs segment input/synthetic_mix3_24/fragments --out output/s1/mix3_24 -v
```

**24 fragments, 0 failed, 10.09 s wall (109.52 s of work)**, 24 caches written, working meshes
8 596–200 000 faces from originals of 8 596–2 440 516, every one watertight. R §3.2's thickness
estimate spreads **5.87–16.91 mm** over the 24 fragments although every vessel was scaled to the same
8 mm wall — a reminder, on a set built so that thickness *cannot* be an object label, of how noisy
that feature is on cells this large. The coloured meshes go
through R §3.1–§3.7 without a complaint, and `Features::with_colour` picks the colours up: the run
of §4 reports a `lab_l` / `lab_a` / `lab_b` consensus per group, so the photograph reaches the Rust
side and is already in the cache.

**`preview_segmentation.png` does not show the texture, and this is not a bug in the sets.** That
preview is the tail of the reference's `segment_only`: it paints shell 0.8 grey and fracture red
(`pipeline.rs:1591`), which is what it is *for*, and the vertex colours never reach it — R §3.1
reduces them to four Lab fields at read time (`fragment/mod.rs:142`) and the working mesh, which is
what the preview samples, carries no colours at all. Showing the photograph there would be a Rust
change (carrying per-vertex colour through decimation into the preview), and this task was scoped to
add none. The pictures that do show colour are the generator's own, listed above.

A second, smaller thing the same picture shows: on a mixed collection whose vessels differ in size
by 3.4×, the preview's layout — each fragment offset by 1.3× *its own* x-extent, then all of it
fitted into one 1400 × 600 frame — renders 24 fragments too small to read. Both are worth a line in
whichever later task touches the previews; neither affects a number here.

---

## 4. The baseline, before colour is used anywhere

```
target/release/sherd-refit-rs run input/synthetic_mix3_24/fragments \
    --out output/s1/run_mix3_24 --backend cpu --seed 0 --no-preview --no-meshes -v
python tools/evaluate.py output/s1/run_mix3_24 input/synthetic_mix3_24
```

**Wall 31.5 s**, peak 1 941 MiB — preprocess 10.1 s, matching 17.5 s, tiers 2.5 s, assembly 0.01 s,
objects 0.00 s, refine 1.3 s. 24 fragments, 275 pairs (1 skipped by `--thick-ratio`), 505
candidates, 56 accepted; the tier puts **11** pairs confirmed, **7** probable, and the rest
rejected.

| | seed 0 |
|---|---|
| joins used | 11 |
| correct | **11** |
| wrong pose / non-adjacent / **cross-object** | 0 / 0 / **0** |
| precision | **1.000** |
| recall (of 40 adjacent pairs) | 0.275 |
| **fragment accuracy** | **13 / 24 = 54.2 %** |
| groups | 15, of which 4 hold two or more; largest 4 |
| **group purity** | **1.000** |

Per object at seed 0, and the same four numbers over the gate's five seeds:

| object | fragments | gt pairs | frag acc | recall | | seed | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| `V012` | 8 | 14 | 50.0 % | 0.286 | | confirmed, correct | 11 | 7 | 6 | 7 | 8 |
| `V049` | 8 | 13 | 75.0 % | 0.385 | | confirmed, **false** | **0** | **0** | **0** | **0** | **0** |
| `V094` | 8 | 13 | 37.5 % | 0.154 | | probable | 7 | 10 | 10 | 11 | 9 |
| | | | | | | correct, both bands | 18 | 17 | 16 | 18 | 17 |
| | | | | | | fragment accuracy | 54.2 % | 37.5 % | 37.5 % | 37.5 % | 45.8 % |
| | | | | | | cross-object / purity | 0 / 1.000 | 0 / 1.000 | 0 / 1.000 | 0 / 1.000 | 0 / 1.000 |

**Precision is 1.000 at every seed and no join crosses a vessel** — the set does not start out
broken, so a later rule has to be judged on what it does to recall. **Recall is where the room is:**
confirmed 0.150–0.275 of the 40 pairs, 0.400–0.450 with the probable band. `evaluate.py`'s
"missed true joins" list at seed 0 is the sharpest statement of it — nine of the top ten true pairs
the run did not use have a pose that is already *right*:

| pair | score | seam | tight A/B | pen | pose |
|---|---:|---:|---|---:|---|
| `V094_frag_000` – `V094_frag_007` | 110.7 | 145.3 | 0.76 / 0.81 | 0.0000 | 0.1°, 0.04 t — **ok** |
| `V094_frag_006` – `V094_frag_007` | 104.4 | 133.3 | 0.80 / 0.78 | 0.0000 | 0.0°, 0.01 t — **ok** |
| `V049_frag_001` – `V049_frag_005` | 56.7 | 80.7 | 0.72 / 0.70 | 0.0000 | 0.0°, 0.01 t — **ok** |
| `V049_frag_003` – `V049_frag_007` | 51.7 | 68.0 | 0.76 / 0.79 | 0.0000 | 0.2°, 0.02 t — **ok** |
| `V012_frag_001` – `V012_frag_003` | 12.1 | 20.0 | 0.60 / 0.65 | 0.0000 | 0.1°, 0.00 t — **ok** |

These are not matching failures; they are joins the assembly did not *use* (R §8's greedy pass, one
group per seed's worth of them) or the tier declined to confirm. The colour evidence of §5 has
nothing to fix in precision on this set and everything to offer to the recall column and to the
confidence that lets a probable join be confirmed — which is exactly the shape the workflow's goal
asks for.

---

## 5. Colour, measured for the first time on a collection with object ids

Reproducing `Features::with_colour` in Python (the same `srgb_to_lab`, `features.rs:518`) over the
24 fragments' vertex colours, then taking the CIE76 distance between two fragments' mean Lab:

| | pairs | ΔE76 min / median / max |
|---|---:|---|
| same object | 84 | 0.6 / **2.3** / 7.2 |
| different objects | 192 | **5.0** / 10.5 / 18.6 |
| | | **AUC 0.996** |

Per-fragment mean Lab, which is what the cache already stores:

| object | L | a | b |
|---|---|---|---|
| `V012` | 37.5 – 43.0 | 3.6 – 4.3 | 10.1 – 12.4 |
| `V049` | 40.7 – 45.0 | 13.5 – 17.7 | 16.1 – 21.9 |
| `V094` | 39.3 – 42.6 | 5.5 – 7.9 | 17.2 – 19.6 |

`a` alone separates `V049` from the other two completely; `b` separates `V012` from `V094`. Set
against audit §D.2's own bar — *"a feature whose measured AUC exceeds 0.8"* — and against M1 §4's
finding that **no** feature in the cache reaches 0.800 on any collection with real object ids
(thickness, shell radius and fracture roughness top out at 0.74 and cannot tell Pot_E from Pot_I),
colour is the first object feature on this project that clears the bar with room.

Four cautions belong beside that number, and they are why nothing is proposed here.

1. **The overlap is not empty.** The largest same-object distance (7.2) exceeds the smallest
   different-object one (5.0). A hard veto at any single threshold costs true pairs.
2. **The three vessels are unusually distinct.** They were chosen for their decoration. Two vessels
   of the same ware from the same kiln would not separate like this, and `mixed_all`'s Pot_E/Pot_I
   problem is exactly that case — with no colour at all, since the SfS++ OBJs carry none.
3. **The synthetic clay body is a constant.** §1.3 — real fabric varies within one pot.
4. **Fragment size correlates with object here** (§2), so any rule fitted on these sets has to be
   shown not to be reading size.

The acceptance twin `synthetic_mix3_60` was **not** measured, on purpose: audit §D.2's
"validation without leakage" puts the acceptance collection out of reach until the step that scores
it. Its colours are there when that step comes.

---

## 6. The gate script

`tools/quality_gate.py` gains one development set and one acceptance row.

* **`SETS` + `BANDS["synthetic_mix3_24"]`** — `input/synthetic_mix3_24/fragments`, scored against
  `input/synthetic_mix3_24`. It carries `mixed_ABG`'s prohibition and no more: `pure_tiers`, i.e.
  **cross-object joins 0 and group purity 1.000 in the confirmed tier at every seed**, beside
  `tier_verdict`'s zero-false-joins rule that applies to every set. There is **no `recall_floor`**:
  audit §E's twelve is a number measured on `mixed_ABG` before the tier existed, this set has no
  such measurement behind it, and a floor fitted to what the tree reaches today would make the set
  unimprovable — V8-D3's argument, applied to a new set instead of to the terracotta. Recall is
  reported per seed. `acc`/`prec` are `None`: R §13 never saw this collection.
* **`ACCEPTANCE["synthetic_mix3_60"]`** — above D §12's 27-fragment rule like the other three, so it
  is reachable only through `--score-only` and no ordinary run of the script can produce it. It
  inherits the acceptance `pure_tiers` row automatically.

The docstring's counts follow (nine development sets, eight of them with a `ground_truth.json`, four
acceptance collections).

---

## 7. Byte-reproducibility with the new behaviour off

**The generator.** `input/synthetic_pingsdorf_20` regenerated from the parameters its own README
records, without `--texture`:

```
python tools/make_synthetic.py input/source_models/074_tongefaess.glb --out <tmp> \
    --fragments 20 --seed 0 --wall 8.0 --voxel 0.6 --edge 0.6 --wear 0.25 --missing 0.0 \
    --spread 500 --no-preview --source-url … --license CC-BY-4.0 --author "…"
```

`ground_truth.json` **identical**, and all **20** `fragments/*.ply` **identical by MD5** — vertices,
faces and vertex colours, not merely the colours the brief asked to check (`frag_000`: 314 798
vertices, 154 distinct colours, mean (189, 110, 76) sRGB; `frag_007`: 260 011 vertices, 151 distinct,
mean (178, 104, 72); both equal element by element to the shipped files). This is structural rather
than lucky: `--texture` adds no rng draw, `colour_fragment` consumes the same `coherent_noise`
sequence in both modes, and the flat path's arithmetic is the expression it always was.

The one thing that is not byte-identical is the set's `README.md`, which gains a `colour` row saying
"flat terracotta" — that file was never reproducible anyway (it records the generation wall clock).

**The binary.** `git diff --stat HEAD -- crates Cargo.toml Cargo.lock rust-toolchain.toml
rustfmt.toml` is **empty**: this step changed no file the binary is built from, so
`target/release/sherd-refit-rs` is the previous step's binary. Measured rather than argued, at
seed 0, CPU, `--no-preview --no-meshes`, on `terracotta`, `pot_A`, `pot_H` and `synthetic_20`, two
independent trees compared file by file:

**55 files, 47 byte-identical, 8 exempt, 0 differing.** The eight are the four `report.json` and
four `report.md`, and their only differing leaves are under `timings` and `memory` (13, 13, 11, 13
leaves) — every decision field is equal. The hashes are in
`output/s1/byteid_hashes.txt`; every `.sherd` cache and every `transforms.json` is bit-identical.

---

## 8. Standing gates, on the final tree

| gate | result |
|---|---|
| `cargo fmt --check` | **pass** — no output |
| `cargo clippy --workspace --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo clippy -p sherd-cli --no-default-features --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo build -p sherd-cli --no-default-features` | **pass** |
| `cargo test --workspace` (debug) | **pass** — 19 targets, **444 passed, 0 failed** |
| `cargo test --workspace --release` | **pass** — 19 targets, **444 passed, 0 failed** |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, **256 rows** (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed** — the frozen total, unmoved |
| `pytest -q` | **pass** — **60 passed in 77.8 s** |
| `tools/quality_gate.py`, 9 sets × seeds 0–4 | **exit 1 on the terracotta's recall row alone** — the intended state since task G — **804.4 s (13.4 min)**, 45 runs, **169 correct confirmed joins, 0 false**, 17.4 % of the 970 adjacent pairs, 428 correct with the probable band (44.1 %) |
| byte identity, seed 0, four sets, `--texture` off | **pass** — 55 files, 47 byte-identical, 8 exempt (timings and memory only), **0 differing** (§7) |
| the gate's forty pre-existing rows vs the previous step | **pass** — `output/quality/quality.json` (task C8) vs this step's, wall clock excluded: **1 565 cells, 0 differing**, and all eight old verdicts identical |

The quality-gate table, per set over seeds 0–4 (`output/s1/quality/quality.md` has the full
45 rows):

| set | gate | confirmed correct, by seed | confirmed false | cross-object | purity | correct, both bands |
|---|---|---|---:|---:|---|---|
| `terracotta` | **FAIL** — recall row | 2 / 2 / 2 / 2 / 1 | 0 | – | – | 2 / 2 / 2 / 2 / 2 |
| `pot_A` | pass | 4 / 4 / 3 / 4 / 6 | 0 | 0 | 1.000 | 8 / 8 / 6 / 8 / 9 |
| `pot_B` | pass | 4 / 3 / 4 / 3 / 3 | 0 | 0 | 1.000 | 13 / 11 / 11 / 11 / 10 |
| `pot_C` | pass | 0 / 1 / 0 / 1 / 0 | 0 | 0 | 1.000 where a group has two | 2 / 2 / 1 / 2 / 2 |
| `pot_G` | pass | 0 / 0 / 0 / 0 / 0 | 0 | 0 | – (no group of two) | 0 / 0 / 0 / 0 / 0 |
| `pot_H` | pass | 0 / 1 / 0 / 0 / 0 | 0 | 0 | 1.000 where a group has two | 3 / 3 / 2 / 3 / 3 |
| `synthetic_20` | pass | 5 / 9 / 9 / 8 / 11 | 0 | 0 | 1.000 | 23 / 24 / 23 / 22 / 27 |
| `mixed_ABG` | pass | 8 / 7 / 7 / 7 / 9 | 0 | **0** | **1.000** | 21 / 19 / 17 / 19 / 19 |
| **`synthetic_mix3_24`** | **pass** | **11 / 7 / 6 / 7 / 8** | **0** | **0** | **1.000** | **18 / 17 / 16 / 18 / 17** |

Every column is read out of `output/s1/quality/quality.json`. The terracotta has no staged ground
truth, so its cross-object and purity cells are empty by construction and its "correct" is R §13's
two decisions; `pot_C`, `pot_G` and `pot_H` produce seeds with no group of two, where purity is
undefined rather than 1.000. Nothing moved on the eight old sets and the new one enters with zero
false confirmed joins at every seed.

---

## 9. Housekeeping

**Disk.** Started at 27 GiB free, never below 19 GiB, 19 GiB free now. Written and kept:
`input/synthetic_mix3_24` (462 MB) and `input/synthetic_mix3_60` (496 MB), both gitignored and
reproducible from `tools/make_mix3.sh`; two new occupancy caches beside the source models
(`.voxcache_012…`, `.voxcache_094…`); `output/s1/` (283 MB — the `segment` caches, the baseline run,
the four byte-identity trees and the quality gate). Deleted once measured: the per-vessel part
directories `input/.mix3_parts` (443 MB), the pingsdorf regeneration tree (312 MB), the four-cell
kelch test, the second byte-identity tree.

**Runs.** Nothing above 27 fragments was matched. The largest collection matched here is
`synthetic_mix3_24` (24 meshes), inside §4 and inside the standing gate. `synthetic_mix3_60` was
generated and never run.

**Untouched.** `sherd_refit/` (frozen), `crates/` and the whole Rust tree (no file changed), the
parity harness and its fixtures, `input/sfspp`, `input/test_fragments_1`,
`input/synthetic_pingsdorf_*`, `output/fixtures/`, `fixtures/`, `output/acceptance/`,
`output/quality/`. The branch was never switched, nothing was pushed, nothing was stashed.

**Commits.** `S1.1` the generator's texture path and the `synth` extra; `S1.2`
`tools/merge_collections.py` and `tools/make_mix3.sh`; `S1.3` the gate script's two rows; `S1.4`
this note.

## 10. What is open, for the step that uses colour

1. **Colour is in the cache and used by nothing.** `Features::with_colour` already stores `lab_mean`,
   `lab_spread`, `colour_points` and `colour_distinct`, and `objects.rs` already reports `lab_l`,
   `lab_a`, `lab_b` in a group consensus — reported and never acted on, because M1 §4 found no
   feature above AUC 0.8. §5 is the first feature above it. The decision the audit asks for (which
   features veto, which only report) can now be taken on a table that has a colour row.
2. **Colour along the seam is not measured at all.** §5 compares whole-fragment means, which is an
   object-level statement. A join-level one — the photograph's continuity *across* the seam, and the
   clay body's agreement on the two fracture faces — is what could move the recall column of §4, and
   it needs new evidence computed beside the frozen stages, not a new threshold on an old one.
3. **The sets' two known biases** (fragment size correlated with object; clay body constant per
   vessel) must be controlled for in whatever is fitted on them.
4. **Two preview matters** (§3), for whichever task next touches `render.rs`: the segmentation
   preview cannot show colour, and its layout does not survive a mixed collection of very different
   vessel sizes.
