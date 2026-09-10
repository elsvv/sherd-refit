# V10 — independent verification of the colour/recall cycle (S1, S2, S3, A2)

**Date:** 2026-09-11. **Tree:** branch `rust-core`, the 22 commits `ab25411`…`34ab5cd`
(`ff6a1fc..HEAD`), verified at `34ab5cd`. **Machine:** Apple M2 Pro, 10 cores (6P + 4E), 16 GB,
macOS 24.6.0, rustc 1.97.0, `--release`, Python 3.12 in `.venv` (open3d 0.19.0, numpy 2.5.2,
scipy 1.18.1, trimesh 5.1.0, pygltflib 1.16.5). **Nothing in this note is taken from the four
notes it checks**: every number below was produced by a run made here, and where a figure of
theirs is quoted it is quoted to be compared with mine. Working tree at the start: clean, no
uncommitted work to keep.

**`gates_ok = false`, and the reason is three sentences of prose and nothing else.** Every
measured item reproduces: the diff is additive to R's frozen stages, no decision reads ground
truth or an object id, the shipped thresholds are the ones the S2/S3 tables chose, both tables
re-derive **exactly** on runs I made, the generator is byte-reproducible down to its preview PNGs,
all eleven standing gates pass, the quality gate exits 0 at **225 correct confirmed joins and 0
false**, the two acceptance runs reproduce task A2's confirmed tier **field for field** inside
5 % of its wall clock, and a fresh `ff6a1fc` built outside the repository gives **35 135 of
35 187 output leaves identical** with the new behaviour switched off. What fails is brief item 7:
`README.md`'s museum section still describes the object-disagreement arm with task O1's numbers
under M1's rule, and two docstrings inside `tools/` still say the quality gate fails on the
terracotta and that M1's rule is the shipped one. Three edits, no arithmetic — but the item as
written is "README's museum section matches the tree", and it does not.

---

## 1. What was checked, and against what

| brief item | verdict |
|---|---|
| 1a additive to R §3–§9, §11 | **pass** — §2; parity re-run by me at **23 804 / 0** |
| 1b no decision reads ground truth or object ids | **pass** — §3 |
| 1c the shipped thresholds are the ones the tables chose | **pass** — §4.1 |
| 1d the S2 AUC table reproduces | **pass** — §4.2, every cell |
| 1e the S3 rule table reproduces (3 sets × 5 seeds) | **pass** — §4.3, every compared row |
| 1f the generator is byte-reproducible | **pass** — §5, 24/24 meshes, ground truth, six previews |
| 2 standing gates, both profiles, both `--no-default-features`, pytest | **pass** — §6 |
| 3 `tools/quality_gate.py` | **pass** — §7, exit 0, 0 false in all 45 runs, recall up on every set |
| 4 acceptance re-measured once | **pass** — §8, identical confirmed tier, wall −5.9 % and −4.7 % |
| 5 byte identity against a fresh `ff6a1fc` built outside the repo | **pass** — §9 |
| 6 the showcase opens | **pass, with one caveat** — §10 |
| 7 README's museum section matches the tree | **FAIL** — §11, defect V10-D1 |

---

## 2. Is every change additive to R's frozen stages?

`git diff ff6a1fc..HEAD --stat` touches five files under the frozen directories and **no others**;
`mesh/` and `refine/` are untouched:

| file | added | what it is |
|---|---:|---|
| `matching/pair.rs` | +199 | `WideRival`, `RivalSource`, `match_pair_wide`, `wide_rival` |
| `fragment/features.rs` | +302 | `ColourStats`, `RawColours`, `split_colour`, three tests |
| `fragment/mod.rs` | +12 | the one call site of `split_colour` |
| `fragment/cache.rs` | +9 | **a test fixture only** |
| `assembly/greedy.rs` | +1 | **a test fixture only** (`wide: None`) |

**The search is the search it was.** `Pair::match_pair` is now `self.match_pair_wide(engine, p,
keep, false)` (`matching/pair.rs:359-361`), and with `wide = false` the added expression is

```rust
let rival = (wide && candidates.iter().any(|c| c.accepted))
    .then(|| self.wide_rival(...)).flatten();
```

which is `None` before `candidates.truncate(keep)` and stamps nothing. `wide_rival` borrows
`&candidates` immutably, and the only computation it can start — `stage2_batch` — is deterministic
and draws no random number (`grep -n "rng\|Rng\|rand\|seed" matching/pair.rs` has exactly one hit,
`Probe::draw(..., p.seed)` at `pair.rs:160`, on the unchanged stage-1 path). The pipeline passes
`params.tiers.is_some()` (`pipeline.rs:1263`), so `--tiers off` neither computes the rival nor
carries the field. `crates/sherd-core/tests/slab_pair.rs:317` asserts the additivity pose for
pose, score for score, verdict for verdict; §9 measures it on four collections.

**The colour split is written once and read by nothing geometric.** `split_colour` is called in
`fragment/mod.rs:207-213`, after `segment_working_mesh` and before `breaklines_of`, and assigns
only `Features::{shell_colour, frac_colour}`. Nothing in R §3.5–§3.7, R §5 or R §6 reads either.

**The one arithmetic change outside the object layer is `Consensus::mads`**
(`objects.rs:442-445`): `self.mad > 0.0 && is_finite` became `scale = self.mad.max(feature.
mad_floor())`. `FeatureKey::mad_floor()` returns `0.0` for **every** geometric key
(`objects.rs:235-249`), and `x.max(0.0)` reproduces the old guard on every input including `NaN`
(Rust's `f64::max` returns the non-NaN side, so `NaN.max(0.0) = 0.0` and the test still refuses)
and `+inf` (`is_finite` still refuses). The geometric consensus is the test it was; only the nine
Lab channels see `COLOUR_MAD_FLOOR = 3.0`.

**`CACHE_VERSION` 6 → 7** (`lib.rs:150`), which refuses an older cache and recomputes — the
documented and measured consequence of §9's one differing byte.

**Parity, re-run by me** over the seven `output/fixtures` dumps plus the committed
`fixtures/slab/dump`, both modes, 16 runs, all exit 0:

**256 rows — 213 PASS, 43 SKIP, 0 FAIL — 23 804 checks, 0 failed.** The frozen total, unmoved.

The two lines the diff adds to `crates/sherd-parity` are `wide: None` struct initialisers.

---

## 3. Does any decision read ground truth or an object id?

**No.** `grep -rn "ground_truth\|object_of\|objects\.json"` over `crates/` returns nine hits: five
in `tests/slab_pair.rs` (a test fixture reads the slab's own truth), four in doc comments
(`measure.rs:80`, `fragment/segment.rs:19,79`, `matching/hypotheses.rs:235`). `crates/sherd-parity`
has none. `grep` for `Pot_`, `V012`, `V049`, `V094` and for name splitting in `sherd-core/src` and
`sherd-cli/src` returns three hits, all inside doc comments of `objects.rs`.

An "object" in `objects.rs` is an **assembled group** (`objects.rs:801-812`, `operator_view`,
`demotions` all iterate `assembly.groups`); nothing parses a file name. The scoring tools
(`evaluate.py`, `quality_gate.py`, `measure_tiers.py`) do read `ground_truth.json`, which is what
a scorer is.

---

## 4. The thresholds, the rule, and the two tables

### 4.1 The rule in the binary is the rule the S3 table chose

`Thresholds::S3` (`tiers.rs:473-484`) is `{ margin_rival: Wide, min_rival_t: RIVAL_FAR_T = 5.0,
min_research: 1, research_seeds: 2, ..Thresholds::M1 }`, and `Thresholds::default()` is `S3`.
`Thresholds::M1` (`tiers.rs:450-466`) is M1's strict half value for value: `tight ≥ 0.35`,
`gap ≤ 0.015 t`, `seam ≥ 5 t`, `cont_n ≥ 0.90`, `pen ≤ 0`, `slide ≤ 0.1 t`, `margin ≥ 2`,
`support ≥ 1`. `Thresholds::arm`/`by_margin` (`tiers.rs:563-583`) is

```
support ≥ 1  ∨  ( margin(wide) ≥ 2  ∧  wide rival ≥ 5 t away  ∧  research_agree ≥ 1 )
```

**M1 is restored exactly, not approximately**: `rival()` (`tiers.rs:1041-1063`) only ever returns a
placement strictly beyond `SAME_PLACEMENT_T = 1.0`, so under `RivalKind::Kept` the added conjunct
`d ≥ min_rival_t = 1.0` is vacuous, and `min_research = 0` makes the third vacuous too.

**Checked against a run rather than by reading.** Over the 39 confirmed candidates of the
committed showcase run (`output/showcase/synthetic_mix3_24/report.json`), I recomputed the rule
from each candidate's own evidence: **39 of 39 clear it, 39 of 39 carry the `arm` word my
recomputation predicts, and 0 carry a non-empty `failed`.** `params.tiers` in that file is
`Thresholds::S3` key for key, and the generated paragraph above `## Confirmed joins` states the
same rule in words.

`COLOUR_MAD_FLOOR = 3.0` is at `objects.rs:95` and applies to the nine Lab keys and to nothing
else. `FeatureKey::REPORTED` carries the three clay-body channels; `ObjectParams::demote` ships
empty.

The Python offline table agrees with the Rust term for term: `S3_STRICT`
(`tools/measure_tiers.py:1281-1282`) is M1's strict half, and the shipped arm in `s3_rules` is
`support ≥ 1 or (res_agree ≥ 1 and m_wide ≥ 2 and far(r, 5))`.

### 4.2 Task S2's AUC table, re-derived on `synthetic_mix3_24`

`python tools/measure_tiers.py --stage features --sets synthetic_mix3_24 --colour`, a fresh
`segment` pass, preprocessing only. **Every cell of S2 §2.1 and §2.2 reproduces:**

| feature | S2 §2.1 says | my run |
|---|---:|---:|
| `thick` | 0.516 | **0.516** |
| `shell_radius` | 0.777 | **0.777** |
| `frac_rough` | 0.597 | **0.597** |
| `axis_residual` | 0.770 | **0.770** |
| `lab_L` / `lab_a` / `lab_b` | 0.610 / 0.977 / 0.917 | **0.610 / 0.977 / 0.917** |
| `shell_lab_L` / `_a` / `_b` | 0.478 / 0.972 / 0.931 | **0.478 / 0.972 / 0.931** |
| `frac_lab_L` / **`frac_lab_a`** / `frac_lab_b` | 0.707 / **0.985** / 0.871 | **0.707 / 0.985 / 0.871** |

| pair distance | S2 §2.2 AUC / med same / med diff | my run |
|---|---|---|
| `frac_delta_e` | 0.935 / 3.83 / 11.94 | **0.935 / 3.827 / 11.94** |
| `shell_delta_e` | 0.954 / 2.23 / 11.52 | **0.954 / 2.227 / 11.52** |
| `frac_hist` | 0.974 / 0.263 / 0.837 | **0.974 / 0.263 / 0.8369** |
| **`shell_hist`** | **1.000** / 0.239 / 0.927 | **1.000 / 0.2385 / 0.9268** |

84 same-object and 192 different-object pairs, as S2 says. S2 §10's veto row also reproduces:
`shell_hist ≥ 0.5` removes **0.990** of the different-object pairs at a cost of **0.000** of the
adjacent ones.

**Task S1 §5's first colour AUC, re-derived independently** (whole-fragment `lab_mean`, CIE76,
computed in Python from the same feature dump rather than from S1's script):

| | pairs | ΔE76 min / median / max |
|---|---:|---|
| same object | 84 | 0.60 / **2.35** / 7.18 |
| different objects | 192 | **4.98** / 10.51 / 18.63 |
| | | **AUC 0.996** |

and the per-object Lab ranges are S1's to the tenth (`V012` a 3.6–4.3, `V049` a 13.5–17.7,
`V094` a 5.5–7.9).

### 4.3 Task S3's rule table, re-derived on 3 sets × 5 seeds

`--stage rule-runs --sets pot_H synthetic_20 synthetic_mix3_24 --seeds 0 1 2 3 4
--resample-seeds 2`, **15 runs of my own** (223.6 s of matching and 135.2 s of tier pass), then
`--stage rules`. The comparison column is task S3's own 45-run table restricted to the same three
sets, read out of `output/s3/measure/rules.json`'s `per_set` — so the two columns are the same
quantity on the same population.

| rule | mine: correct / **false** | S3's, restricted | where the false ones are |
|---|---:|---:|---|
| `support ≥ 1 OR margin(kept) ≥ 2` — M1's | 82 / **0** | 82 / 0 | — |
| `support ≥ 1` alone | 39 / **0** | 39 / 0 | — |
| `margin(kept) ≥ 2` alone | 67 / **0** | 67 / 0 | — |
| `margin(wide) ≥ 2` alone | 141 / **6** | 141 / 6 | `pot_H` ×6 |
| re-search agreed on 1 draw, alone | 151 / **12** | 151 / 12 | `pot_H` ×12 |
| re-search agreed on both draws, alone | 128 / **10** | 128 / 10 | `pot_H` ×10 |
| `support ∨ (wide ≥ 2 ∧ re-search 1)` | 136 / **5** | 136 / 5 | `pot_H` ×5 |
| `support ∨ (wide ≥ 2, rival ≥ 3 t, re-search 1)` | 130 / **1** | 130 / 1 | `pot_H` ×1 |
| `support ∨ (wide ≥ 2, rival ≥ 5 t)` — no re-search | 120 / **1** | 120 / 1 | `pot_H` ×1 |
| **`support ∨ (wide ≥ 2, rival ≥ 5 t, re-search 1)` — shipped** | **115 / 0** | 115 / 0 | — |
| `support ∨ (wide ≥ 2, rival ≥ 8 t, re-search 1)` | 107 / **0** | 107 / 0 | — |
| `support ∨ (wide ≥ 2, rival ≥ 5 t, re-search 2)` | 103 / **0** | 103 / 0 | — |
| `support ∨ (wide ≥ 1.2 / ≥ 3 / ≥ 5, rival ≥ 5 t, re-search 1)` | 115 / 115 / 113, all **0** | same | — |
| M1's rule ∨ the new arm | 123 / **0** | 123 / 0 | — |

**Sixteen rows compared, sixteen exact matches.** The two claims the rule rests on both reproduce:
the *cliff* — 5 false joins with no distance conjunct, 1 at three walls, **0** at five — and the
*plateau* — 1.2, 2 and 3 confirm the same 115, 5 confirms 113.

The rule's own per-set counts (`pot_H` 2, `synthetic_20` 59, `synthetic_mix3_24` 54 = 115) are the
same numbers my quality-gate run of §7 produced from the binary, which is an independent
cross-check of the offline table against the shipped implementation.

**The eight joins the shipped rule loses against M1's** come back seed for seed and pair for pair
as S3 §4.1 lists them: `synthetic_20` s0 `frag_003–frag_018`; s1 `frag_001–frag_006`,
`frag_005–frag_008`, `frag_007–frag_009`; s2 `frag_001–frag_006`; s3 `frag_003–frag_018`;
s4 `frag_001–frag_006`; and `synthetic_mix3_24` s0 `V094_frag_000–V094_frag_005`. Gained on the
same three sets: 41.

---

## 5. Is the generator byte-reproducible?

`input/synthetic_mix3_24` regenerated from the committed command — the three `make_synthetic.py`
invocations of `tools/make_mix3.sh` at seeds 0, 1, 2 with `--texture --body-colour dark-quantile`,
then `tools/merge_collections.py` — into a scratch directory outside the repository, and compared
with the shipped set:

| | result |
|---|---|
| `fragments/*.ply` | **24 of 24 byte-identical**, 0 differing, 0 missing |
| `ground_truth.json` | **byte-identical** |
| `preview_V0{12,49,94}_{fragments,assembled}.png` | **6 of 6 byte-identical** |
| `README.md` | **byte-identical** |

`ground_truth.json` carries 24 fragments in three objects, 40 adjacent pairs (14 / 13 / 13) and
the three clay bodies (65, 55, 44), (106, 63, 44), (75, 54, 36) — S1 §1.3 and §2 to the number.
Nothing under `input/` was written: the regeneration went to the scratchpad and was deleted after
it was compared.

---

## 6. Standing gates, run by me on `34ab5cd`

| gate | result |
|---|---|
| `cargo fmt --check` | **pass** — no output |
| `cargo clippy --workspace --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo clippy -p sherd-cli --no-default-features --all-targets -- -D warnings` | **pass** — 0 warnings |
| `cargo build -p sherd-cli --no-default-features` | **pass** |
| `cargo test --workspace` (debug) | **pass** — 19 targets, **453 passed, 0 failed**, 3 ignored |
| `cargo test --workspace --release` | **pass** — 19 targets, **453 passed, 0 failed**, 3 ignored |
| `parity --stage all`, both modes, eight dumps | **pass** — 16 runs exit 0, **256 rows** (213 PASS, 43 SKIP, **0 FAIL**), **23 804 checks, 0 failed** |
| `pytest -q` | **pass** — **60 passed in 126.1 s** |
| `tools/quality_gate.py`, 9 sets × seeds 0–4 | **exit 0** — §7 |
| byte identity, seed 0, four sets, `--tiers off --objects off` | **pass** — §9 |
| the acceptance runs | **pass** — §8 |

---

## 7. The quality gate, run by me

`python tools/quality_gate.py`, shipped defaults, nine development sets × seeds 0–4, 45 runs.
**Exit 0.** Wall **1 489.6 s (24.8 min)**, against task A2's 988.0 s — mine shared the machine
with the `cargo test`/parity stream, and the figure is reported rather than compared. On an idle
machine A2's 16.5 min is the number; both are inside D §10.4's 25-minute budget, mine by
**10.4 seconds**, which is worth knowing before anything else is added to the tier pass.

| set | correct confirmed, by seed | total | **false** | **cross-obj** | recall | ff6a1fc's recall (task C8's gate) | both bands |
|---|---|---:|---:|---:|---:|---:|---:|
| `terracotta` | 2 / 2 / 2 / 2 / 2 | **10** | **0** | 0 | **1.000** | 0.900 | 10 |
| `pot_A` | 6 / 5 / 4 / 4 / 7 | **26** | **0** | 0 | **0.347** | 0.280 | 39 |
| `pot_B` | 6 / 4 / 4 / 3 / 5 | **22** | **0** | 0 | **0.293** | 0.227 | 56 |
| `pot_C` | 1 / 1 / 0 / 1 / 1 | **4** | **0** | 0 | **0.160** | 0.080 | 9 |
| `pot_G` | 0 / 0 / 0 / 0 / 0 | **0** | **0** | 0 | 0.000 | 0.000 | 0 |
| `pot_H` | 0 / 1 / 0 / 1 / 0 | **2** | **0** | 0 | **0.024** | 0.012 | 14 |
| `synthetic_20` | 10 / 11 / 11 / 13 / 14 | **59** | **0** | 0 | **0.236** | 0.168 | 119 |
| `mixed_ABG` | 12 / 9 / 8 / 7 / 12 | **48** | **0** | **0** | **0.240** | 0.190 | 95 |
| `synthetic_mix3_24` | 12 / 10 / 7 / 12 / 13 | **54** | **0** | **0** | **0.270** | 0.195 (task S1) | 86 |
| **total** | | **225** | **0** | **0** | **0.232** | 0.169 / 0.174 | **428 (44.1 %)** |

* **Zero false confirmed joins in all 45 runs** — the hard gate, held on every set at every seed.
* **Recall is above the a1/g baseline on every set that has one and below it on none**: the
  right-hand column is `output/quality/quality.json` (task C8, the tree at `ff6a1fc`).
* **The terracotta's ten (seed, join) slots are met** — 2/2/2/2/2, which is what turns the gate's
  exit code from 1 to 0.
* Every cell is task S3's and task A2's, seed for seed. The gate is deterministic given a binary
  and a seed, and this is the third independent run to say the same thing.

---

## 8. The acceptance, re-measured once

Same command and flags as task A2, cold work directory, CPU, on an otherwise idle machine:

```
/usr/bin/time -l target/release/sherd-refit-rs run <input> --out output/v10/work/<set> \
    --backend cpu --seed 0 --no-preview --no-meshes --memory-budget 4.83 -v
```

| | `synthetic_mix3_60` seed 0 | | `mixed_all` seed 0 | |
|---|---:|---:|---:|---:|
| | **mine** | A2 | **mine** | A2 |
| real | **124.0 s** | 131.8 s | **2 419.3 s** | 2 539.1 s |
| Δ wall | **−5.9 %** | | **−4.7 %** | |
| preprocess / matching / tiers | 16.02 / 86.16 / 19.43 | 16.31 / 94.12 / 19.96 | 10.02 / 1 241.78 / 1 166.10 | 10.78 / 1 323.15 / 1 203.74 |
| candidates / accepted | 3 836 / **181** | – / 181 | 60 372 / **8 444** | 60 372 / 8 444 |
| peak RSS | 2 068 MiB | 1 746 MiB | 1 231 MiB | 1 406 MiB |

Both walls are inside the brief's 15 %, and both are *faster*, which is what an idle machine looks
like beside A2's (its note discloses that other work ran beside two of its stages). Peak RSS moved
in both directions and stayed inside the spread A2's own four `mix3_60` runs show (1 746–2 024 MiB).

**The confirmed tier is identical, field for field**, scored through `tools/quality_gate.py
--score-only`, which is the scorer A2 used:

| field | `synthetic_mix3_60` s0 | A2 | `mixed_all` s0 | A2 |
|---|---:|---:|---:|---:|
| confirmed | **38** | 38 | **37** | 37 |
| correct | **38** | 38 | **32** | 32 |
| wrong pose / non-adjacent / **cross-object** | 0 / 0 / **0** | 0 / 0 / 0 | 4 / 1 / **0** | 4 / 1 / 0 |
| confirmed recall | **0.299** | 0.299 | **0.111** | 0.111 |
| probable / correct in it | **23 / 23** | 23 / 23 | **2 747 / 48** | 2 747 / 48 |
| joins used / groups / largest | **38 / 25 / 11** | 38 / 25 / 11 | **35 / 129 / 6** | 35 / 129 / 6 |
| fragment accuracy / precision / purity | **69.0 % / 1.000 / 1.000** | same | **33.8 % / 0.857 / 1.000** | same |

The five false confirmed joins of `mixed_all` seed 0 come back **by name**:
`Pot_D_Piece_24–Pot_D_Piece_26` (non-adjacent), `Pot_E_Piece_13–Pot_E_Piece_27`,
`Pot_I_Piece_02–Pot_I_Piece_18`, `Pot_I_Piece_03–Pot_I_Piece_14`, `Pot_I_Piece_07–Pot_I_Piece_08`
(wrong pose). A2 §4's table is confirmed, including the sentence that matters most: **step 12's
prohibition is breached on `mixed_all` and the notes say so plainly.**

Both fragment caches were deleted once the runs were scored, as the brief asks.

---

## 9. Byte identity against a fresh `ff6a1fc` built outside the repository

`git archive ff6a1fc | tar -x` into the scratchpad, `cargo build --release` there (its own
`target/`, so nothing of this tree's build is shared), then both binaries run at seed 0, CPU,
`--no-preview --no-meshes --tiers off --objects off` on `terracotta`, `pot_A`, `pot_H` and
`synthetic_20`, into separate output trees.

**35 187 leaves of `report.json` and `transforms.json` compared, 52 differing, and every one of
the 52 falls into one of four buckets:**

| bucket | leaves | why it is exempt |
|---|---:|---|
| `engine.commit` | 8 | the binary changed (the archived tree stamps `unknown`) |
| `engine.cache_version` | 8 | 6 → 7, task S2's split (D §11, `lib.rs:150`) |
| `timings` | 16 | wall clock |
| `memory` | 20 | RSS |

**There is no fifth bucket.** Not one leaf under `/candidates`, `/joins_used`, `/joins_rejected`,
`/groups`, `/fragments`, `/thickness`, `/params` or anywhere in `transforms.json` outside `engine`
differs on any of the four sets. `report.json` with both switches off carries no `tiers` and no
`objects` key at all, so the two layers are absent rather than neutral. `report.md` differs in
**five lines over four files** and every one is a `- <stage>: N.N s` timing line.

**The caches say the same thing more sharply.** 43 of 43 `.sherd` files differ, as they must at a
new `CACHE_VERSION` — but `pot_A`, which carries no vertex colours, differs in **exactly one byte**
at the **same file size** (4 659 689 bytes both), which is the version field and nothing else;
the terracotta's grows by 736 bytes, which is the two `ColourStats` blocks task S2 added. The
geometry the cache carries is bit-identical wherever there is no colour to add.

---

## 10. The showcase

`output/showcase/synthetic_mix3_24/`, 563 MB, opens.

| | |
|---|---|
| `review/` | **18 PNGs** — 12 confirmed + 6 probable, which is the run's own band count |
| a caption, read off `V094_frag_000__V094_frag_007.png` | `A=… GREY \| B=… ORANGE \| PROBABLE \| ARM - \| MARGIN 6.15 \| SUPPORT 0 \| SEED 0` and a fourth line `COLOUR: CLAY BODY DE76 3.0 \| SKIN HISTOGRAM DISTANCE 0.18 \| NOT PART OF THE BAND` — task S2's line and task S3's `ARM` are both there |
| `report.md` | `## Objects` with the three clay-body channels, `## Confirmed joins` with the generated rule paragraph, the `margin (wide)`, `re-search`, `fracture dE`, `shell hist` and `arm` columns, 12 rows |
| `assembly_0..3.ply`, `placed/*.ply` | headers carry `property uchar red/green/blue` — **the photograph is in the meshes** |
| `preview_0..3.png` | open and render correctly, in R §11.5's **per-fragment identity colours** (grey / orange / blue / green) |

**The caveat, and it is task S1's own open item rather than a new one.** The run's PNG previews do
not show the photograph: R §11.5 paints fragment identity, and R §3.1 reduces the file's vertex
colours to four Lab fields before the working mesh exists (S1 §3). The photograph reaches a viewer
through the PLYs above and through the generator's own `input/synthetic_mix3_24/preview_*.png`.
And `preview_segmentation.png` is **unreadable on this collection**: 24 fragments of three vessels
that differ 3.4× in size, laid out across one 2800 × 600 frame, come out as a row of specks a few
pixels across. S1 §3 predicted exactly this and left it open; the showcase inherits it. Neither is
a claim any note makes and neither is a wrong number, so item 6 passes — but if the showcase is
what a museum is shown first, that segmentation preview is the one thing in it that does not work.

---

## 11. The documents against the tree

**D §10.3, §10.4, §11, §12 are correct.** The four-run acceptance table, the two gate margins
(6.8× and 2.8×), the `+39.6 %` and `tiers +107 %` decomposition, the 45-runs-in-16.5-min row, the
"exits 0" row with the terracotta at ten of ten, item 4's colour cell (`frac_lab_a` 0.985/0.990,
`CACHE_VERSION` 7, `COLOUR_MAD_FLOOR`) and step 12a all match what I measured.

**README's museum section is right except in one paragraph** — the tier section (225/0, 23.2 %,
the two-arm rule, `--tier-rule m1`, the `arm` column), the real-collection table (38–41 / 37–38,
13 false, 42.3 min), the colour section (S1's sets, 0.985/0.990, the three measured reasons colour
still refuses nothing) and the quality-gate section (nine sets, forty-five runs, exit 0,
ten of ten) are all the tree's. **Defect V10-D1 below is the exception.**

### Defects

| # | where | what it says | what the tree does |
|---|---|---|---|
| **V10-D1** | `README.md:647-650` | «в подтверждённой полосе **на восьми наборах** при сидах 0–4 ноль ложных стыков … включение снимает на `mixed_ABG` **27 подтверждённых стыков из 38** по пяти сидам (**8 / 7 / 7 / 7 / 9** становится 2 / 3 / 2 / 4 / 0)» | The gate is **nine** sets (task S1), and `mixed_ABG` confirms **12 / 9 / 8 / 7 / 12 = 48** under the shipped rule (§7, and task S3 §8 / A2 §8). 8/7/7/7/9 = 38 is task O1's measurement under **M1's** rule; the paragraph reads as current and is not attributed. The `--object-disagreement` cost itself has not been re-measured since the rule changed. |
| **V10-D2** | `tools/quality_gate.py:80-82`, `:228-235`, `:481` | "the gate asks for ten. **On this tree it reaches nine**; the run says which slot is short"; "the run **fails** on it, and the failure is printed with the slots that are missing"; "This tree reaches nine and the gate fails on it" | The tree reaches **ten** and the gate **exits 0** (§7). `TERRACOTTA_CONFIRMED_SLOTS = 2 * 5` and the code are right; three prose blocks inside the gate script were left behind when task S3 closed the row and D §10.4 and the README were updated. |
| **V10-D3** | `tools/measure_tiers.py:1428`, and `:1451`, `:1463`, `:1517-1521`, `:1666`, `:1686`, `:1709`, `:1720` | the rule named **`shipped`** is `support >= 1 OR margin(kept) >= 2`, and `rules.md` §3 prints "what it gains or loses against **the shipped rule**" against that baseline | Since `c0d9b16` (S3.4) the shipped rule is `Thresholds::S3`. The S3 note's own §4 table labels the row correctly ("M1's, the rule in force until this step"), so the note is right and the tool's label is not: a `--stage rules` run made today prints M1's rule under the word `shipped` and measures every other rule's cost against it. Numbers unaffected — only the label and the baseline's name. |
| **V10-D4** (minor) | `README.md:452` | the review-image caption is "оценки, полоса, отрыв от второй позы, сид" | the caption's first line now also carries `ARM` (task S3) and, on a coloured collection, a fourth `COLOUR:` line (task S2, §10). The colour half is documented two sections later; `ARM` is documented as a `report.md` column but not as a caption field. |

### One observation, not a defect

`WideRival::moved_t` is measured from the **pair's best** candidate (`matching/pair.rs:483`),
while `Probes::wide_margin` is each candidate's **own** `score / rival.score`
(`tiers.rs:1021-1022`). For a pair's non-best accepted candidate the margin arm's two halves are
therefore referenced to two different poses — unlike `rival()` (`tiers.rs:1055`), which measures
from the candidate being judged. It is documented where it is done (`pair.rs:637-640`) and it is
immaterial to every number in this cycle, because the scored population is
`representatives`, which takes the best candidate of a pair. It is written down so that a later
step reading a non-best candidate's evidence knows which pose the distance is from.

---

## 12. Housekeeping

**Disk.** 77 GiB free at the start, never below 74 GiB, 74 GiB now. Written and kept:
`output/v10/` (92 MB — the two acceptance reports without their caches, the quality-gate JSON and
Markdown, the 16 parity logs, the feature and rule dumps, the byte-identity reports, every run
log, and the seven scripts this note's numbers come from). Deleted once measured: the two
acceptance fragment caches, the four byte-identity caches, the regenerated
`synthetic_mix3_24` (442 MB) and its part directories, and the `ff6a1fc` build tree in the
scratchpad.

**Runs.** Development matching stayed inside the small-set rule: the quality gate's 45 runs, the
rule stage's 15, the byte-identity pass's 8 (four sets × two binaries), and one `segment`
pass on `synthetic_mix3_24`. The two collections above 27
fragments were matched **once each, at seed 0**, which is brief item 4 and nothing more.

**Untouched.** `sherd_refit/` (frozen — `git diff --name-only ff6a1fc..HEAD -- sherd_refit/` is
empty for the range under review and this task changed nothing anywhere but this note),
`input/`, `output/fixtures/`, `fixtures/`, `output/acceptance/`, `output/acceptance2/`,
`output/quality/`, `output/s1`–`s3`, `output/showcase/`. The branch was never switched, nothing
was pushed, nothing was stashed, nothing was sent anywhere.

**The range's own hygiene.** All 22 commits are authored `elsvv` and carry the
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer; each touches only files its
task names; **no file under `output/` or `input/` is committed anywhere in the range**
(`input/` is gitignored, and the two new collections are reproducible from `tools/make_mix3.sh`
— §5 shows they are reproducible byte for byte).

**Commit.** `V10` — this note, and nothing else.
