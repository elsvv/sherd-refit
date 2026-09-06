# Rust core — phase 0 and phase 1a plan

**Date:** 2026-09-06. Branch `rust-core`. Design: `docs/superpowers/specs/2026-09-06-rust-core-design.md`
(D), algorithm reference: `docs/superpowers/specs/2026-09-06-algorithm-reference.md` (R).
Decisions taken by the team: separate branch; gates measured on this M2 Pro (10 cores, 16 GB);
statistical parity for the working mesh (D §13.1); tolerance-based parity (D §13.2); the Rust
binary takes the name `sherd-refit` once phase 1 passes its gates, Python becomes
`sherd-refit-py` (D §13.6); prefer existing crates over own implementations wherever a crate
passes the experiment — own code only where no crate does (justify in the experiment note).

## Phase 0 — experiments and the parity harness (parallel)

| id | task | pass criterion | output |
|---|---|---|---|
| E1 | decimation: Open3D vs `meshopt` vs `baby_shark` on the 4 terracotta scans + pots A/B | D §3 E1 (face count ±5 %, closed_enough, res ±10 %, thickness ±2 %, segmentation agreement ≥ 0.97) | note `docs/superpowers/notes/2026-09-06-e1-decimation.md`, recommendation |
| E2 | mesh IO crates: `ply-rs` / `mesh-loader` / own PLY, `tobj`, `stl_io`, `gltf`; round-trip every benchmark file incl. colours; speed on 1.3 M-face PLY | counts/colours identical to Open3D's reader; < 2 s per 25 MB PLY | note `…-e2-io.md`, recommendation (own parser only if no crate passes) |
| E3/E4 | spatial: `parry3d` closest point / ray cast / inside vs Open3D `RaycastingScene` on terracotta samples; `kiddo` bounded NN vs a simple hash grid on `pc_reg` clouds | D §3 E3, E4 (|Δd| ≤ 1e-4 t; sign flips only at |d| < 1e-4 t) | note `…-e3e4-spatial.md`, recommendation |
| E7/E8 | `wgpu` on this Mac (Metal): adapter self-test, fixed-order reduction bit-parity with CPU (E7), a batched radius-NN micro-kernel timing; also lavapipe/software adapter availability | kernel runs; sum bit-identical; timing recorded | note `…-e7-wgpu.md` |
| P0 | parity harness in Python: `SHERD_REFIT_FIXTURES=DIR` sink dumping `.npy`/JSON at every stage boundary of D §10.1, `tools/dump_fixtures.py`, `tools/compare_fixtures.py` with D §10.2 tolerances; fixtures for terracotta, pots A/B/C/G/H, synthetic 20, slab (committed under `fixtures/slab/`) | two dumps from `9d4b9d3` byte-identical; compare tool passes on itself | Python commits + note `…-p0-fixtures.md` |

## Phase 1a — workspace and working mesh (sequential, each step commits)

| step | content | exit criterion |
|---|---|---|
| S1 | Cargo workspace per D §2 (crates: sherd-core, sherd-cli, sherd-parity; gpu/py later), pinned deps per phase-0 results, CI matrix (GitHub Actions: macOS arm64/x86_64, Windows, Ubuntu) building and running `cargo nextest`, README-dev section | `cargo build --workspace` and `cargo test` green locally |
| S2 | IO: PLY (binary/ASCII, vertex colours) / OBJ / STL / OFF / GLB readers, PLY writer (D §3, R output schema), cleaning (duplicate vertices, degenerate faces, unreferenced), largest component | E2 round-trip tests on the benchmark files pass |
| S3 | mesh ops: face geometry, edge adjacency, `closed_enough`, decimation (chosen crate) with the adaptive face budget (R), Taubin smoothing with Open3D's weights (R), working-mesh assembly | fixtures "working mesh" stage within D §10.2 native tolerances on all fixtures |
| S4 | thickness (histogram mode, R), fragment cache (safetensors, versioned), fixture reader (`npyz`), `sherd-refit parity --stage working-mesh` in sherd-cli, determinism test (two runs byte-identical) | `parity` reports pass for the working-mesh stage on every fixture |
| V | independent verification agent: rebuild from scratch, run all tests and the parity command on all fixtures, review the code against R for silent deviations | written report; gates green or a precise defect list |

Later phases (1b–1e, 2, 3) follow D §12 after this plan's V step passes.
