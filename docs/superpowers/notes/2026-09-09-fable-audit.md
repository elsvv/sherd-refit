# Audit: an independent review of the port, its open problems, and the plan for roadmap items 3–4

**Date:** 2026-09-09. **Tree:** branch `rust-core` at `5e2b0e9` (W8), clean. **Auditor:** Fable, a
reading audit only — no build, no run, no experiment was made; every number below is quoted from a
note, a spec or the code, and every proposal is written for an Opus executor to carry out.
**Read:** the four specs (R, D, the 2026-09-05 design, the roadmap), all 45 notes, all of
`crates/` (`sherd-core`, `sherd-gpu`, `sherd-cli`, `sherd-parity`, their tests and examples), the
kernels, `tools/evaluate.py` and the parts of `sherd_refit/` and `tools/make_synthetic.py` that a
claim below rests on, and the wgpu-hal 30.0.1 Metal backend where W-D1 leads. Line numbers are as
read on this tree; a citation marked *(approx.)* is one I did not re-verify after reading.

---

## Executive summary

1. **W-D1 is most likely not in the kernel and not in the host pairing: it is that a command buffer
   the driver aborted is indistinguishable from one that completed.** wgpu-hal's Metal fence counts
   `MTLCommandBufferStatus::Error` as completion (`wgpu-hal-30.0.1/src/metal/mod.rs:1272-1284`),
   wgpu-core never surfaces it, the port decodes the staging buffer with no validation
   (`crates/sherd-gpu/src/icp.rs:394-440`) and its "device errors" counter increments only on a
   Rust-level `Err` (`crates/sherd-gpu/src/executor.rs:404-421`). A second process on the same
   adapter is exactly the condition under which macOS restarts the GPU and marks every in-flight
   command buffer of every process as an error. Every symptom W §6 records fits a readback of an
   unwritten or recycled `MAP_READ` buffer. Six sub-ten-minute experiments in §A.1 decide it; the
   mitigation in §A.1.5 — a nonce the kernel must write plus a validity check on every candidate,
   failure delegating the batch to the CPU and counting it — is worth shipping whatever the cause.
2. **The f32/f64 ladder divergence is a stopping-rule discontinuity, not a precision shortfall**, and
   production never sends stage 2 to the device (10 candidates < `MIN_CANDIDATES` 16). D §12's 2b
   criterion should be restated at the production policy with the translation row read at the
   cloud; then it is met on every set. Freeze the GPU path as an opt-in for discrete adapters,
   leave `Auto` on the CPU, and stop tuning on this integrated part (§A.2).
3. **The code is of high quality and over-documented; the real correctness risks are three**:
   unvalidated device readback (above), R §9's refinement loading every original scan in parallel
   with no memory semaphore (`crates/sherd-core/src/pipeline.rs:883-908`), and BVHs that are never
   evicted although D §8 says they live in an LRU (`crates/sherd-core/src/fragment/mod.rs:88-92`).
   Dead weight worth removing: `slots::SlotTable`, the E5 numerics instruments and their CLI flags,
   the `--dump-fixtures` stub, the `Auto` selection machinery (§B).
4. **The plan's two premises failed on measurement and the plan did not move**: the partner search
   (item 2) hit a geometric ceiling and all-pairs is the design; the port (item 7) was built before
   items 3–6, so every algorithm change now costs Python, Rust and fixtures. Flip the reference to
   Rust now. The dominant quality limiter — fracture-mask precision 0.50–0.63 on thin pots — has no
   roadmap item, and "zero false joins" is undefined until it is stated over the seed spread that
   R §13 measures (§C).
5. **Items 3 and 4 need a measurement before a design**: tiers defined by margin and stability under
   resampling, not thresholds alone; object features scored by their separation on the sets with
   object ids before any veto is written — thickness and shell radius are twins on four of the ten
   SfS++ pots, colour is absent from every benchmark that has object ids, so the benchmark cannot
   validate the roadmap's own feature list. The museum has to supply the acceptance material (§D).

---

## A. Root causes and fixes

### A.1 W-D1: the ICP answer is not a function of its input when a second process drives the adapter

#### A.1.1 What is known

From `notes/2026-09-08-w-phase2-findings.md` §6 and D §7's row: two of eight `gpu-check` runs on
`pot_H` printed `icp s2 deg` 4.081° instead of 1.466° with `icp s2 iter` 30 instead of 9; two
`run --backend gpu` of `synthetic_20` returned 376 and 375 candidates on one pair with byte-identical
caches; `examples/repeat_probe` answering one `IcpBatch` 64 times finds 0 of 64 differing on an
idle adapter, 0 of 64 under a ten-core CPU load, and 5–6 of 64 with another `run --backend gpu`
process alive. A differing repeat has *every* candidate moved, up to thirty iterations apart, with
"0 device errors, 0 delegated", and the answer is not the initial pose.

Two things about the reproduction the note does not say. The probe runs 64 point-to-plane repeats
*then* 64 point-to-point repeats (`crates/sherd-gpu/examples/repeat_probe.rs:98-105`), so the
table's "PointToPlane 0 of 64, PointToPoint 5–6 of 64" is confounded with *when* the other process
was in its matching stage; it does not establish that one kernel is affected and the other is not.
And the probe's only test of a bad answer is equality with the initial pose
(`repeat_probe.rs:122-134`); it never asks whether the words are zero, NaN, or a previous answer.

#### A.1.2 What reading rules out

I re-read both kernels against the list the brief gives, and agree with W that nothing in the port
explains it:

* **Workgroup memory.** `scratch`, `total`, `pose`, `done`, `applied`, `corres_count`, `error2`
  (`crates/sherd-gpu/src/kernels/icp.wgsl:103-113`) are written by lane 0 or inside `reduce8`
  between barriers, and every read of `total` in a pass happens before the first barrier of the
  next `reduce8` (`icp.wgsl:120-150`), which is the only writer. `load_state` initialises every
  scalar before the first barrier (`icp.wgsl:477-489`); `total` is read only after a `reduce8` has
  written the slots read (`point_to_plane` reads `total[27..28]` in `advance` after the fourth
  reduce, `icp.wgsl:707-709`; `point_to_point` reads `total[0..14]` and `[27]` after its own move
  block at `icp.wgsl:787-801`). No uninitialised read.
* **Barriers and uniformity.** Every `workgroupBarrier` is in uniform control flow: the loop bound is
  a uniform (`params.max_iter`), the early return is per workgroup, `reduce8` is called by every
  lane unconditionally, and `done` is read by all lanes only after the barrier that follows lane
  0's write (`icp.wgsl:640-647, 707-710`).
* **Storage buffers.** `corres[base + i]` is written and read by the same lane with the same stride
  (`icp.wgsl:538` against `657` and `771/817`), so device-memory visibility across lanes is never
  needed. Each candidate's slice of `corres` and `state` is disjoint by construction
  (`base = candidate * n_src`).
* **Host pairing.** One submission carries every chunk's dispatch and its copy into one staging
  buffer (`icp.rs:240-266`); `Submitter::submit` calls `map_async` after `queue.submit`
  (`crates/sherd-gpu/src/pipeline.rs:160-175`), `retire` polls that submission's own index
  (`pipeline.rs:177-186`) and `collect` waits for the map callback before reading
  (`pipeline.rs:189-207`). Every buffer the command buffer references lives until `submit_and_read`
  returns (`icp.rs:266-272`, `held` dropped after). wgpu-hal retains command-buffer references by
  default (`metal/mod.rs:390`), and `create_buffer_init`'s staging copies are the first command
  buffer of the same submit (wgpu-core `queue.rs`, `pending_writes`). The probe runs one job at a
  time, so `DEPTH = 4` is not involved.
* **Fast math.** Contraction changes results deterministically, not from run to run.

#### A.1.3 What reading rules in: the completion path cannot see an aborted command buffer

* `Fence::get_latest` in wgpu-hal's Metal backend treats `MTLCommandBufferStatus::Error` exactly as
  `Completed` (`wgpu-hal-30.0.1/src/metal/mod.rs:1272-1284`). wgpu-core's `maintain` then fires the
  `map_async` callback with `Ok`, and `device.poll(Wait { submission_index })` returns `Ok`.
* The port has no other channel. `GpuError::Readback` fires only if the map itself fails
  (`pipeline.rs:195-203`); `GpuError::Poll` only if `poll` errs. `Stats::errors` counts those and
  nothing else (`executor.rs:415-420`). "0 device errors" therefore says nothing about whether the
  GPU executed the command buffer.
* The staging buffer is created `MAP_READ | COPY_DST`, `mapped_at_creation: false`, never cleared
  (`crates/sherd-gpu/src/buffers.rs:136-143`). If the copy at the end of the command buffer never
  ran, the host reads whatever the allocation held: zero pages for a fresh allocation, or the
  previous buffer of the same size that Metal's allocator handed back. Every ICP call in a rung
  sequence allocates a staging buffer of identical size (`n × 16 × 4` bytes) and frees it just
  before the next call, which is the recycling case.
* `registration()` decodes the sixteen words with no check that they were written
  (`icp.rs:394-440`): a zero block decodes as rotation 0, translation `centre_t`, 0 iterations,
  not converged; a recycled block decodes as a perfectly plausible pose of the *previous* call.
* What aborts a command buffer without the port doing anything wrong: on macOS a GPU restart —
  triggered by *another* process's fault or watchdog timeout — terminates every command buffer in
  flight on the device with `status = Error` (the console message is "innocent victim of GPU
  restart"). A second process running big coarse dispatches on the same adapter is precisely the
  condition W measured; a ten-core CPU load is not.

The hypothesis against the symptoms: only with a second device user (yes); 0 device errors (yes,
by construction); every candidate moved (a whole readback is bad, yes); up to thirty iterations apart
(the synthetic batch has unconverged candidates at `max_iter = 30`; a zero `applied` word is thirty
apart, yes); not the initial pose (zeros and stale blocks are neither, yes); 4.081° with `iter` 30
on `pot_H` (a recycled block holding the *previous rung's* poses of the same ten candidates would
give a few degrees, where zeros would give about 75° — consistent, and it is the reading the
sentinel experiment decides); 376 against 375 candidates (one corrupted stage-1 rung changes the
second suppression's kept set, yes). This is ruled in, not proven; the experiments below are what
prove it.

#### A.1.4 Secondary hypotheses, kept until the experiments close them

* **Driver preemption save/restore.** Apple GPUs preempt compute at fine granularity when another
  client needs the device; a driver defect in restoring threadgroup memory or SIMD state would
  corrupt a reduction mid-kernel. It would show as arithmetic drift from one iteration on, not as a
  whole-block readback, and only the per-iteration trace (E6) separates it from A.1.3. Not fixable
  from user space; the mitigation of A.1.5 still catches its gross cases and the stability check of
  §D.1 catches its subtle ones.
* **Timing-dependent host bug.** None found; the probe's sequential calls leave nothing to race.

#### A.1.5 Experiments (each under ten minutes on this machine, Opus executor, no algorithm change)

Environment for all: the reproduction is `examples/repeat_probe` with a second process running
`run --backend gpu` on `synthetic_20`; nothing above 27 fragments; write outputs under
`output/audit/`.

* **E1 — sentinel fill (decides "unwritten" against "computed differently").** Create the staging
  buffer `mapped_at_creation: true` and fill it with `0xFFFF_FFFF` (a NaN) before submission, or
  `queue.write_buffer` the pattern; on a differing repeat print how many of the `n × 16` words are
  still the sentinel, how many are zero, how many finite. Any sentinel word proves the copy did not
  run. ~20 lines in `buffers::staging` plus a counter in the probe.
* **E2 — the console (decides "aborted").** Run the reproduction beside
  `log stream --predicate 'eventMessage CONTAINS[c] "GPU"'` (or `log show --last 10m` afterwards)
  and look for "GPU Restart", "IOAF code", "Execution of the command buffer was aborted". No code.
* **E3 — interleave the estimators.** Alternate point-to-plane and point-to-point per repeat and
  timestamp each call; decides whether the effect is kernel-specific or a function of when the
  other process was busy. Ten lines in the probe.
* **E4 — read the status (decides definitively).** A local `[patch.crates-io]` of wgpu-hal under
  `output/audit/` that logs `status` and `error` in the `addCompletedHandler` block
  (`metal/mod.rs:756`) and, if `Error`, makes `get_latest` *not* count the buffer. Ten minutes of
  patching; it is also the upstream report to file against wgpu (a failed command buffer must not
  resolve a fence as success).
* **E5 — the shape of the foreign load.** Second process = the probe itself (many tiny dispatches)
  against second process = `run --backend gpu` (12 M-query coarse dispatches). If only the latter
  triggers it, the mechanism is a watchdog or fault in the foreign process's long dispatches, which
  is the A.1.3 path.
* **E6 — a per-iteration trace (decides preemption drift).** A debug binding the kernel writes
  `(fitness, rmse, count)` into per candidate per iteration under a `Params.first`-style flag; diff
  a bad repeat against a good one and report the first divergent iteration and the size of the
  step. Garbage from iteration 0 is A.1.3; a one-ULP drift from iteration `k` is A.1.4.

#### A.1.6 The fix to ship whatever E1–E6 say

1. **A nonce in the state block.** The host writes a per-call nonce into word 15 of every candidate's
   initial state (it is `done`, currently uploaded as 0); `store_state` writes `f32(done) + nonce`
   or a separate word (`icp.wgsl:492-501`); `registration()` refuses a block whose nonce is not
   `nonce + 1` or whose rotation is not orthonormal to 1e-3, whose words are not finite, whose
   `iterations > max_iter`, or whose `count > n_src`. The same for `coarse.wgsl`'s counts
   (`agree ≤ points`). A refused block sends the *whole batch* to `CPU.icp_rung` and increments a
   new `MethodStats::corrupt`, printed in the run's device lines. Cost: one word and one branch per
   candidate.
2. **Make the invisible visible.** Rename `errors` to `host_errors` or document that a device-side
   abort is not counted; D §7's row should say so.
3. **Clear the staging buffer.** `mapped_at_creation: true` with a sentinel fill costs nothing at
   these sizes and turns "recycled block" into "detected block".
4. **Keep the operational rule** (a reproducible run has the adapter to itself) until E4 is in and
   upstream has an answer; add it to `info`'s output.

### A.2 The f32/f64 ladder divergence

#### A.2.1 The facts (W §2–3, D §12 rows "2b, task W")

At D §10.2's own tolerances and with the control-excused candidates counted: stage 1 inside the
rotation row on 7 of 8 sets and the translation row at the cloud on 8 of 8 (origin-referenced 5 of
8); stage 2 inside on 4 of 8 (rotation) and 1 of 8 (translation); double-single accumulation of the
27 partials and of the 6×6 moved none of the failing sets and cost up to 1.22× device time; the
`icp s2 iter` row differs by up to 30 between the two sides.

#### A.2.2 Why more precision cannot close it

Two discontinuities sit between the two implementations, and neither is a rounding of the sums:

* **The correspondence set.** `nearest_below` tests `d2 < r2` in `f32` on a transformed point that
  Metal has contracted into FMAs (`kernels/grid.wgsl`, E7 §4.2; `icp.wgsl:505-551`). A source point
  within an ULP of the radius flips in or out between the two sides, one row of the 6×6 changes,
  the step changes, and the next iteration's set changes with it. C2 §5 measured Open3D itself
  moving by 25–177° under a one-ULP change of the initial pose on such candidates.
* **The stopping rule.** `|Δfitness| < 1e-6 ∧ |Δrmse| < 1e-6` (`icp.wgsl:553-576`) is evaluated on
  `f32` sums whose own noise the kernel's comment puts at 1.6e-6, and the Kahan compensation that
  was meant to tighten it is folded away by fast math on Metal (`icp.wgsl:511-531`, W §1). The
  device rung stops at a different iteration; W §2.4 found the whole residue there.

Both are threshold comparisons of a discontinuous function of the input; a wider accumulator moves
the argument by parts in 1e-7 and leaves the comparison where it was.

#### A.2.3 The three questions the brief asks

* **A fixed iteration count** removes the second discontinuity only if *both* executors use it, which
  is an algorithm change to R §7 (a new PMC row) costing about 2× on rungs that converge early —
  stage 2 is 27 % of a pair's time (`notes/2026-09-07-e2-tuning.md` §9.1) — and it leaves the first
  discontinuity in place. Not worth it for a path production does not take.
* **An f64-emulated convergence test** would need f64-emulated sums under it; W's double-single
  experiment is that, and it moved nothing, because the operands differ by a correspondence, not by
  a rounding.
* **CPU fine rungs** are the production policy already: `Pair::stage2_batch` climbs
  `params.stage2 = 10` candidates (`crates/sherd-core/src/matching/pair.rs:256-330`), and
  `IcpKernel::run` returns the batch to the CPU below `MIN_CANDIDATES = 16` and `MIN_WORK = 100 000`
  (`icp.rs:71-79, 155`). Every stage-2 row of `gpu-check` is measured under `force_device`
  (`crates/sherd-cli/src/gpu.rs:591`), a path no run takes. The device does the coarse score and
  the two stage-1 breakline rungs (250 candidates), where the cloud-referenced rows pass on 8 of 8.
* **What others do.** Open3D's tensor (CUDA) ICP, PCL's GPU ICP and every published GPU ICP I know
  are tested against their CPU counterpart by tolerance on the final transform and on the decision,
  never bit for bit; Open3D's own legacy CPU ICP is thread-count dependent on the chaotic candidates
  (C2 §5: 10.4°, 22.1°, 11.4° between one and ten OpenMP threads). "Same decisions, poses within a
  tolerance at the cloud, chaotic candidates excused and counted" is the standard and is what this
  port already meets on every development set.

#### A.2.4 Recommendation

1. Restate D §12's 2b exit criterion and D §10.4 layer 3 at the production policy: `gpu-check`
   defaults to `--policy`; the kernel rows are coarse and stage 1; stage-2 rows print `cpu by
   policy`; the translation row is read at the cloud (D §10.2's origin form is a lever arm of
   130 units times the angle on these scans, `gpu.rs:999-1011`); the two run-level rows that
   already hold — identical used joins, groups and `evaluate.py` scores on both backends, final
   poses inside the `refine` row — are the gate. Under that statement the criterion is met.
2. Make "stage 2 on the CPU" an explicit policy in `sherd-gpu` rather than a consequence of two
   thresholds a future tuning could move.
3. Freeze the GPU path: `--backend gpu` remains available and documented as an opt-in for discrete
   adapters; `Auto` stays on the CPU; no further tuning on integrated parts; the vendor matrix waits
   for a machine that has one. The measured 1.07–1.43× on the matching stage
   (`crates/sherd-gpu/src/selftest.rs:72-78`) does not pay for another engineer-week, and the CPU
   projections meet every gate of D §10.3 with 4–7× to spare.

---

## B. Code quality, ranked by impact

The crates are well built: one arithmetic per stage, every deviation from R named and measured,
`unsafe` confined to two bytemuck derives (`crates/sherd-core/src/vec3.rs:12`), errors typed and
returned, panics guarded, results collected by index everywhere, and a test suite that mostly tests
claims rather than lines. What follows is what I would change, most important first.

1. **Device readback is trusted unconditionally** (`icp.rs:394-440`, and the coarse counts in
   `crates/sherd-gpu/src/coarse.rs` *(approx.)*). No finiteness, range or orthonormality check;
   with A.1.3 this is the one place a hardware or driver event becomes a wrong pose that the
   pipeline cannot tell from a right one. Fix: A.1.6. Test: a unit test that feeds `registration()`
   a zero block and a NaN block and asserts refusal.
2. **R §9's refinement loads every original scan of every grouped fragment in a `par_iter` with no
   memory reservation** (`crates/sherd-core/src/pipeline.rs:883-908`). Preprocessing and the placed
   writers both go through `MemorySemaphore` (`pipeline.rs:101-119`, `report.rs:798-808`); this
   stage does not. At 170 multi-million-face scans that is `threads × (vertices + normals + KD
   query)` of originals resident at once. One-hour fix: the same `acquire(reservation(scan_faces))`
   around `load_mesh` and `fracture_cloud`.
3. **BVHs are never evicted.** `Fragment::bvh_full` and `bvh_frac` are `OnceLock`s that live for the
   process (`fragment/mod.rs:88-92, 357-376`); `run_with` builds every full BVH eagerly before
   assembly (`pipeline.rs:447-449`). D §8 books the full BVH as "in the LRU"; there is no LRU. At
   200 000 faces a `parry3d` `TriMesh` with its tree is of the order of 10–20 MB, so 170 fragments
   hold 2–3 GB of BVHs beside everything else. Not fatal on 16 GB, unmeasured on 170, and the fix
   is small (drop the fracture scenes after matching, or an `Arc` LRU keyed by id). Measure the
   peak RSS in the final acceptance before deciding.
4. **The `Stats::errors` counter cannot see device-side failure** (`executor.rs:56-57, 415-420`) and
   was read as evidence in W §6. Rename or document; add the `corrupt` counter of A.1.6.
5. **Dead and single-purpose code to remove or fence.**
   * `crates/sherd-gpu/src/slots.rs` (315 lines with tests): `SlotTable` has no consumer
     (`grep SlotTable crates` finds only a doc reference in `device.rs:250`); the resident-set
     design of D §6.3 was never built — every kernel uploads per call. Remove, and strike D §6.3's
     slot table or mark it "not built".
   * The E5 instruments: `icp::Assembly::Centred`, `Precision::F32`, `IcpTarget::narrowed`, and the
     CLI flags `--icp-precision/--icp-assembly` (`crates/sherd-cli/src/main.rs:398-430`). They
     measured a question D §7 has answered; keep the enum behind a test if a future GPU experiment
     wants it, drop the flags.
   * `Kernel::dispatch` (`crates/sherd-gpu/src/shader.rs:146`) is used only by
     `examples/fastmath_probe.rs`; `types::Pose` (the `Isometry3` newtype, `types.rs:834-868`) is
     used by nothing but its own test — the pipeline's pose is `Matrix4` under the alias
     `icp::Pose`; `fixture.rs` is a twelve-line stub and `--dump-fixtures` is a flag that only
     fails (`main.rs:679-685`). Remove all three; the fixture writer is not needed once the
     reference flips (§C.1).
   * `GpuExecutor::AUTO_ELIGIBLE = false` with `Selection::decide`'s four-branch rule
     (`selftest.rs:325-364`): dead in practice. Keep `Selection`, replace the constant with a
     documented "CPU until a discrete adapter is measured".
   * `buffers::Chunking` is *not* dead: `coarse.rs:222-223` uses it.
6. **The parity harness (≈ 9 000 lines, 16 stages) is justified while the Python is the reference and
   not after.** Its cost is not the lines, it is the rows that have to be re-argued at every
   licensed deviation (`NATIVE_P99_T` and `NATIVE_KS` calibrated twice,
   `crates/sherd-parity/src/stages/breakline.rs:36-57`). After §C.1's flip it becomes a regression
   harness for the frozen stages: keep it green, stop extending it, and put the quality gates of
   items 3–4 in `tools/evaluate.py` (or a Rust twin) against ground truth instead.
7. **Analysis logic lives in the CLI crate.** `gpu.rs::check`, `compare_pair`, `determined_batch` and
   `Column` (`crates/sherd-cli/src/gpu.rs:414-537, 553-914, 1080-1216`, ≈ 700 lines) run only with
   a device and have no unit tests; move them into `sherd-gpu` as a `crosscheck` module with tests
   on the synthetic batches `tests/adapter.rs` already builds.
8. **`matching::cache::MatchCache`** (`crates/sherd-core/src/matching/cache.rs`, ≈ 300 lines with
   tests, `CAPACITY = 64`): measured at −0.9 % CPU on the largest set and nothing on the wall clock
   (`notes/2026-09-07-e2-tuning.md` §3), keyed by the fragment's *address* beside its id to survive
   `id = 0`. E2 kept it for item 5's sake. It is not wrong; it is 300 lines for one per cent, and its
   address key is the kind of thing that bites when fragments are cloned. Keep only if item 5 is
   scheduled; otherwise remove it and the `--workers` block schedule it was written for can stay.
9. **Four implementations of "the angle between two poses"** with two conventions: the trace form in
   `sherd-parity` (`stages/mod.rs:407-420`) and in `assembly/consistency.rs:495-498`, the Frobenius
   form in `cli/gpu.rs:982-997` and again in `tests/adapter.rs:425-426`. The parity one reports
   3.6e-2° for a pose against itself (`gpu.rs:967-980`). One `icp::pose_gap` with both forms, tested
   once.
10. **Duplicated helpers**: `centroid`/`shifted`/`narrow` in `sherd-gpu/src/icp.rs:340-392` beside
    `centroid_of`/`narrow_point` in `sherd-core/src/matching/icp.rs:483-520`; a second
    `pairwise_sum` in `parity/stages/breakline.rs:439-446` whose block rule (plain sum up to 128) is
    *not* the core's numpy transcription (`mesh/geometry.rs:107-138`) — harmless where it is used,
    misleading by its name. The `nearest_within`/`nearest_below` doc paragraphs in
    `spatial/kdtree.rs:125-141` and `175-181` are the same text twice.
11. **Comments longer than the code.** Many modules carry 20–60-line measurement narratives that
    restate a note and reference verification ids (`render.rs`, `verify.rs`, `types.rs`,
    `kdtree.rs`). They are accurate today and will rot at the first re-measurement. Keep the "why",
    move the tables to the notes, keep the note link.
12. **Small things.** `memory::physical_memory` spawns `sysctl` to avoid `unsafe`
    (`memory.rs:919-943`; a `sysinfo` dependency is cleaner). `Poses::get` copies a `Matrix3` per
    pose in the coarse inner loop (`executor/batch.rs:123-135` *(approx.)*) — measure before
    touching, the stage is 27 % of a pair. Vacuous tests worth deleting:
    `the_extension_is_the_designs` (`fragment/cache.rs:844`), `zero_threads_leaves_the_pool_alone`
    (`pipeline.rs:1191`), `the_constants_are_open3ds` (`mesh/taubin.rs:201`),
    `the_binary_is_named_for_the_transition` (`main.rs:993`). Missing tests: a refused device
    readback (after 1), `refine` under a memory budget (after 2), and a ground-truth gate in Rust —
    today only `tools/evaluate.py` scores a run and only the slab has a Rust ground-truth test
    (`tests/slab_pair.rs`).
13. **What is right and should stay as it is**: the executor boundary and its batches; `Submitter`
    and `Occupancy`; `MemorySemaphore` and its admission rule; the deterministic thickness estimator;
    the `apply_transform_fused` distinction; the near mask; the ground-truth tests on the slab; the
    cancellation contract and its test.

Not read before the cutoff: nothing in `crates/`; of the Python reference only `evaluate.py`,
`make_synthetic.py`'s colouring and `fragment.py`'s thickness and segmentation entry points.

---

## C. What the design and the plan got wrong or missed for the real target

The target (roadmap §"Operating conditions"): 170+ fragments, scans of millions of faces, several
objects mixed, pieces missing, no ground truth, sculptural terracotta *and* thin pots, museum
machines and users.

1. **The order of the roadmap was inverted by events and the plan did not react.** Item 2's partner
   search failed on a geometric ceiling — one seam is 5–22 % of a fragment's breakline and the best
   of thousands of poses reaches that on pairs that never touched (`notes/2026-09-06-scale-pairs.md`
   §4.4) — so all-pairs is the design and the 164-fragment collection is a 28-minute CPU run
   (D §10.3). Item 7 ("Rust/GPU deliberately last: it must port a settled algorithm") was executed
   before items 3–6 on an algorithm whose known blocker is unsettled. The consequence is
   structural: every algorithm change now has to be made in Python, dumped as fixtures, ported and
   re-gated. **Flip the reference to Rust now** (D §13 question 7 proposed the end of phase 3a):
   the Python stays frozen at `9d4b9d3` as the parity oracle for the stages it already gates, items
   3–4 are built in Rust with ground-truth gates, and phase 3a (pyo3) is dropped unless the museum
   asks for Python.
2. **The dominant quality limiter has no roadmap item.** Fracture-mask precision is 0.50–0.63 on the
   SfS++ pots (thin-walls note §8.1, `segment.rs:78-88`); pot G produces no candidate at all, pot H
   places 27–36 % of its fragments, pot C's decisions flip on the seed (R §13). Items 3–6 add
   evidence *after* the mask; none improves it. The museum's material is half thin pots. A
   segmentation item — a two-sided wall test, shell/fracture contrast in roughness and curvature,
   colour where the scanner gives it — belongs before or beside item 4, and its gate is
   `tools/eval_segmentation.py` on the SfS++ surface truth.
3. **"Zero false joins on all benchmarks" is not a testable statement as written.** R §13 shows the
   reference's *decisions* moving with the seed on three of seven sets (pot_C accuracy 50–75 %,
   precision 0.500–0.667; pot_H 27.3–36.4 %), and wrong-pose joins on adjacent pairs at every seed
   on pot_C, pot_G and pot_H — those slides of 0.67–0.97 t along the seam (sfspp note §4) are the
   false joins a conservator would see. The port is byte-deterministic per seed, which hides this;
   neither CLI has `--seed`. The tier definition of §D.1 has to be stated and gated over the seed
   sweep, and `--seed` has to exist.
4. **The benchmarks do not cover the target.** The museum contributed 4 pieces; the SfS++ pots are
   thin, from one lab, and carry no colour (the OBJs have bare `v x y z` lines); the synthetic sets
   come from one pot with one intruder vessel, colour is a constant terracotta for both
   (`tools/make_synthetic.py:340-355`), and the wear is "only an arris chip" (synthetic note
   §"Realism caveats"). Item 4's colour/texture consensus is therefore **untestable on every set
   that has object ids**, and rim/base evidence has no labels anywhere. The plan needs an acceptance
   set from the museum: 20–30 conservator-confirmed joins, a handful of confirmed non-joins, and
   fragments with known object attribution, from their own scans. Without it the tiers are
   calibrated on the wrong material.
5. **Groups never merge** (`assembly/greedy.rs:369-371`, `Rejection::MergesGroups`). On four pieces
   it never matters; on 170 the greedy pass seeds groups from the strongest joins and every later
   join between two groups is refused. Item 5 answers it with group-level matching; merging two
   groups through a confirmed join, with the penetration and consistency checks already in
   `try_place`, is a day's work and belongs with item 4.
6. **State the scaling law.** All-pairs at 0.52–0.87 core-s a pair and 6.4× on ten cores
   (`e2-tuning.md` §10): 170 fragments 17–28 min, 300 fragments ≈ 1–1.6 h, 500 ≈ 3–5 h. The
   museum's "170+" is inside the gate; the README should say where the edge is and that
   `--screen-top-k` is the knob that buys time for measured recall.
7. **The "not built" items, triaged against the target.** Needed: `--constraints`, `--review-images`
   (item 3), `--seed` (3 above), the readback validation (A.1.6), the refinement semaphore (B.2).
   Not needed: `--dump-fixtures` and `--inject-from` (after 1), the vendor matrix and the BVH
   kernels (after A.2.4), out-of-core decimation (D §13 question 8 — ask the museum for the face
   counts of their scans; at ≤ 10 M faces and 16 GB the semaphore already covers it), pyo3
   (after 1), `--export-glb` and the desktop app (phase 3, out of scope here).
8. **Domain pushback a museum team will raise.** (a) The most useful output for a bench is not the
   assembly but, per fragment, its best two or three candidate partners with pictures and numbers —
   item 3's probable list, which should be per fragment, not only per pair. (b) "Same object, not
   adjacent" is a statement the tool cannot make today; item 4's per-group consensus can, and it is
   worth more to a conservator than another join. (c) Rim diameter and profile are the classic
   object-separation evidence for wheel-thrown pots and are absent from the plan; fabric (colour,
   temper visible as fracture roughness) is the classic evidence for hand-built ware and is
   measurable on the fracture faces. (d) Worn and abraded edges are the norm in an excavated
   collection; `min_tight = 0.25` within `0.01 t` was calibrated on lab scans and is where recall
   will go on museum material — the probable tier exists for exactly this and must be honest about
   it. (e) `report.md`'s plain-language rejections are the right shape; keep them for the tiers.
9. **Byte-identity is a reproducibility property, not a stability property.** It is valuable and the
   run should record its seed; it must not be presented as "the tool always gives the same answer"
   when a one-ULP change of an initial pose moves a chaotic candidate by 25° (C2 §5).

---

## D. Roadmap items 3 and 4

### D.1 Item 3 — confidence tiers, review images, constraints

**Define a tier by evidence beyond the five scores.** R §6.5 is one threshold set; a second, stricter
set alone reproduces the pot_C situation (a candidate sitting on `min_tight` flips on nothing). Three
quantities are cheap and discriminative on what the notes already measured:

* **Margin.** The ratio of the best candidate's `seam · tight` to the second-best *placement* of the
  same pair (placements more than one wall apart, `candidates.rs`'s `SAME_PLACEMENT_T`). On the slab
  the join leads by a factor of four (`tests/slab_pair.rs:261-276`); the reference's own second
  placement of the slab against itself sits on `min_tight` at 88°. A wrong-pose join on an adjacent
  pair is typically a slide along the seam with a near-equal score at the true pose.
* **Stability under resampling.** Re-verify the accepted candidate on two independent redraws of
  `Pf`, `S` and the margin (two extra `SampleParams.seed`s, only for accepted candidates, so the
  cost is a few per cent of a pair): a confirmed join keeps `tight` above the strict threshold and
  `gap` under its limit on all three draws. This is the port's own seed spread made cheap, and it is
  what turns the byte-deterministic run into an honest confidence.
* **Determinedness and slide.** The `determined` probe already exists (`parity/stages/mod.rs:446-472`,
  `cli/gpu.rs:498-537`): a candidate whose last two rungs move under one-ULP perturbations is not
  confirmed. Add a slide probe: restart the last fracture rung from ±0.5 t along the breakline
  tangent and require convergence back within 0.1 t; two extra rungs, accepted candidates only.

**Confirmed** = strict thresholds ∧ margin ≥ m ∧ stable ∧ determined ∧ slide-stable; **probable** =
R §6.5 accepted and not confirmed; **rejected** = the rest, with the reason it has today. Strict
thresholds and `m` are chosen from the measurement of step E2 below, not now; candidates from the
notes are `min_tight 0.35`, `gap 0.02 t`, `seam 5 t`, `cont_n 0.9`, `pen 0`, `m = 2`.

**Define "zero false joins" so it can fail.** On the eight sets with ground truth (terracotta,
pot_A/B/C/G/H, synthetic_20, mixed_ABG) at seeds 0–4, every confirmed-tier join classified by
`tools/evaluate.py` must be `correct`; `wrong_pose`, `non_adjacent` and `cross_object` must be 0 in
the tier on all 40 runs; `unscorable` joins are listed. Report confirmed recall per set beside it
(expect 60–80 % of the true joins; that loss is the product). At final acceptance the same on
`mixed_all` and `synthetic_170` (intruders). The assembly is built from confirmed joins only;
probable joins are listed and rendered, never placed unless a constraint says so.

**Review images.** `render::render_pair(a, b, T)` on the existing splat renderer: three views (down
the seam's mean shell normal, and the two shells), A grey and B orange, the seam's `t/3` voxels
(the cell list `seam_score` already forms, `verify.rs:376-404`) drawn white, B's fracture samples
coloured by their distance to A's fracture surface (green under `tight`, yellow under the gap limit,
red beyond), a caption with the numbers, the tier and the seed. `review/<a>__<b>.png` for probable
*and* confirmed joins, deterministic (seeded sampler), and a per-fragment index page in `report.md`
listing its candidates best first. About 150 lines of renderer, one flag.

**`constraints.json`.** Version the file and validate names against the collection (an unknown name
is an error, not a skip). Semantics, on top of D §11's:

* `must_not_join: [[a, b]]` — the pair is removed before matching and any candidate for it is
  rejected in assembly, listed under "constraints honoured".
* `must_join: [{a, b, pose?}]` — without a pose: the pair is matched with the second-pass budget;
  its best *probable* candidate is promoted to confirmed and seeded first; if none exists the report
  says "unsatisfiable" and the run still succeeds. With a pose (a 4×4 from the review, or written by
  the desktop app): matching is skipped for the pair and the pose is a confirmed candidate.
* `same_object: [[a, b]]` and `different_object: [[a, b]]` — item 4's consensus and the group
  purity reporting read them; `different_object` also vetoes a join.
* A pair in two lists is an error; constraints never edit a score, only filter candidates and seed
  the assembly; the report lists every constraint and what it did.

### D.2 Item 4 — object separation

**Measure the features before writing a veto.** Per fragment, at preprocessing, stored in the cache
(`CACHE_VERSION` 6): wall thickness `t` (exists), shell radius from a sphere or quadric fit on the
outer-shell samples (`samples.s` with `surface_normals`, shell-labelled), fracture roughness (RMS of
`Pf` to local plane fits at `0.5 t`, a proxy for temper and fabric), Lab colour mean and spread
where vertex colours exist (the museum PLYs have them; the SfS++ OBJs do not; the synthetic sets
have one constant colour), rim flag and rim diameter where `thick_mode / thick > 1.15` already
flags a rim (`fragment/mod.rs:149-156`) and a circle fits the rim edge, and an axis-of-rotation fit
with its residual for fragments that are rotational (the 2026-09-05 design rejected SfS++'s axis
for the terracotta; in a mixed collection the pots are where it works). Then, on pots A–H and
`synthetic_170` (fitting) and `mixed_ABG` and `mixed_all` (test only, object ids used for scoring
alone): per feature, the separation between same-object and different-object pairs (AUC), the share
of pairs a `k·MAD` veto would remove, and the true adjacent pairs it would cost. Scale-pairs §4.3
already says what to expect from the first two: pots A, B, F and G are twins in both, and a 2×/3×
gate on thickness and radius removes 34 % of `mixed_all`'s pairs for 6 % of its adjacent ones. The
decision which features veto and which only report is made on that table.

**Cycle consistency as evidence, not only as a filter.** Today `try_place` checks the new fragment
against the direct alternatives (`greedy.rs:276-300`) and a loop-closing edge against the placed
poses (`greedy.rs:351-368`). Add: (a) a support count per placement — the number of independent
accepted joins that agree with it — which feeds the tier (two consistent paths confirm a join whose
own scores are probable); (b) when two accepted joins into one group disagree about a fragment,
demote both unless one is confirmed; (c) group merging through a confirmed join, with the penetration
test across both groups and consistency with every cross-group accepted join, replacing
`MergesGroups`.

**Per-group consensus.** `GroupFeatures` = median and MAD of each feature over members; a join whose
new fragment deviates by more than `k·MAD` on a feature whose measured AUC exceeds 0.8 is demoted
to probable, not rejected — the conservator decides, with the numbers in the review image. Every
group is reported as an object with its consensus, its rim diameter if any, and the fragments the
consensus rejects.

**Validation without leakage.** Object ids are read by `evaluate.py` only; thresholds are chosen on
the single-object sets and `synthetic_170`; `mixed_ABG` is the development test (baseline: port 3
cross-object joins at purity 0.864, reference 1 at 0.706, `notes/2026-09-08-z-phase1e-findings.md`
§6) and `mixed_all` the acceptance test; the gate is cross-object 0 in the confirmed tier with the
true-join recall reported beside it.

**Spikes worth a day each, measured against ground truth before any of them is adopted.**

1. *Seam-segment matching as a partner search.* The scale-pairs signatures described windows along
   the whole breakline and hit the coverage ceiling because a seam is a fraction of the crack line.
   Split each breakline at its corners (curvature maxima of the chain) into corner-to-corner
   segments; a shared seam is one such segment on both fragments, with equal length, mirrored
   dihedral profile and equal local thickness profile. Match segments, not points, and keep pairs
   with at least one segment match. The unit of comparison is the seam itself, so the 5–22 %
   ceiling does not apply; whether corners are detected consistently on both sides is the question
   the spike answers, on `mixed_all` with its adjacency list.
2. *Thickness profile along the seam as a post-ICP veto.* Both sides of a crack measure the same
   local wall; the per-point thickness is available from R §3.2's rays. A cheap, discriminative
   check item 4's per-group consensus does not contain.
3. *Progressive stage 2* (scale-pairs §8, item 2): probe the top stage-1 candidates first and skip
   the rest when the post-`pc_reg` tight estimate is hopeless; needs the distribution the note says
   is unmeasured. Buys time on the 12 000 pairs that have nothing to find.
4. *Rim-diameter consistency* for wheel-thrown pots, gated on the axis fit's residual.

---

## E. The amended plan

Rules that hold for every step: Opus executors; nothing above 27 fragments is *matched* before
step 10 (the small-set rule; the one stated exception is step 7's preprocessing-only feature pass,
which is linear in the fragment count); every step ends with `cargo test --workspace`, the two
`--no-default-features` commands, `cargo clippy --all-targets -- -D warnings`, and the byte-identity
check of the four development sets against the previous step where the algorithm did not change;
outputs under `output/`; one note per step in `docs/superpowers/notes/`. Hours are agent-hours for
one Opus executor including its own verification.

| # | step | brief | gate | hours |
|---|---|---|---|---|
| 1 | **W-D1: diagnose** | Run E1–E6 of §A.1.5 with the second process alive; record per experiment what it found; if E4's status read shows `Error`, file the wgpu issue with the patch | a note naming the cause, or the two experiments that rule the leading hypothesis out | 4 |
| 2 | **W-D1: mitigate** | A.1.6: nonce in the state block, validity check in `registration()` and the coarse decoder, whole-batch delegation on refusal, a `corrupt` counter in `MethodStats` and the device lines, sentinel-filled staging, `errors` renamed; D §7's row and `info` updated | unit tests for a zero, NaN and stale block; `repeat_probe` under contention reports 0 differing or the exact number delegated; CPU outputs byte-identical | 4 |
| 3 | **2b criterion at policy; freeze the GPU** | `gpu-check` defaults to `--policy`, stage-2 rows `cpu by policy`, translation at the cloud; stage-2-on-CPU an explicit policy constant; D §10.2, §10.4, §12 restated; `Auto` documented as CPU until a discrete adapter exists; vendor matrix and BVH kernels struck from D §12 | `gpu-check` exits 0 on all 8 development sets; both backends' run outputs byte-identical to W8's | 3 |
| 4 | **Scaling fixes** | B.2 (refinement under the semaphore), B.3 (fracture scenes dropped after matching, or an LRU; a peak-RSS log line per stage), `--seed` on `run` and `bench` (seeds every `Draw`; recorded in `engine`) | tests: `refine` under `Budget::bytes(1)` runs one scan at a time; seed 0 outputs byte-identical to before; seeds 1–4 run on the slab and terracotta | 4 |
| 5 | **Remove dead weight** | B.5 and B.9–10: `SlotTable`, the E5 flags, `types::Pose`, `fixture.rs` and `--dump-fixtures`, `Kernel::dispatch` into the probe, one `pose_gap`, one `centroid`, the duplicated docs; B.7 moves the cross-check into `sherd-gpu` with tests | suite green on all four platforms' commands; no output moves | 4 |
| 6 | **Flip the reference** | Decision recorded in D §13/§12 and README: from this step the Rust core is the algorithm; `sherd_refit/` frozen at `9d4b9d3` as the parity oracle for R's stages; parity harness frozen (green, not extended); quality gates for new work are `tools/evaluate.py` over the GT sets at seeds 0–4, run by a script committed under `tools/` | the script runs the eight development sets at five seeds on the CPU (< 25 min wall) and writes one table | 3 |
| 7 | **Measure before the tiers** | For every accepted candidate of the 40 runs of step 6: R §6's scores, margin to the second placement, `determined`, stability under two resamples, slide, support count, and `evaluate.py`'s class; one table, one note; the same pass (preprocessing only) computes §D.2's per-fragment features on pots A–H, `mixed_ABG`, `synthetic_170` and `mixed_all` and reports per-feature AUC same-vs-different object | the table exists; the strict thresholds, `m` and the feature shortlist are chosen on it and written down with their false-join margin | 8 |
| 8 | **Tiers** | `Tier` on `Candidate`; strict `Thresholds` in `Params`; the three probes of §D.1 on accepted candidates; assembly from confirmed only; `report.md` sections Confirmed / Probable / Rejected and the per-fragment candidate index; `transforms.json` carries the tier | zero false joins in the confirmed tier on 8 sets × 5 seeds; confirmed recall reported per set; terracotta's two joins confirmed; byte-identical outputs with `--tiers off` | 10 |
| 9 | **Review images and constraints** | `render_pair` and `--review-images`; `constraints.json` v1 with the semantics of §D.1 (`must_not_join`, `must_join` with optional pose, `same_object`, `different_object`), validation, report section | images deterministic between two runs; terracotta with `must_not_join [021, 094]` yields 094–104 only and reports it; `must_join [007, 021]` reports unsatisfiable; a pinned pose places without matching | 8 |
| 10 | **Object separation** | `Features` in the cache (`CACHE_VERSION` 6), `GroupFeatures` consensus, demotion on `k·MAD` for the shortlisted features, support counts and mutual-disagreement demotion, group merging through a confirmed join, groups reported as objects; `same_object`/`different_object` honoured | `mixed_ABG`: cross-object joins 0 in the confirmed tier, purity 1.000, correct joins ≥ 12 (the baseline's) at seeds 0–4; no change of used joins on the seven single-object sets or an explained one; pot_H's largest group not smaller | 12 |
| 11 | **Spike: seam segments** (optional, after 10) | §D.2 spike 1 on `mixed_all`'s adjacency with preprocessing only plus segment matching (no pair matching): recall of adjacent pairs at K = 5/10/15 against scale-pairs §4.2's 0.36/0.47/0.53 | a note with the numbers; adopted only if recall at K = 10 exceeds 0.8 | 6 |
| 12 | **Final acceptance** | On the CPU, warm cache, previews off: `synthetic_60`, `synthetic_170`, `mixed_all` at seeds 0 and 1; `evaluate.py` per run; peak RSS per stage; the projections of D §10.3 replaced by measurements; the museum's acceptance set (§C.4) run the same way when it arrives | wall inside D §10.3's CPU gates; cross-object 0 and purity 1.000 in the confirmed tier; recall and the probable list reported; a README section for museum users on reading the tiers | 6 |

Total ≈ 72 agent-hours; steps 1–5 are independent of 6–12 and can run first; 7 must precede 8 and
10; 11 is optional; 12 is the only step that touches a collection above 27 fragments.
