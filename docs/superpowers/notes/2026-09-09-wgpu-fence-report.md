# An aborted Metal command buffer resolves its fence as success (wgpu 30.0.1)

**Written for wgpu, not filed.** This is the upstream report the audit's experiment E4 asks for
(`notes/2026-09-09-fable-audit.md` §A.1.5). It is a file in this repository and nothing more; the
decision to open an issue against `gfx-rs/wgpu`, and the account it would be opened from, are the
user's. Everything below was measured on 2026-09-09 by task H1
(`notes/2026-09-09-h1-wd1.md`); the reproduction and the diagnostic patch are in that note and in
`output/h1/wgpu-hal-e4.patch`.

---

## Summary

On the Metal backend a command buffer that the driver aborts —
`MTLCommandBufferStatus::Error` — is indistinguishable, to every wgpu API a caller has, from one
that completed. `Device::poll(PollType::Wait { .. })` returns `Ok`, the `map_async` callback fires
with `Ok(())`, and the mapped range holds whatever the allocation held before. A compute pipeline
that reads its results back therefore silently consumes an unwritten buffer.

Two separate things produce this, and either alone is enough:

1. **`Fence::get_latest` counts `Error` as completion.**
   `wgpu-hal-30.0.1/src/metal/mod.rs:1272-1284`:

   ```rust
   for &(value, ref cmd_buf) in pending_command_buffers.iter() {
       match cmd_buf.status() {
           MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error => {
               max_value = value;
           }
           _ => {}
       }
   }
   ```

2. **The fence signal is attached to only the last command buffer of a submission.**
   `Queue::submit` (`metal/mod.rs:737-770`) takes `command_buffers.last()`, relabels it
   `"(wgpu internal) Signal"` and calls `addCompletedHandler` on it; the other command buffers of
   the same submission are committed with no handler. A submission built by wgpu-core from one
   `CommandEncoder` with one compute pass and one `copy_buffer_to_buffer` is **five** command
   buffers on this machine, and the compute pass is number 3 of the 5. When the driver aborts the
   compute pass, the buffer carrying the fence completes normally and the fence's own view of the
   world is correct as far as it goes — it simply never looks at the one that failed.

wgpu-core does not inspect command-buffer status anywhere, so neither defect surfaces as an error
to the caller.

## Environment

* wgpu / wgpu-core / wgpu-hal **30.0.1** (exact, from crates.io), `objc2-metal` as that version
  pins it.
* Apple M2 Pro, 16-core integrated GPU, macOS 24.6.0 (Darwin), one Metal adapter, no software
  fallback.
* rustc 1.97.0, release build.

## Reproduction

Any two processes doing sustained compute on the one adapter. In our case:

* process A: a probe that submits one compute dispatch of 64 workgroups × 256 lanes, 30 internal
  iterations over 4 000 points per workgroup, then copies a 4 KB result buffer to a `MAP_READ`
  staging buffer, and repeats that 64 times;
* process B: the same binary, or our application's own GPU run.

Over one such contended pair of runs the driver aborted **29** of A's command buffers, and over
every contended run of the investigation, ninety. Every one of them was logged by Metal to the
system console as

```
(Metal) Execution of the command buffer was aborted due to an error during execution.
Internal Error (0000000e:Internal Error)
```

and **not one** of them produced an error from wgpu: `poll(Wait)` returned `Ok`, `map_async`
returned `Ok(())`, and the staging buffer's contents were the bytes it had held before the
submission.

## The diagnostic patch

`output/h1/wgpu-hal-e4.patch` (against `wgpu-hal-30.0.1/src/metal/mod.rs`) does two things:

* adds an `addCompletedHandler` to **every** command buffer of a submission, printing the label,
  the index within the submission, the status and the `NSError` when the status is not
  `Completed`;
* prints in `Fence::get_latest` when it resolves an `Error` buffer.

With it, the aborts read

```
[E4] command buffer 3 of 5 ("icp point", fence value 267) ended with status
     MTLCommandBufferStatus(5): Internal Error (0000000e:Internal Error)
```

and `get_latest`'s own print never fires, because the buffer it is watching (index 4) completed.
Attaching the handler to the last buffer only is therefore the load-bearing half of the defect on
this workload; the `Error`-counts-as-`Completed` arm is the other half, and would fire whenever
the failing buffer *is* the last one.

## How a caller can tell today

It cannot, through the wgpu API. Our own mitigation is entirely in the application: every readback
buffer is created `mapped_at_creation` and filled with a sentinel, every dispatch carries a
per-call nonce the kernel must give back incremented, and the decoder refuses a readback that
fails either check and recomputes the batch on the CPU. That is a workaround for one application's
kernels; it is not something every wgpu user can be expected to invent.

## Suggested fixes

1. `Fence::get_latest` should not treat `MTLCommandBufferStatus::Error` as completion. It has to
   advance the fence — a caller waiting on it must not hang — but the error should be recorded
   and returned, most naturally as `DeviceError::Lost` or a new `DeviceError::CommandBufferFailed`
   carrying the `NSError` description, from the next `poll`.
2. The completion handler should observe **every** command buffer of a submission, not only the
   one that signals. A cheap form: keep a per-submission `AtomicBool` and have each buffer's
   handler set it when its status is not `Completed`; the fence reads it when it resolves.
3. Failing both, a `Queue::on_submitted_work_done`-style path that can report failure, and
   documentation on `map_async` and `poll` stating explicitly that `Ok` today means "the host did
   not fail" rather than "the device ran".

## Prior art in the tracker

Not searched. Whoever files this should check for existing issues about
`MTLCommandBufferStatus::Error`, GPU restarts and "innocent victim" aborts before opening a new
one, and quote the two source locations above.
