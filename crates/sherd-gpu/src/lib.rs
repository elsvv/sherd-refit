//! `sherd-gpu` — the wgpu executor of D §6, and everything that lets a kernel be trusted.
//!
//! # What this crate is, and what phase it is in
//!
//! D §6.1 puts four inner loops behind
//! [`Executor`](sherd_core::executor::Executor): the coarse breakline score, one rung of the ICP,
//! the bounded point-to-surface distance and the inside test. `sherd-core`'s `CpuExecutor` is the
//! reference implementation of all four. This crate is the second one.
//!
//! **Phase 2a (task G1) builds the trust, not the kernels.** What is here is the device
//! ([`device`]), the buffer and dispatch arithmetic ([`buffers`]), the fragment-slot LRU
//! ([`slots`]), the self-test of D §6.8 as E7 §8 corrects it ([`selftest`]) and a
//! [`GpuExecutor`] that **routes every method to the CPU executor** and says
//! so. The four kernels arrive in phases 2b and 2c; when they do, the self-test, the cross-check
//! harness (`sherd-refit-rs gpu-check`) and the slot table are already there to measure them.
//! Nothing here silently pretends to be a GPU result: [`selftest::SelfTest`] reports which kernels
//! actually ran on the device, and `gpu-check` marks a delegated row `delegated` rather than
//! printing a deviation of zero.
//!
//! # The two kernels that do run
//!
//! Both come from experiment E7 (`notes/2026-09-06-e7-wgpu.md`), which is why they are the ones
//! the self-test is built on: their expected answers are measured, not assumed.
//!
//! * **A fixed-order reduction of 1e7 `f32` terms** (E7 §3): 256 workgroups × 256 lanes, lane `l`
//!   accumulating `a[w·block + l], a[w·block + l + 256], …`, then a shared-memory tree
//!   256 → 128 → … → 1, then a second pass over the partials. E7 measured this **bit-identical**
//!   to a single-threaded Rust mirror, twice, and the self-test asserts exactly that: not a
//!   tolerance, the same 32 bits.
//! * **D §6.2's radius-bounded nearest neighbour over the hash grid** (E7 §5), one invocation per
//!   (pose, source point). E7 measured 9.0–9.5 ns/query saturated, 56 index disagreements in
//!   24.6 M queries and `max |Δd| = 2.4e-7` of a unit cloud. The self-test runs a smaller sweep of
//!   the same kernel and holds it to those numbers.
//!
//! # The arithmetic rules E7 leaves behind
//!
//! Metal compiles every shader with fast math on and wgpu does not turn it off (E7 §4). The team
//! decision is to accept it — not to patch or vendor `wgpu-hal` — which makes three rules binding
//! on every WGSL file under `src/kernels/`:
//!
//! 1. **Addition-only reductions, in a fixed order.** Those are bit-exact (E7 §3). Never
//!    `subgroupAdd`, never a floating-point atomic: neither has a defined order.
//! 2. **No `dot`, `length`, `distance` or `normalize` where parity matters.** `metal::dot` is a
//!    library function and stays a fused chain even when contraction is disabled (E7 §4.2); the
//!    kernels write the sums out by hand, and so does the CPU mirror in
//!    [`HashGrid`](sherd_core::spatial::grid::HashGrid).
//! 3. **No reliance on denormals.** Apple GPUs flush them in hardware and no compiler option
//!    changes that. Nothing in R produces `f32` denormals at these scales.
//!
//! Agreement is therefore gated on D §10.2's tolerances **plus identical decisions on the
//! development sets**, which is what `gpu-check` measures.

pub mod buffers;
pub mod coarse;
pub mod device;
pub mod executor;
pub mod icp;
pub mod pipeline;
pub mod selftest;
pub mod shader;
pub mod slots;

pub use device::{AdapterChoice, AdapterEntry, Gpu};
pub use executor::{GpuExecutor, MethodSnapshot, Stats};
pub use selftest::{Selection, SelfTest};

/// Everything that can stop the GPU path, with the message the CLI prints.
///
/// D §6.8 wants `--backend gpu` to fail loudly rather than fall back silently, and on macOS there
/// is no software adapter to fall back *to* for a second opinion (E7 §6), so every variant here
/// names what was tried.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// No adapter at all on Metal, Vulkan or DX12.
    #[error(
        "no GPU adapter found on Metal, Vulkan or DX12; on macOS there is no software adapter \
         either, so there is nothing to fall back to but the CPU"
    )]
    NoAdapter,

    /// `--gpu-adapter` named one that is not there.
    #[error("no adapter matches `{wanted}`; this machine offers:\n  {}", available.join("\n  "))]
    NoSuchAdapter {
        /// What the flag asked for.
        wanted: String,
        /// The adapters that do exist, as `info` lists them.
        available: Vec<String>,
    },

    /// The adapter exists but cannot run D §6's kernels.
    #[error("{adapter} is below the limits D §6 needs:\n  {}", unmet.join("\n  "))]
    Limits {
        /// The adapter that was tried.
        adapter: String,
        /// One line per requirement it failed.
        unmet: Vec<String>,
    },

    /// `request_device` refused.
    #[error("{adapter}: opening the device failed: {message}")]
    Device {
        /// The adapter that was tried.
        adapter: String,
        /// The driver's own message.
        message: String,
    },

    /// `device.poll` failed, which on a real device means it was lost.
    #[error("the device stopped responding: {0}")]
    Poll(String),

    /// A buffer could not be read back.
    #[error("reading {what} back from the device failed: {message}")]
    Readback {
        /// Which buffer.
        what: &'static str,
        /// The driver's own message.
        message: String,
    },

    /// A self-test check did not hold, which under D §6.8 means the CPU path.
    #[error("the GPU self-test failed on {adapter}:\n  {}", failures.join("\n  "))]
    SelfTest {
        /// The adapter that was tried.
        adapter: String,
        /// One line per check that failed.
        failures: Vec<String>,
    },
}
