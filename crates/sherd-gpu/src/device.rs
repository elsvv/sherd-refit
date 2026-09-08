//! Adapter enumeration, selection and the device the kernels run on (D §6.8).
//!
//! Three things happen here and nowhere else: the list of adapters `sherd-refit-rs info` prints,
//! the rule that turns `--gpu-adapter NAME|INDEX` into one of them (D §9), and the limits check
//! that decides whether a device can run D §6's kernels at all.
//!
//! # What E7 found on this machine, and what the code does about it
//!
//! * **One Metal adapter, and no software fallback of any kind** (E7 §6): `enumerate_adapters`
//!   returns a single entry and `force_fallback_adapter: true` fails with "metal had no fallback
//!   adapters". So a self-test failure on macOS means the CPU path with no second opinion —
//!   [`Selection::reason`](crate::selftest::Selection::reason) says so out loud rather than
//!   leaving the operator to infer it.
//! * **Metal reports no vendor, device id, driver or driver_info** (E7 §7.6), so `--gpu-adapter
//!   NAME` can only match on the adapter name and a run report cannot record a driver version.
//! * **The adapter's own limits are accepted verbatim** (E7 §2), and they are far above the wgpu
//!   defaults here (32 KB of workgroup storage against 16, a 4 GiB binding cap against 128 MB).
//!   The device therefore *requests* what the adapter offers — nothing is gained by asking for
//!   less — while the kernels are written to the **defaults**, because that is what a portable
//!   build gets. [`Requirements`] is the floor both must clear.

use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, DeviceType, Instance, InstanceDescriptor, Limits,
    Queue,
};

use crate::GpuError;

/// The workgroup size every kernel of D §6.4 is written for.
pub const WORKGROUP: u32 = 256;

/// `max_compute_workgroups_per_dimension`, which is 65 535 both on this adapter and in the wgpu
/// defaults (E7 §2) — a hard wall, not a Metal quirk.
pub const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// The default `max_storage_buffer_binding_size`: 128 MB, the cap D §6.3 sizes batches to.
pub const DEFAULT_BINDING_CAP: u64 = 128 * 1024 * 1024;

/// The default `max_compute_workgroup_storage_size`: 16 KB, which is what D §6.8 tells the ICP
/// reduction to fit into — Metal's own 32 KB is not portable.
pub const DEFAULT_WORKGROUP_STORAGE: u32 = 16 * 1024;

/// One adapter as `info` lists it and as `--gpu-adapter` names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterEntry {
    /// Its position in the enumeration — what `--gpu-adapter 0` selects.
    pub index: usize,
    /// `Metal`, `Vulkan`, `Dx12`.
    pub backend: String,
    /// The adapter's name; on Metal this is the only field that carries information.
    pub name: String,
    /// `IntegratedGpu`, `DiscreteGpu`, `Cpu`, `VirtualGpu`, `Other`.
    pub device_type: String,
    /// The PCI vendor id, `0` on Metal.
    pub vendor: u32,
    /// The PCI device id, `0` on Metal.
    pub device: u32,
    /// The driver name, empty on Metal.
    pub driver: String,
    /// The driver version string, empty on Metal.
    pub driver_info: String,
}

impl AdapterEntry {
    /// Reads one adapter's `AdapterInfo`.
    fn of(index: usize, adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        Self {
            index,
            backend: format!("{:?}", info.backend),
            name: info.name,
            device_type: format!("{:?}", info.device_type),
            vendor: info.vendor,
            device: info.device,
            driver: info.driver,
            driver_info: info.driver_info,
        }
    }

    /// Whether this adapter is the one `--gpu-adapter NAME` asks for: a case-insensitive substring
    /// of the name or of the backend.
    #[must_use]
    pub fn matches(&self, needle: &str) -> bool {
        let needle = needle.to_lowercase();
        self.name.to_lowercase().contains(&needle) || self.backend.to_lowercase().contains(&needle)
    }

    /// Whether this is a CPU implementation of the API (lavapipe, WARP, SwiftShader).
    ///
    /// Those are the adapters D §10.5's `gpu-software` job runs the kernel cross-checks on, and
    /// the ones `Backend::Auto` must never prefer to the real CPU path.
    #[must_use]
    pub fn is_software(&self) -> bool {
        self.device_type == "Cpu"
    }
}

impl fmt::Display for AdapterEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} {} ({})", self.index, self.backend, self.name, self.device_type)?;
        if !self.driver.is_empty() || !self.driver_info.is_empty() {
            write!(f, " driver={} {}", self.driver, self.driver_info)?;
        }
        Ok(())
    }
}

/// What `--gpu-adapter` was given (D §9).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum AdapterChoice {
    /// No flag: the first adapter the instance offers, which is wgpu's `HighPerformance` pick.
    #[default]
    Default,
    /// `--gpu-adapter 2`.
    Index(usize),
    /// `--gpu-adapter "M2 Pro"`, matched as a case-insensitive substring of the name or backend.
    Name(String),
}

impl AdapterChoice {
    /// Parses the flag: all-digits is an index, anything else a name.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let text = text.trim();
        if text.is_empty() {
            return Self::Default;
        }
        text.parse::<usize>().map_or_else(|_| Self::Name(text.to_owned()), Self::Index)
    }

    /// Picks the adapter out of an enumeration, or says why it cannot.
    ///
    /// Kept apart from the wgpu call so that the rule is testable without an adapter.
    pub fn pick(&self, entries: &[AdapterEntry]) -> Result<usize, GpuError> {
        if entries.is_empty() {
            return Err(GpuError::NoAdapter);
        }
        match self {
            Self::Default => Ok(0),
            Self::Index(i) => {
                (*i < entries.len()).then_some(*i).ok_or_else(|| GpuError::NoSuchAdapter {
                    wanted: i.to_string(),
                    available: entries.iter().map(ToString::to_string).collect(),
                })
            }
            Self::Name(name) => {
                entries.iter().position(|entry| entry.matches(name)).ok_or_else(|| {
                    GpuError::NoSuchAdapter {
                        wanted: name.clone(),
                        available: entries.iter().map(ToString::to_string).collect(),
                    }
                })
            }
        }
    }
}

/// The floor D §6's kernels need from a device, and what a given [`Limits`] gives.
///
/// Every number is a wgpu *default*, because a portable build gets the defaults; an adapter that
/// offers more (this one offers 32 KB of workgroup storage and a 4 GiB binding) is welcome to, and
/// the kernels still do not use it (E7 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirements {
    /// Invocations in one workgroup: D §6.4's ICP rung uses 256.
    pub invocations_per_workgroup: u32,
    /// The x extent of a workgroup, same number.
    pub workgroup_size_x: u32,
    /// Workgroup storage: D §6.8 designs the reduction for 16 KB.
    pub workgroup_storage: u32,
    /// The storage-buffer binding cap batches are sized to.
    pub storage_binding: u64,
    /// Storage buffers one dispatch binds: grid header, slots, counts, indices, points, poses,
    /// output — seven, inside the default of eight.
    pub storage_buffers: u32,
}

impl Default for Requirements {
    fn default() -> Self {
        Self {
            invocations_per_workgroup: WORKGROUP,
            workgroup_size_x: WORKGROUP,
            workgroup_storage: DEFAULT_WORKGROUP_STORAGE,
            storage_binding: DEFAULT_BINDING_CAP,
            storage_buffers: 8,
        }
    }
}

impl Requirements {
    /// Every requirement this `limits` fails, as a human-readable list; empty means it passes.
    #[must_use]
    pub fn unmet(&self, limits: &Limits) -> Vec<String> {
        let mut out = Vec::new();
        let mut check = |name: &str, have: u64, want: u64| {
            if have < want {
                out.push(format!("{name}: {have} < {want}"));
            }
        };
        check(
            "max_compute_invocations_per_workgroup",
            u64::from(limits.max_compute_invocations_per_workgroup),
            u64::from(self.invocations_per_workgroup),
        );
        check(
            "max_compute_workgroup_size_x",
            u64::from(limits.max_compute_workgroup_size_x),
            u64::from(self.workgroup_size_x),
        );
        check(
            "max_compute_workgroup_storage_size",
            u64::from(limits.max_compute_workgroup_storage_size),
            u64::from(self.workgroup_storage),
        );
        check(
            "max_storage_buffer_binding_size",
            limits.max_storage_buffer_binding_size,
            self.storage_binding,
        );
        check(
            "max_storage_buffers_per_shader_stage",
            u64::from(limits.max_storage_buffers_per_shader_stage),
            u64::from(self.storage_buffers),
        );
        check(
            "max_compute_workgroups_per_dimension",
            u64::from(limits.max_compute_workgroups_per_dimension),
            1,
        );
        out
    }
}

/// D §1's device-memory ceiling: slots plus batches ≤ 1 GB.
pub const DEFAULT_MEMORY_BUDGET: u64 = 1024 * 1024 * 1024;

/// What the kernels have allocated on the device, and the ceiling they may not cross (D §6.8's
/// "slots + batches ≤ 1 GB", `--gpu-memory`).
///
/// A batch is *reserved before it is uploaded*: the two matching kernels add up the buffers they
/// are about to create, ask for that many bytes, and — if the budget cannot hold them — answer the
/// batch on the CPU instead. A refusal is a delegation, counted like every other, never a failure
/// and never an allocation that is made anyway.
///
/// The rule is [`SlotTable`](crate::slots::SlotTable)'s, one level down: a request larger than the
/// whole budget is refused rather than admitted after everything else has been given up. What is
/// different is that there is nothing to evict — a batch's buffers live exactly as long as the
/// call that made them — so the reservation is an RAII guard and the "eviction" is the guard going
/// out of scope.
#[derive(Debug)]
pub struct Allocations {
    live: AtomicU64,
    peak: AtomicU64,
    refused: AtomicU64,
    budget: AtomicU64,
}

impl Default for Allocations {
    fn default() -> Self {
        Self {
            live: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            budget: AtomicU64::new(DEFAULT_MEMORY_BUDGET),
        }
    }
}

impl Allocations {
    /// Sets the ceiling; `0` removes it, as `--memory-budget 0` does for the host (D §9).
    pub fn set_budget(&self, bytes: u64) {
        self.budget.store(if bytes == 0 { u64::MAX } else { bytes }, Ordering::Relaxed);
    }

    /// The ceiling in force.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.budget.load(Ordering::Relaxed)
    }

    /// Bytes reserved right now.
    #[must_use]
    pub fn live(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }

    /// The most that were ever reserved at once — what a run reports and what D §1's 1 GB row is
    /// checked against.
    #[must_use]
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// How many batches the budget sent to the CPU.
    #[must_use]
    pub fn refused(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    /// Reserves `bytes`, or `None` when they would cross the ceiling.
    ///
    /// The compare-and-swap is what makes it right under D §6.4's pool: several workers form
    /// batches at once and the budget is the device's, not a thread's.
    pub fn reserve(&self, bytes: u64) -> Option<Reservation<'_>> {
        let budget = self.budget();
        let mut live = self.live.load(Ordering::Relaxed);
        loop {
            let wanted = live.saturating_add(bytes);
            if wanted > budget {
                self.refused.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            match self.live.compare_exchange_weak(
                live,
                wanted,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(wanted, Ordering::Relaxed);
                    return Some(Reservation { owner: self, bytes });
                }
                Err(now) => live = now,
            }
        }
    }
}

/// Bytes held on the device for as long as this value lives.
#[derive(Debug)]
pub struct Reservation<'a> {
    owner: &'a Allocations,
    bytes: u64,
}

impl Reservation<'_> {
    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.owner.live.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// How much of the wall clock the device had work of its own to do (D §6.4's "GPU busy
/// fraction", measured rather than inferred).
///
/// The counter that used to answer this question was `MethodStats::gpu_nanos` — the sum, over
/// every worker thread, of `submit` + `poll(Wait)`. On one thread that is the device's own time;
/// on nine it is not, because a thread that submits behind eight others waits for all nine and
/// reports the total as its own. Measured on `synthetic_20`: 4.33 s of coarse score on one thread
/// against **27.2 s** summed over nine, for the same 1.65 G queries.
///
/// This is the other measurement, and it is the one D §6.4 asks for. A submission is *outstanding*
/// from the moment it is handed to the queue to the moment its `poll` returns, and the device has
/// something to run for as long as at least one submission is outstanding. So the busy wall is the
/// union of the intervals rather than their sum: the clock starts when the count goes 0 → 1 and
/// stops when it comes back to 0, and nine threads waiting on one submission are counted once.
///
/// It is an upper bound on device execution — a submission can sit in the queue behind another —
/// and it is an honest lower bound on *idleness*, which is what the scheduler question is about.
#[derive(Debug, Default)]
pub struct Occupancy {
    state: Mutex<InFlight>,
    busy_nanos: AtomicU64,
    submissions: AtomicU64,
    peak: AtomicUsize,
}

/// The outstanding submissions and when the oldest of them started.
#[derive(Debug, Default)]
struct InFlight {
    count: usize,
    since: Option<Instant>,
}

impl Occupancy {
    /// A submission has been handed to the queue.
    pub fn entered(&self) {
        let Ok(mut state) = self.state.lock() else { return };
        if state.count == 0 {
            state.since = Some(Instant::now());
        }
        state.count += 1;
        self.submissions.fetch_add(1, Ordering::Relaxed);
        self.peak.fetch_max(state.count, Ordering::Relaxed);
    }

    /// A submission has finished.
    pub fn left(&self) {
        let Ok(mut state) = self.state.lock() else { return };
        state.count = state.count.saturating_sub(1);
        if state.count == 0
            && let Some(since) = state.since.take()
        {
            let nanos = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.busy_nanos.fetch_add(nanos, Ordering::Relaxed);
        }
    }

    /// The wall time during which at least one submission was outstanding.
    #[must_use]
    pub fn busy(&self) -> Duration {
        Duration::from_nanos(self.busy_nanos.load(Ordering::Relaxed))
    }

    /// How many submissions have been made.
    #[must_use]
    pub fn submissions(&self) -> u64 {
        self.submissions.load(Ordering::Relaxed)
    }

    /// The most submissions that were ever outstanding at once — how deep the queue in front of
    /// the device actually got.
    #[must_use]
    pub fn peak_in_flight(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Forgets everything, so that a stage can be measured on its own.
    pub fn reset(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.count = 0;
            state.since = None;
        }
        self.busy_nanos.store(0, Ordering::Relaxed);
        self.submissions.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
    }
}

/// An open device: the adapter that was chosen, the limits it granted, and the queue.
pub struct Gpu {
    adapter: Adapter,
    entry: AdapterEntry,
    device: Device,
    queue: Queue,
    limits: Limits,
    occupancy: Arc<Occupancy>,
    allocations: Allocations,
    submitter: crate::pipeline::Submitter,
}

impl fmt::Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpu").field("adapter", &self.entry).finish_non_exhaustive()
    }
}

impl Gpu {
    /// The backends D §6.8 asks for: Metal, Vulkan and DX12. No GL.
    pub const BACKENDS: Backends = Backends::METAL.union(Backends::VULKAN).union(Backends::DX12);

    /// Every adapter the instance offers, in enumeration order.
    ///
    /// This is what `sherd-refit-rs info` prints and what `--gpu-adapter` indexes. It opens no
    /// device, so it is cheap and cannot fail with a driver error.
    #[must_use]
    pub fn adapters() -> Vec<AdapterEntry> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        pollster::block_on(instance.enumerate_adapters(Self::BACKENDS))
            .iter()
            .enumerate()
            .map(|(i, adapter)| AdapterEntry::of(i, adapter))
            .collect()
    }

    /// Opens the adapter `choice` names, requesting the limits it offers.
    ///
    /// Fails with [`GpuError::NoAdapter`] when there is none, [`GpuError::NoSuchAdapter`] when the
    /// flag names one that does not exist, and [`GpuError::Limits`] when the device cannot run
    /// D §6's kernels.
    pub fn open(choice: &AdapterChoice) -> Result<Self, GpuError> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapters = pollster::block_on(instance.enumerate_adapters(Self::BACKENDS));
        let entries: Vec<AdapterEntry> =
            adapters.iter().enumerate().map(|(i, a)| AdapterEntry::of(i, a)).collect();
        let index = choice.pick(&entries)?;
        let adapter = adapters.into_iter().nth(index).ok_or(GpuError::NoAdapter)?;
        let entry = entries[index].clone();

        let limits = adapter.limits();
        let unmet = Requirements::default().unmet(&limits);
        if !unmet.is_empty() {
            return Err(GpuError::Limits { adapter: entry.to_string(), unmet });
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("sherd-refit"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| GpuError::Device { adapter: entry.to_string(), message: e.to_string() })?;
        tracing::info!(adapter = %entry, "gpu device opened");
        let occupancy = Arc::new(Occupancy::default());
        // D §6.4's one submitting thread, started with the device and stopped with it. It is given
        // clones of the two wgpu handles rather than a reference to this `Gpu`, so it adds no
        // cycle and the device is still dropped when the last `Gpu` is.
        let submitter = crate::pipeline::Submitter::start(
            device.clone(),
            queue.clone(),
            Arc::clone(&occupancy),
        );
        Ok(Self {
            adapter,
            entry,
            device,
            queue,
            limits,
            occupancy,
            allocations: Allocations::default(),
            submitter,
        })
    }

    /// The adapter this device came from.
    #[must_use]
    pub fn entry(&self) -> &AdapterEntry {
        &self.entry
    }

    /// The limits the device was granted.
    #[must_use]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// The device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The queue.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Whether the adapter is a software implementation (lavapipe, WARP).
    #[must_use]
    pub fn is_software(&self) -> bool {
        self.entry.is_software()
    }

    /// Whether the adapter shares memory with the host, which is what decides between
    /// mapped-at-creation uploads and a staging copy (D §6.3).
    #[must_use]
    pub fn is_unified(&self) -> bool {
        matches!(self.adapter.get_info().device_type, DeviceType::IntegratedGpu | DeviceType::Cpu)
    }

    /// What the kernels hold on the device, and the ceiling they may not cross
    /// ([`Allocations`]).
    #[must_use]
    pub fn allocations(&self) -> &Allocations {
        &self.allocations
    }

    /// How long the device had work outstanding, and how deep its queue got ([`Occupancy`]).
    #[must_use]
    pub fn occupancy(&self) -> &Occupancy {
        &self.occupancy
    }

    /// Runs one recorded command buffer on D §6.4's submitting thread and reads back what it
    /// wrote.
    ///
    /// This is the path every matching kernel takes. The calling thread blocks on a channel, not
    /// on `poll`, which is what keeps eighteen workers from queueing inside the device's own
    /// locks (`pipeline`).
    pub fn run_job(
        &self,
        commands: wgpu::CommandBuffer,
        staged: Option<crate::pipeline::Staged>,
    ) -> Result<Vec<u8>, GpuError> {
        self.submitter.run(commands, staged)
    }

    /// Hands one command buffer to the queue and counts it as outstanding.
    ///
    /// Every submission this crate makes on a run path goes through here, and every one of them is
    /// paired with exactly one [`Gpu::wait_for`] — that pairing is what makes [`Occupancy`] the
    /// union of the intervals rather than a guess.
    pub fn submit(&self, commands: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        self.occupancy.entered();
        self.queue.submit(Some(commands))
    }

    /// Submits the work recorded so far and blocks until the device has finished it.
    ///
    /// `PollType::wait_indefinitely()` is wgpu 30's spelling of what used to be `Maintain::Wait`,
    /// and `poll` returns a `Result` now (E7 §1).
    pub fn wait(&self) -> Result<(), GpuError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| GpuError::Poll(e.to_string()))
    }

    /// Blocks until **this** submission has finished, and no longer.
    ///
    /// [`Gpu::wait`] waits for the most recent submission at the time of the poll, whoever made
    /// it. That is the right thing for a self-test on an idle device and the wrong thing for a
    /// matching stage: ten rayon threads share one queue, and a thread that waits for "the latest
    /// submission" waits for nine other pairs' work as well as its own. Measured on
    /// `synthetic_20`, the per-dispatch times summed over the pool came to 20.6 s against a
    /// matching stage of 23.1 s — most of it other threads' work counted ten times over.
    pub fn wait_for(&self, submission: wgpu::SubmissionIndex) -> Result<(), GpuError> {
        let polled = self
            .device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
            .map(|_| ())
            .map_err(|e| GpuError::Poll(e.to_string()));
        self.occupancy.left();
        polled
    }
}

#[cfg(test)]
mod tests {
    use super::{AdapterChoice, AdapterEntry, Allocations, DEFAULT_MEMORY_BUDGET, Requirements};
    use crate::GpuError;

    fn entry(index: usize, backend: &str, name: &str, kind: &str) -> AdapterEntry {
        AdapterEntry {
            index,
            backend: backend.to_owned(),
            name: name.to_owned(),
            device_type: kind.to_owned(),
            vendor: 0,
            device: 0,
            driver: String::new(),
            driver_info: String::new(),
        }
    }

    /// D §1's ceiling refuses rather than over-allocates, releases on drop, remembers its peak
    /// and can be turned off — all of it arithmetic, so it is tested on every CI platform with no
    /// adapter at all (D §10.4 layer 1).
    #[test]
    fn the_device_budget_refuses_what_it_cannot_hold_and_forgets_nothing() {
        let allocations = Allocations::default();
        assert_eq!(allocations.budget(), DEFAULT_MEMORY_BUDGET, "D §1: slots + batches <= 1 GB");
        allocations.set_budget(1000);

        let a = allocations.reserve(600).expect("600 of 1000");
        assert_eq!(allocations.live(), 600);
        let b = allocations.reserve(300).expect("900 of 1000");
        assert_eq!(allocations.live(), 900);
        assert_eq!(allocations.peak(), 900);

        // The one that does not fit is refused, and nothing about the live total changes.
        assert!(allocations.reserve(200).is_none(), "1100 is over the budget");
        assert_eq!((allocations.live(), allocations.refused()), (900, 1));
        // A request larger than the whole budget is refused rather than admitted after everything
        // else has gone — `SlotTable`'s rule, one level down.
        drop(b);
        drop(a);
        assert_eq!(allocations.live(), 0, "a reservation is released when it goes out of scope");
        assert!(allocations.reserve(1001).is_none(), "bigger than the budget on an empty device");
        assert_eq!(allocations.refused(), 2);
        assert_eq!(allocations.peak(), 900, "the peak is the high-water mark, not the current");

        // `0` is `--memory-budget 0`'s spelling of "no bound" (D §9).
        allocations.set_budget(0);
        let huge = allocations.reserve(1 << 40).expect("no bound");
        assert_eq!(allocations.peak(), 1 << 40);
        drop(huge);
    }

    /// `--gpu-adapter` parses as an index when it is all digits and as a name otherwise, and the
    /// name matches the adapter's name or its backend, case-insensitively.
    #[test]
    fn the_adapter_flag_selects_by_index_or_by_name() {
        let entries = vec![
            entry(0, "Metal", "Apple M2 Pro", "IntegratedGpu"),
            entry(1, "Vulkan", "llvmpipe (LLVM 17)", "Cpu"),
        ];
        assert_eq!(AdapterChoice::parse("1"), AdapterChoice::Index(1));
        assert_eq!(AdapterChoice::parse("warp"), AdapterChoice::Name("warp".to_owned()));
        assert_eq!(AdapterChoice::parse("  "), AdapterChoice::Default);

        assert_eq!(AdapterChoice::Default.pick(&entries).unwrap(), 0);
        assert_eq!(AdapterChoice::parse("1").pick(&entries).unwrap(), 1);
        assert_eq!(AdapterChoice::parse("m2 PRO").pick(&entries).unwrap(), 0);
        assert_eq!(AdapterChoice::parse("vulkan").pick(&entries).unwrap(), 1);
        assert_eq!(AdapterChoice::parse("llvmpipe").pick(&entries).unwrap(), 1);
        assert!(entries[1].is_software() && !entries[0].is_software());

        // A name or an index that names nothing lists what there was, which is the error message
        // an operator can act on.
        let err = AdapterChoice::parse("nvidia").pick(&entries).unwrap_err().to_string();
        assert!(err.contains("nvidia") && err.contains("Apple M2 Pro"), "{err}");
        let err = AdapterChoice::parse("7").pick(&entries).unwrap_err().to_string();
        assert!(err.contains('7'), "{err}");
        assert!(matches!(AdapterChoice::Default.pick(&[]), Err(GpuError::NoAdapter)));
    }

    /// The limits check names every requirement a device fails, and passes wgpu's own defaults.
    #[test]
    fn the_limits_check_is_against_the_wgpu_defaults() {
        let requirements = Requirements::default();
        assert!(requirements.unmet(&wgpu::Limits::default()).is_empty(), "defaults are the floor");

        let poor = wgpu::Limits {
            max_compute_invocations_per_workgroup: 64,
            max_compute_workgroup_storage_size: 8192,
            ..Default::default()
        };
        let unmet = requirements.unmet(&poor);
        assert_eq!(unmet.len(), 2, "{unmet:?}");
        assert!(unmet[0].contains("max_compute_invocations_per_workgroup: 64 < 256"), "{unmet:?}");
        assert!(unmet[1].contains("max_compute_workgroup_storage_size: 8192 < 16384"), "{unmet:?}");

        // The adapter E7 measured clears the floor with room: 1024 lanes, 32 KB, a 4 GiB binding.
        let metal = wgpu::Limits {
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_storage_size: 32768,
            max_storage_buffer_binding_size: u64::from(u32::MAX) - 3,
            max_storage_buffers_per_shader_stage: 29,
            ..Default::default()
        };
        assert!(requirements.unmet(&metal).is_empty());
    }

    /// The display line is what `info` prints; on Metal it carries no driver, and it says so by
    /// leaving the field out rather than printing an empty one (E7 §7.6).
    #[test]
    fn the_adapter_line_omits_what_metal_does_not_report() {
        let metal = entry(0, "Metal", "Apple M2 Pro", "IntegratedGpu");
        assert_eq!(metal.to_string(), "[0] Metal Apple M2 Pro (IntegratedGpu)");
        let mut vulkan = entry(1, "Vulkan", "NVIDIA RTX 4090", "DiscreteGpu");
        vulkan.driver = "NVIDIA".to_owned();
        vulkan.driver_info = "550.54".to_owned();
        assert!(vulkan.to_string().ends_with("driver=NVIDIA 550.54"), "{vulkan}");
    }
}
