//! D §5 step 2's memory-aware preprocessing semaphore, on E1's measured constant.
//!
//! Preprocessing is the memory peak of a run: R §3.1 reads a whole scan into memory before R §3.3
//! decimates it, so `k` concurrent scans hold `k` originals at once and the pool is as wide as the
//! machine has cores. D §5 bounds that with an admission rule rather than with a smaller pool —
//! four small scans should still run four wide — and D §9 gives it the flag `--memory-budget GB`.
//!
//! # The constant
//!
//! E1 fitted seven single-scan runs of 53 k to 1.34 M faces
//! (`notes/2026-09-07-e1-profile.md` §7):
//!
//! ```text
//! peak RSS = 98.4 MiB + 361 B per original face      (R² = 0.978, residuals <= 36 MiB)
//! ```
//!
//! and — this is the part that decides the rule — the fixed term is the **process**, not the
//! scan: four concurrent scans of 4.33 M faces measured 1 492 MiB against 1 589 MiB predicted by
//! charging the floor once, and 1 886 MiB predicted by charging it per job. So a job reserves the
//! *slope alone*, [`BYTES_PER_FACE`], and the floor, [`PROCESS_FLOOR`], comes off the budget once.
//! D §8's row is that arithmetic: `⌊(budget − 98 MiB) / cost⌋` concurrent scans.
//!
//! # The rule, and why it cannot deadlock
//!
//! [`admits`] is the whole of it: a job may start when nothing else holds a reservation, or when
//! its own reservation still fits in the budget. The first clause is what makes a scan larger than
//! the entire budget run at all — alone, which is the most the machine can do for it — and it is
//! why no arrangement of scan sizes can wedge the pool.
//!
//! # Why it moves no result
//!
//! The semaphore reorders *when* fragments are preprocessed, never what they are: [`preprocess`]
//! collects by index, R §3's per-fragment work reads nothing but its own file, and every seeded
//! draw is seeded from `Params` and the fragment. A run under a budget that forces one scan at a
//! time writes the same caches and the same outputs, byte for byte, as an unbounded one — which
//! is the gate step E2 checked it against.
//!
//! [`preprocess`]: crate::pipeline::preprocess

use std::path::Path;
use std::sync::{Condvar, Mutex};

use crate::io::MeshFormat;

/// E1's slope: bytes of peak RSS per face of the *original* scan.
///
/// 361 B/face is 344 MiB per million faces, which is the number D §5 and D §8 quote.
pub const BYTES_PER_FACE: u64 = 361;

/// E1's intercept: the process itself, paid once however many scans run.
pub const PROCESS_FLOOR: u64 = 98 * 1024 * 1024;

/// The budget assumed when the machine's physical memory cannot be read.
///
/// Deliberately small: an unknown machine is more likely to be a CI container with a few
/// gigabytes than a workstation, and the flag is there for anyone who knows better.
pub const FALLBACK_MEMORY: u64 = 8 * 1024 * 1024 * 1024;

/// Bytes of peak RSS a scan of `faces` faces is expected to add.
#[must_use]
pub fn reservation(faces: u64) -> u64 {
    faces.saturating_mul(BYTES_PER_FACE)
}

/// D §5's admission rule: may a job wanting `want` bytes start now?
///
/// `in_flight` is what the jobs already admitted reserved and `running` how many of them there
/// are. A job is admitted when it fits, and — whatever its size — when it would otherwise be the
/// only thing the machine is doing. Without that second clause a scan bigger than the budget
/// would wait for a reservation that can never be released.
#[must_use]
pub fn admits(budget: u64, in_flight: u64, running: usize, want: u64) -> bool {
    running == 0 || in_flight.saturating_add(want) <= budget
}

/// How many scans of `faces` faces a budget of `budget` bytes admits at once (D §8's row).
///
/// At least one, because [`admits`] always lets a lone job through.
#[must_use]
pub fn concurrent_scans(budget: u64, faces: u64) -> usize {
    let cost = reservation(faces);
    if cost == 0 {
        return usize::MAX;
    }
    usize::try_from(budget / cost).unwrap_or(usize::MAX).max(1)
}

/// What a run is allowed to hold in flight during preprocessing (D §9's `--memory-budget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Bytes available to concurrent scans — the flag's value minus [`PROCESS_FLOOR`], or
    /// `u64::MAX` when there is no bound.
    available: u64,
    bounded: bool,
}

impl Budget {
    /// No bound at all: every fragment starts as soon as the pool has a thread for it.
    ///
    /// This is what the parity harness and the tests want, and what a run gets when the flag is
    /// given a non-positive number.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self { available: u64::MAX, bounded: false }
    }

    /// A budget of `bytes`, out of which [`PROCESS_FLOOR`] is the process's own.
    #[must_use]
    pub fn bytes(bytes: u64) -> Self {
        Self { available: bytes.saturating_sub(PROCESS_FLOOR), bounded: true }
    }

    /// D §9's flag: a budget in gigabytes (`10^9`, the unit a user types on a command line).
    ///
    /// Zero or less is [`Budget::unbounded`].
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped positive and far under 2^64 bytes"
    )]
    pub fn gigabytes(gb: f64) -> Self {
        if gb <= 0.0 || gb.is_nan() {
            return Self::unbounded();
        }
        Self::bytes((gb * 1e9) as u64)
    }

    /// D §5's default: half of the machine's physical memory.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a byte count")]
    pub fn default_for_machine() -> Self {
        Self::bytes(physical_memory().unwrap_or(FALLBACK_MEMORY) / 2)
    }

    /// Bytes concurrent scans may hold between them.
    #[must_use]
    pub fn available(self) -> u64 {
        self.available
    }

    /// False for [`Budget::unbounded`].
    #[must_use]
    pub fn is_bounded(self) -> bool {
        self.bounded
    }

    /// How many scans of `faces` faces this budget admits at once.
    #[must_use]
    pub fn concurrent_scans(self, faces: u64) -> usize {
        if !self.bounded {
            return usize::MAX;
        }
        concurrent_scans(self.available, faces)
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self::default_for_machine()
    }
}

/// The semaphore itself: reservations in bytes, granted in the order threads arrive at the
/// condition variable.
///
/// It is a counting semaphore with variable weights rather than a permit count, because the jobs
/// are not interchangeable: one 10 M-face scan and thirty 50 k-face ones must not be admitted by
/// the same rule.
#[derive(Debug)]
pub struct MemorySemaphore {
    budget: Budget,
    state: Mutex<State>,
    released: Condvar,
}

#[derive(Debug, Default)]
struct State {
    in_flight: u64,
    running: usize,
    /// Jobs that had to wait for a reservation, for the log line and the tests.
    waited: usize,
    /// The most bytes ever reserved at once.
    peak: u64,
    /// The most jobs ever admitted at once.
    peak_running: usize,
}

/// What the semaphore did over a stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemaphoreStats {
    /// Jobs that blocked before they were admitted.
    pub waited: usize,
    /// The largest reservation held at one instant, in bytes.
    pub peak: u64,
    /// The most jobs admitted at one instant.
    pub peak_running: usize,
}

impl MemorySemaphore {
    /// A semaphore over `budget`.
    #[must_use]
    pub fn new(budget: Budget) -> Self {
        Self { budget, state: Mutex::new(State::default()), released: Condvar::new() }
    }

    /// The budget it was built with.
    #[must_use]
    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Blocks until `want` bytes may be held, then returns the permit that releases them.
    ///
    /// An unbounded budget never blocks and never touches the mutex twice.
    pub fn acquire(&self, want: u64) -> Permit<'_> {
        if !self.budget.bounded {
            return Permit { semaphore: self, held: 0 };
        }
        let Ok(mut state) = self.state.lock() else {
            // A poisoned mutex means another thread panicked while holding it; the run is already
            // failing, and blocking here would only turn a panic into a hang.
            return Permit { semaphore: self, held: 0 };
        };
        let mut waited = false;
        while !admits(self.budget.available, state.in_flight, state.running, want) {
            waited = true;
            let Ok(next) = self.released.wait(state) else {
                return Permit { semaphore: self, held: 0 };
            };
            state = next;
        }
        state.in_flight = state.in_flight.saturating_add(want);
        state.running += 1;
        state.waited += usize::from(waited);
        state.peak = state.peak.max(state.in_flight);
        state.peak_running = state.peak_running.max(state.running);
        Permit { semaphore: self, held: want }
    }

    /// What it did so far.
    #[must_use]
    pub fn stats(&self) -> SemaphoreStats {
        self.state.lock().map_or_else(
            |_| SemaphoreStats::default(),
            |s| SemaphoreStats { waited: s.waited, peak: s.peak, peak_running: s.peak_running },
        )
    }

    fn release(&self, held: u64) {
        if held == 0 && !self.budget.bounded {
            return;
        }
        if let Ok(mut state) = self.state.lock() {
            state.in_flight = state.in_flight.saturating_sub(held);
            state.running = state.running.saturating_sub(1);
        }
        self.released.notify_all();
    }
}

/// A reservation, released when it is dropped.
#[derive(Debug)]
pub struct Permit<'a> {
    semaphore: &'a MemorySemaphore,
    held: u64,
}

impl Permit<'_> {
    /// Bytes this permit holds.
    #[must_use]
    pub fn held(&self) -> u64 {
        self.held
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.semaphore.release(self.held);
    }
}

/// How many faces a scan file holds, without reading the mesh.
///
/// PLY and OFF state the count in their headers and binary STL in its 84-byte prologue, so those
/// three are exact. OBJ, ASCII STL and GLB do not, and a reservation is not worth a full parse, so
/// the count is estimated from the file's size with a per-format constant measured on this
/// project's own collections — deliberately **below** the measured bytes per face, so that the
/// estimate errs high and the semaphore errs towards admitting fewer jobs.
///
/// `None` means the file could not be read at all, which the caller treats as "no reservation":
/// the load is about to fail anyway and it should fail with the reader's error, not by waiting.
#[must_use]
pub fn scan_faces(path: &Path) -> Option<u64> {
    let format = MeshFormat::from_path(path).ok()?;
    let size = std::fs::metadata(path).ok()?.len();
    match format {
        MeshFormat::Ply => ply_header_faces(path).or(Some(size / PLY_BYTES_PER_FACE)),
        MeshFormat::Off => off_header_faces(path).or(Some(size / OFF_BYTES_PER_FACE)),
        MeshFormat::Stl => Some(stl_faces(path).unwrap_or(size / STL_BYTES_PER_FACE)),
        MeshFormat::Obj => Some(size / OBJ_BYTES_PER_FACE),
        MeshFormat::Glb => Some(size / GLB_BYTES_PER_FACE),
    }
}

/// Bytes per face of a binary PLY with `float` positions, `uchar` colours and a `uchar int` face
/// list: 15 B a vertex and 13 B a face over about half as many vertices as faces — used only when
/// the header cannot be parsed.
const PLY_BYTES_PER_FACE: u64 = 20;
/// OFF is ASCII; the terracotta-sized files run about 45 B a face.
const OFF_BYTES_PER_FACE: u64 = 30;
/// ASCII STL is ~250 B a facet, binary 50; the estimate is only reached for the ASCII form.
const STL_BYTES_PER_FACE: u64 = 150;
/// Measured on the eleven `pot_H` OBJ files: 123.6–131.0 B a face. 100 rounds the estimate up.
const OBJ_BYTES_PER_FACE: u64 = 100;
/// GLB is binary: 12 B a vertex position plus 6 B an index, over half as many vertices as faces.
const GLB_BYTES_PER_FACE: u64 = 12;

/// `element face N` out of a PLY header, reading no more than the header.
fn ply_header_faces(path: &Path) -> Option<u64> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut faces = None;
    for _ in 0..512 {
        line.clear();
        // A binary PLY's *header* is ASCII, but `read_line` would choke on the body that follows;
        // the loop stops at `end_header`, and a header line that is not UTF-8 ends the search.
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed == "end_header" {
            break;
        }
        let mut words = trimmed.split_ascii_whitespace();
        if words.next() == Some("element") && words.next() == Some("face") {
            faces = words.next().and_then(|n| n.parse::<u64>().ok());
        }
    }
    faces
}

/// The face count on an OFF file's counts line (the second non-comment line).
fn off_header_faces(path: &Path) -> Option<u64> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut seen_magic = false;
    for line in reader.lines().take(64) {
        let line = line.ok()?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !seen_magic {
            seen_magic = true;
            // `OFF 8 12 0` on one line is legal, and so is `OFF` on its own.
            let mut words = trimmed.split_ascii_whitespace();
            let head = words.next()?;
            if !head.ends_with("OFF") {
                return None;
            }
            if let Some(faces) = words.nth(1).and_then(|n| n.parse::<u64>().ok()) {
                return Some(faces);
            }
            continue;
        }
        return trimmed.split_ascii_whitespace().nth(1).and_then(|n| n.parse::<u64>().ok());
    }
    None
}

/// The triangle count of a binary STL: a `u32` at offset 80. `None` for the ASCII form.
fn stl_faces(path: &Path) -> Option<u64> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = [0_u8; 84];
    file.read_exact(&mut head).ok()?;
    if head.starts_with(b"solid") {
        return None;
    }
    let count = u32::from_le_bytes([head[80], head[81], head[82], head[83]]);
    Some(u64::from(count))
}

/// The machine's physical memory in bytes, or `None` when this platform has no way to say.
///
/// Linux reads `/proc/meminfo`; macOS asks `sysctl`, which is a process rather than a `libc` call
/// because the workspace denies `unsafe_code` and this is a once-per-run question. Everything else
/// falls back to [`FALLBACK_MEMORY`].
#[must_use]
pub fn physical_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.split_ascii_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8(out.stdout).ok()?.trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// The process's resident set size in bytes, or `None` where the platform will not say.
///
/// Linux reads `/proc/self/statm`, whose second field is the resident pages, **assumed 4 KiB** —
/// `sysconf(_SC_PAGESIZE)` is a `libc` call and the workspace denies `unsafe_code`, so a kernel
/// built with 64 KiB pages would read sixteen times low here; every machine this port is measured
/// on is 4 KiB. macOS asks `ps`, for the same reason [`physical_memory`] asks `sysctl`, and
/// `task_info` is the only other way to the number. The `ps` costs **4.2 ms** measured on this
/// machine, which is why [`RssMonitor`] samples it on a thread of its own and at 100 ms rather
/// than inside the stages.
#[must_use]
pub fn resident_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: u64 = text.split_ascii_whitespace().nth(1)?.parse().ok()?;
        Some(pages * 4096)
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("/bin/ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let kb: u64 = String::from_utf8(out.stdout).ok()?.trim().parse().ok()?;
        Some(kb * 1024)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// How often [`RssMonitor`] reads the resident set.
///
/// 100 ms is a compromise the measurement itself sets: `preprocess` on a warm cache is under a
/// second on the development sets, so a coarser interval would miss its peak, and the sample costs
/// one `ps` — 4.2 ms of one core, about 4 % of one of the ten this machine has, and 0.4 % of the
/// machine. A 28-minute 170-fragment run pays roughly 70 s of one core for it.
const SAMPLE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Peak resident set size, per stage: a sampler thread and the high-water mark since the last
/// stage boundary.
///
/// D §8's table is a model with one measurement in it (E2's 1.9 GiB for a whole run). The audit's
/// §B.3 asks a question the model cannot answer — *which stage* holds the peak, and what a
/// structure that is never freed costs at it — and that needs the number per stage, in the run
/// itself, on every run rather than in a one-off experiment. This is that instrument:
/// [`RssMonitor::take_peak`] closes a window and opens the next, and the pipeline calls it where
/// it writes a timing.
///
/// It is a *sampler*, so it reports the largest resident set it **saw**: a spike shorter than
/// [`SAMPLE_INTERVAL`] can pass between two samples. That is the honest limit of reading a
/// number the kernel only publishes on request, and it is why the block this feeds sits beside
/// `timings` — a wall clock and a sampled peak are the two parts of `report.json` that two runs
/// of the same build legitimately disagree on.
#[derive(Debug)]
pub struct RssMonitor {
    shared: std::sync::Arc<Sampler>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// What the monitor and its thread share.
#[derive(Debug, Default)]
struct Sampler {
    /// The largest sample of the current window, and of the run.
    marks: Mutex<Marks>,
    /// Raised to stop the thread; the condition variable wakes it out of its wait.
    stop: Mutex<bool>,
    wake: Condvar,
}

#[derive(Debug, Default, Clone, Copy)]
struct Marks {
    window: u64,
    run: u64,
}

impl Sampler {
    /// Reads the resident set and folds it into both marks.
    fn sample(&self) -> Option<u64> {
        let rss = resident_memory()?;
        if let Ok(mut marks) = self.marks.lock() {
            marks.window = marks.window.max(rss);
            marks.run = marks.run.max(rss);
        }
        Some(rss)
    }
}

impl RssMonitor {
    /// Starts sampling, or returns `None` on a platform whose resident set cannot be read.
    ///
    /// The first sample is taken on this thread, so a monitor that exists has at least one.
    #[must_use]
    pub fn start() -> Option<Self> {
        let shared = std::sync::Arc::new(Sampler::default());
        shared.sample()?;
        let worker = std::sync::Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("sherd-rss".to_owned())
            .spawn(move || {
                loop {
                    let Ok(stop) = worker.stop.lock() else { return };
                    let Ok((stop, _)) = worker.wake.wait_timeout(stop, SAMPLE_INTERVAL) else {
                        return;
                    };
                    if *stop {
                        return;
                    }
                    drop(stop);
                    if worker.sample().is_none() {
                        return;
                    }
                }
            })
            .ok()?;
        Some(Self { shared, thread: Some(thread) })
    }

    /// The largest resident set **seen while the window was open**, in bytes, and a new window
    /// opens empty.
    ///
    /// A stage shorter than [`SAMPLE_INTERVAL`] still gets a number: this takes a sample of its
    /// own, which closes the window it reads. The next window starts at zero rather than at that
    /// sample, so a stage is charged for what was resident *during* it and never for what the
    /// stage before it was still holding at the boundary — which is the whole question audit §B.3
    /// asks.
    pub fn take_peak(&self) -> u64 {
        self.shared.sample();
        let Ok(mut marks) = self.shared.marks.lock() else { return 0 };
        let peak = marks.window;
        marks.window = 0;
        peak
    }

    /// The largest resident set seen over the whole run, in bytes.
    #[must_use]
    pub fn peak(&self) -> u64 {
        self.shared.sample();
        self.shared.marks.lock().map_or(0, |m| m.run)
    }
}

impl Drop for RssMonitor {
    fn drop(&mut self) {
        if let Ok(mut stop) = self.shared.stop.lock() {
            *stop = true;
        }
        self.shared.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        BYTES_PER_FACE, Budget, MemorySemaphore, PROCESS_FLOOR, RssMonitor, admits,
        concurrent_scans, physical_memory, reservation, resident_memory, scan_faces,
    };

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    /// D §8's own row, on the face counts it quotes: at 8 GiB the budget admits 23 scans of 1 M
    /// faces, 11 of 2 M and 2 of 10 M; at the 4.5 GiB E1 §7.3 derives for a 170-fragment run it is
    /// 13, 6 and 1.
    #[test]
    fn the_admission_rule_reproduces_the_design_table() {
        let wide = Budget::bytes(8 * GIB);
        assert_eq!(wide.concurrent_scans(1_000_000), 23);
        assert_eq!(wide.concurrent_scans(2_000_000), 11);
        assert_eq!(wide.concurrent_scans(4_000_000), 5);
        assert_eq!(wide.concurrent_scans(10_000_000), 2);

        let tight = Budget::bytes(4608 * MIB);
        assert_eq!(tight.concurrent_scans(1_000_000), 13);
        assert_eq!(tight.concurrent_scans(2_000_000), 6);
        assert_eq!(tight.concurrent_scans(10_000_000), 1, "one at a time, never none");

        // The constant is E1's: 344 MiB per million faces over a 98 MiB process floor.
        assert_eq!(reservation(1_000_000), 361_000_000);
        assert_eq!(reservation(1_000_000) / MIB, 344, "344 MiB per million faces");
        assert_eq!(Budget::bytes(8 * GIB).available(), 8 * GIB - PROCESS_FLOOR);
        assert_eq!(BYTES_PER_FACE, 361);
    }

    /// The rule itself, on synthetic face counts: a job that fits is admitted, one that does not
    /// waits — unless it would be the only job on the machine.
    #[test]
    fn a_job_larger_than_the_whole_budget_still_runs_alone() {
        let budget = reservation(1_000_000); // one million faces and not a face more
        assert!(admits(budget, 0, 0, reservation(1_000_000)));
        assert!(!admits(budget, reservation(600_000), 1, reservation(600_000)), "1.2 M > 1 M");
        assert!(admits(budget, reservation(600_000), 1, reservation(400_000)), "exactly 1 M fits");
        // The clause that removes the deadlock: nothing is running, so anything may start.
        assert!(admits(budget, 0, 0, reservation(10_000_000)));
        assert!(!admits(budget, reservation(10_000_000), 1, reservation(1)));

        // An unbounded budget admits everything, and says so.
        assert!(!Budget::unbounded().is_bounded());
        assert_eq!(Budget::unbounded().concurrent_scans(10_000_000), usize::MAX);
        assert_eq!(Budget::gigabytes(0.0), Budget::unbounded());
        assert_eq!(Budget::gigabytes(-1.0), Budget::unbounded());
        assert_eq!(Budget::gigabytes(8.0).available(), 8_000_000_000 - PROCESS_FLOOR);
        assert_eq!(concurrent_scans(0, 0), usize::MAX);
    }

    /// The semaphore holds the rule under threads: with room for two scans of a million faces,
    /// three threads never overlap more than two, and all three finish.
    #[test]
    fn the_semaphore_never_admits_more_than_the_budget() {
        let budget = Budget::bytes(2 * reservation(1_000_000) + PROCESS_FLOOR);
        let semaphore = Arc::new(MemorySemaphore::new(budget));
        let live = Arc::new(AtomicUsize::new(0));
        let worst = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (semaphore, live, worst, done) = (
                    Arc::clone(&semaphore),
                    Arc::clone(&live),
                    Arc::clone(&worst),
                    Arc::clone(&done),
                );
                std::thread::spawn(move || {
                    let permit = semaphore.acquire(reservation(1_000_000));
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    worst.fetch_max(now, Ordering::SeqCst);
                    std::thread::yield_now();
                    live.fetch_sub(1, Ordering::SeqCst);
                    done.fetch_add(1, Ordering::SeqCst);
                    drop(permit);
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no worker panics");
        }
        assert_eq!(done.load(Ordering::SeqCst), 8, "every job ran");
        assert!(worst.load(Ordering::SeqCst) <= 2, "{}", worst.load(Ordering::SeqCst));
        assert_eq!(semaphore.stats().peak_running, worst.load(Ordering::SeqCst));
        assert!(semaphore.stats().peak <= budget.available());
    }

    /// A budget that fits nothing runs one job at a time and every job still finishes.
    #[test]
    fn a_budget_of_almost_nothing_serialises_rather_than_deadlocks() {
        let semaphore = Arc::new(MemorySemaphore::new(Budget::bytes(PROCESS_FLOOR + 1)));
        let worst = Arc::new(AtomicUsize::new(0));
        let live = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..6)
            .map(|k| {
                let (semaphore, live, worst) =
                    (Arc::clone(&semaphore), Arc::clone(&live), Arc::clone(&worst));
                std::thread::spawn(move || {
                    let permit = semaphore.acquire(reservation(100_000 * (k + 1)));
                    worst.fetch_max(live.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    std::thread::yield_now();
                    live.fetch_sub(1, Ordering::SeqCst);
                    drop(permit);
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no worker panics");
        }
        assert_eq!(worst.load(Ordering::SeqCst), 1, "strictly one at a time");
        assert_eq!(semaphore.stats().peak_running, 1);
    }

    /// An unbounded semaphore is not a semaphore: it never blocks and it records no waits.
    #[test]
    fn an_unbounded_budget_admits_everything_at_once() {
        let semaphore = MemorySemaphore::new(Budget::unbounded());
        let a = semaphore.acquire(reservation(10_000_000));
        let b = semaphore.acquire(reservation(10_000_000));
        assert_eq!(a.held(), 0, "an unbounded permit holds nothing");
        assert_eq!(semaphore.stats().waited, 0);
        assert!(!semaphore.budget().is_bounded());
        drop((a, b));
    }

    /// The face count comes out of the header on the formats that state one.
    #[test]
    fn a_scan_declares_its_own_size() {
        let slab = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/slab/input/pieceA.ply");
        let faces = scan_faces(std::path::Path::new(slab)).expect("the committed slab fixture");
        let mesh = crate::io::read_mesh(slab).expect("readable");
        assert_eq!(faces, mesh.f.len() as u64, "the PLY header states its own face count");

        let dir = std::env::temp_dir().join(format!("sherd-scan-faces-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let off = dir.join("cube.off");
        std::fs::write(&off, "OFF\n8 12 0\n").expect("writable");
        assert_eq!(scan_faces(&off), Some(12));
        let inline = dir.join("inline.off");
        std::fs::write(&inline, "OFF 8 12 0\n").expect("writable");
        assert_eq!(scan_faces(&inline), Some(12));
        let obj = dir.join("mesh.obj");
        std::fs::write(&obj, vec![b'x'; 400]).expect("writable");
        assert_eq!(scan_faces(&obj), Some(4), "OBJ is estimated from its size");
        assert_eq!(scan_faces(&dir.join("missing.ply")), None);
        assert_eq!(scan_faces(std::path::Path::new("x.unknown")), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Whatever the platform answers, the default budget is a positive number of bytes.
    #[test]
    fn the_default_budget_is_half_of_what_the_machine_has() {
        let budget = Budget::default_for_machine();
        assert!(budget.is_bounded());
        assert!(budget.available() > 0);
        if let Some(ram) = physical_memory() {
            assert!(ram >= GIB, "a machine with less than a gigabyte is not one of ours");
            assert_eq!(budget.available(), ram / 2 - PROCESS_FLOOR);
        }
    }

    /// The instrument audit §B.3's decision is made on: a resident set that is a real number, and
    /// a window that closes.
    ///
    /// The assertions are the ones a sampler can keep. The process is alive and holds a mesh
    /// library, so its resident set is between a megabyte and the machine's memory; a window's
    /// peak is at least what the *next* window starts from, because `take_peak` reseeds the
    /// window with the sample it just took; and the run's peak is never below a window's.
    #[test]
    fn the_resident_set_can_be_read_and_a_window_closes() {
        let Some(rss) = resident_memory() else {
            return; // not this platform's number to give (the monitor returns `None` too)
        };
        assert!(rss > MIB, "{rss} B resident is not a running process");
        assert!(rss < 1024 * GIB, "{rss} B resident is not this machine");

        let monitor = RssMonitor::start().expect("this platform reads its own resident set");
        let held: Vec<u64> = (0..2_000_000).collect(); // ~16 MB the sampler can see
        let first = monitor.take_peak();
        assert!(first >= rss / 2, "the first window saw the process: {first} against {rss}");
        let second = monitor.take_peak();
        assert!(second > 0, "a window shorter than the interval still takes its own sample");
        assert!(monitor.peak() >= first.max(second), "the run's peak covers every window");
        assert!(second >= rss, "an empty window still closes on its own sample");
        assert_eq!(held.len(), 2_000_000, "the allocation is not optimised away");
    }
}
