//! D §6.4's software pipeline: **one submitting thread, double-buffered.**
//!
//! # Why the design says "one"
//!
//! Task G2 left the matching stage at 1.0× and read the cause as an idle device. Task G3 measured
//! it and the cause is narrower than that: `wgpu::Device::poll` is not a per-thread wait. Every
//! worker that hands a batch to the device calls `poll(Wait { submission_index })`, every one of
//! those takes the device's own locks, and on `synthetic_20` the effect is visible from three
//! sides at once (`notes/2026-09-08-g3-pipeline.md` §3):
//!
//! * the matching stage stops improving at **four** worker threads — 16.08 s at four against
//!   15.97 s at nine, where the CPU work alone would have gone on falling;
//! * with eighteen threads in the pool the profile shows **4.9 of ten cores** doing anything, the
//!   other thirteen threads asleep inside a poll;
//! * the same 330 submissions that take **6.4 s** end to end on one thread take 14 s of wall to
//!   drain through nine.
//!
//! So the queue in front of the device is a host-side queue, and the design's answer — "CPU
//! threads prepare, **one GPU thread submits**; double-buffered" — is the answer to exactly this.
//!
//! # What this is
//!
//! A worker builds its buffers and records its command buffer on its own thread (that part *is*
//! preparation and it parallelises), then hands the finished command buffer and the `MAP_READ`
//! buffer it wants back to [`Submitter::run`] and waits on a channel. One thread owns every
//! `submit`, every `poll` and every `map_async` for the life of the device, and it keeps
//! [`DEPTH`] command buffers in flight so the device starts the next one the instant it finishes
//! the last rather than waiting for a host round trip.
//!
//! Nothing here can move a result. A job computes into its own buffers, the reply goes to the
//! thread that asked, and the pipeline's own results are collected by index (D §5); the order the
//! submitter happens to interleave two pairs' work in is not observable.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::JoinHandle;

use wgpu::{Buffer, CommandBuffer, Device, Queue, SubmissionIndex};

use crate::GpuError;
use crate::device::Occupancy;

/// How many command buffers the submitter keeps in flight.
///
/// D §6.4's word is "double-buffered" and **the measurement says four**. `synthetic_20`'s matching
/// stage, three runs at each depth, the pool at its default nine:
///
/// | depth | matching stage |
/// |---|---|
/// | 2 | 12.75, 13.41, 14.62 s |
/// | **4** | **12.05, 12.13, 12.79 s** |
/// | 8 | 12.79, 13.61, 14.05 s |
///
/// Two is not enough because the submitter has host work of its own between two retires — the map,
/// the copy out, the reply — and while it does that the device can drain a queue of one. Eight is
/// worse than four: the jobs are then queued long before they run, and the worker that is waiting
/// for the oldest of them waits behind seven others.
pub const DEPTH: usize = 4;

/// A `MAP_READ` buffer the submitter maps once its command buffer has run.
#[derive(Debug)]
pub struct Staged {
    /// The buffer the command buffer copied into.
    pub buffer: Buffer,
    /// How many bytes of it the caller wants.
    pub bytes: usize,
}

/// One unit of device work: a recorded command buffer and where to send what it produced.
struct Job {
    commands: CommandBuffer,
    staged: Option<Staged>,
    reply: SyncSender<Result<Vec<u8>, GpuError>>,
}

/// A submitted command buffer, waiting to be retired.
struct Pending {
    index: SubmissionIndex,
    staged: Option<Staged>,
    mapped: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    reply: SyncSender<Result<Vec<u8>, GpuError>>,
}

/// The one thread that submits (D §6.4).
#[derive(Debug)]
pub struct Submitter {
    tx: Option<Sender<Job>>,
    thread: Option<JoinHandle<()>>,
}

impl Submitter {
    /// Starts the thread.
    ///
    /// It is given clones of the device and the queue rather than the [`Gpu`](crate::Gpu) that
    /// holds them — those are handles, cloning one costs a refcount, and a submitter that held its
    /// owner would keep it alive for ever.
    #[must_use]
    pub fn start(device: Device, queue: Queue, occupancy: Arc<Occupancy>) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let thread = std::thread::Builder::new()
            .name("sherd-gpu-submit".to_owned())
            .spawn(move || pump(&device, &queue, &occupancy, &rx))
            .ok();
        Self { tx: thread.is_some().then_some(tx), thread }
    }

    /// Submits `commands` and returns the bytes of `staged` once the device has finished.
    ///
    /// The calling thread blocks on a channel rather than on the device, which is the whole point:
    /// a blocked worker holds no device lock, and [`Executor::device_slack`] then covers the core
    /// it is not using.
    ///
    /// [`Executor::device_slack`]: sherd_core::executor::Executor::device_slack
    pub fn run(
        &self,
        commands: CommandBuffer,
        staged: Option<Staged>,
    ) -> Result<Vec<u8>, GpuError> {
        let Some(tx) = self.tx.as_ref() else {
            return Err(GpuError::Poll("the submitter thread could not be started".to_owned()));
        };
        let (reply, answer) = mpsc::sync_channel(1);
        tx.send(Job { commands, staged, reply })
            .map_err(|_| GpuError::Poll("the submitter thread has stopped".to_owned()))?;
        answer
            .recv()
            .map_err(|_| GpuError::Poll("the submitter thread dropped a job".to_owned()))?
    }
}

impl Drop for Submitter {
    fn drop(&mut self) {
        // Dropping the sender is what tells the pump there is no more work; the join then waits
        // for the jobs that are still in flight, so no device buffer outlives its submission.
        self.tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The submitter's loop: fill to [`DEPTH`], then retire the oldest.
fn pump(device: &Device, queue: &Queue, occupancy: &Occupancy, rx: &Receiver<Job>) {
    let mut inflight: VecDeque<Pending> = VecDeque::with_capacity(DEPTH);
    loop {
        while inflight.len() < DEPTH {
            // Blocking only when there is nothing in flight: with a command buffer on the device
            // the right thing to do is to go and retire it, not to sit on the channel.
            let job = if inflight.is_empty() { rx.recv().ok() } else { rx.try_recv().ok() };
            let Some(job) = job else { break };
            inflight.push_back(submit(queue, occupancy, job));
        }
        let Some(pending) = inflight.pop_front() else { break };
        retire(device, occupancy, pending);
    }
}

/// Hands one job to the queue and asks for its readback.
fn submit(queue: &Queue, occupancy: &Occupancy, job: Job) -> Pending {
    occupancy.entered();
    let index = queue.submit(Some(job.commands));
    // `map_async` after the submission and before the poll — the order `buffers::read_back`
    // established, and the one wgpu wants: the callback is run by the poll that waits for the
    // copy.
    let mapped = job.staged.as_ref().map(|staged| {
        let (tx, rx) = mpsc::channel();
        staged.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        rx
    });
    Pending { index, staged: job.staged, mapped, reply: job.reply }
}

/// Waits for one submission, maps what it wrote and answers the worker that asked.
fn retire(device: &Device, occupancy: &Occupancy, pending: Pending) {
    let Pending { index, staged, mapped, reply } = pending;
    let polled = device
        .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
        .map(|_| ())
        .map_err(|e| GpuError::Poll(e.to_string()));
    occupancy.left();
    let answer = polled.and_then(|()| collect(staged, mapped));
    let _ = reply.send(answer);
}

/// The bytes of a mapped staging buffer, or an empty vector when the job wanted none.
fn collect(
    staged: Option<Staged>,
    mapped: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
) -> Result<Vec<u8>, GpuError> {
    let (Some(staged), Some(mapped)) = (staged, mapped) else { return Ok(Vec::new()) };
    let what = "a device buffer";
    match mapped.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(GpuError::Readback { what, message: e.to_string() }),
        Err(e) => return Err(GpuError::Readback { what, message: e.to_string() }),
    }
    let slice = staged.buffer.slice(..);
    let view = slice
        .get_mapped_range()
        .map_err(|e| GpuError::Readback { what, message: e.to_string() })?;
    let out = view[..staged.bytes].to_vec();
    drop(view);
    staged.buffer.unmap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::DEPTH;

    /// D §6.4 says double-buffered; the sweep in [`DEPTH`]'s own documentation says four, and a
    /// depth of one would be no pipeline at all.
    #[test]
    fn the_pipeline_keeps_more_than_one_command_buffer_in_flight() {
        const { assert!(DEPTH >= 2, "a depth of one is a host round trip between every dispatch") };
        const { assert!(DEPTH == 4, "measured: 4 beats 2 and 8 on synthetic_20") };
    }
}
