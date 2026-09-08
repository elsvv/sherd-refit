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
//!   [`Selection::reason`] says so out loud rather than leaving the operator to infer it.
//! * **Metal reports no vendor, device id, driver or driver_info** (E7 §7.6), so `--gpu-adapter
//!   NAME` can only match on the adapter name and a run report cannot record a driver version.
//! * **The adapter's own limits are accepted verbatim** (E7 §2), and they are far above the wgpu
//!   defaults here (32 KB of workgroup storage against 16, a 4 GiB binding cap against 128 MB).
//!   The device therefore *requests* what the adapter offers — nothing is gained by asking for
//!   less — while the kernels are written to the **defaults**, because that is what a portable
//!   build gets. [`Requirements`] is the floor both must clear.

use std::fmt;

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

/// An open device: the adapter that was chosen, the limits it granted, and the queue.
pub struct Gpu {
    adapter: Adapter,
    entry: AdapterEntry,
    device: Device,
    queue: Queue,
    limits: Limits,
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
        Ok(Self { adapter, entry, device, queue, limits })
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
}

#[cfg(test)]
mod tests {
    use super::{AdapterChoice, AdapterEntry, Requirements};
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
