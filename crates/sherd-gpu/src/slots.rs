//! Fragment slots on the device: an LRU of 32, capped at 400 MB (D §6.3).
//!
//! A pair references two fragments, and a block of 3×3 fragments references nine; the collection
//! has up to 170. D §6.3's answer is a **resident set**: each fragment's `S`, `Pf`, `Nf`, `brk_P`,
//! `brk_ns`, `brk_sub`, `Pm`, `Nm` and its two BVHs are uploaded once and referenced by slot index
//! for as long as the scheduler keeps coming back to that fragment. The block schedule of D §5 is
//! what makes that pay: it visits pairs in 3×3 blocks precisely so a fragment's neighbours are
//! asked for while it is still resident.
//!
//! This module is the bookkeeping — which fragment is in which slot, what to evict, and whether
//! the next upload fits — and it holds no wgpu type at all. That is deliberate: the eviction rule
//! is where a scheduler bug would hide, and it is testable on every CI platform, with no adapter,
//! by parking integers in the slots (D §10.4 layer 1).
//!
//! # The two limits, and which one bites first
//!
//! * **32 slots.** D §6.3's number.
//! * **400 MB.** Also D §6.3's, and on real fragments it is the one that bites: a 200 000-face
//!   working mesh with its two BVHs and eight point arrays is 10–20 MB, so 32 of them is
//!   320–640 MB. [`SlotTable`] evicts on whichever limit is reached first, least-recently-used
//!   first, and a fragment larger than the whole budget is refused rather than admitted after
//!   evicting everything — the caller then runs that pair on the CPU, which is D §6.8's
//!   "on adapters reporting < 2 GB the slot count halves" taken to its limit.

use std::collections::HashMap;

/// D §6.3's slot count.
pub const SLOTS: usize = 32;

/// D §6.3's device budget for the resident set, in bytes.
pub const BUDGET: u64 = 400 * 1024 * 1024;

/// Which fragment a slot holds, and how big it is.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Slot<T> {
    fragment: u32,
    bytes: u64,
    /// The tick of the last [`SlotTable::touch`] or admission — the LRU key.
    used: u64,
    payload: T,
}

/// What one admission did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    /// The slot the fragment now occupies.
    pub slot: usize,
    /// The fragments that were evicted to make room, in eviction order.
    pub evicted: Vec<u32>,
    /// Whether the fragment was already resident (nothing was uploaded).
    pub hit: bool,
}

/// The resident set of D §6.3, generic over whatever the caller parks in a slot.
///
/// `T` is the device-side handle in production (buffers and bind groups) and a plain integer in
/// the tests, which is the whole reason it is a parameter.
#[derive(Debug)]
pub struct SlotTable<T> {
    slots: Vec<Option<Slot<T>>>,
    of_fragment: HashMap<u32, usize>,
    budget: u64,
    resident_bytes: u64,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<T> SlotTable<T> {
    /// D §6.3's table: 32 slots, 400 MB.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(SLOTS, BUDGET)
    }

    /// A table of `slots` slots and `budget` bytes — D §6.8's halved table on a small adapter, and
    /// what the tests use.
    #[must_use]
    pub fn with_capacity(slots: usize, budget: u64) -> Self {
        let slots = slots.max(1);
        Self {
            slots: (0..slots).map(|_| None).collect(),
            of_fragment: HashMap::new(),
            budget,
            resident_bytes: 0,
            tick: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// The slot holding `fragment`, marking it as most recently used.
    pub fn get(&mut self, fragment: u32) -> Option<usize> {
        let slot = *self.of_fragment.get(&fragment)?;
        self.touch(slot);
        Some(slot)
    }

    /// Admits `fragment` of `bytes`, uploading through `build` only on a miss.
    ///
    /// `None` when the fragment cannot fit even in an empty table: the caller runs that pair on
    /// the CPU rather than thrashing the device. Eviction is least-recently-used, one slot at a
    /// time, and stops as soon as both limits are satisfied.
    pub fn admit(
        &mut self,
        fragment: u32,
        bytes: u64,
        build: impl FnOnce() -> T,
    ) -> Option<Admission> {
        if let Some(&slot) = self.of_fragment.get(&fragment) {
            self.touch(slot);
            self.hits += 1;
            return Some(Admission { slot, evicted: Vec::new(), hit: true });
        }
        if bytes > self.budget {
            return None;
        }
        self.misses += 1;
        let mut evicted = Vec::new();
        while self.free_slot().is_none() || self.resident_bytes + bytes > self.budget {
            let victim = self.least_recently_used()?;
            evicted.push(self.evict(victim));
        }
        let slot = self.free_slot().expect("the loop above ends with a free slot");
        self.tick += 1;
        self.resident_bytes += bytes;
        self.of_fragment.insert(fragment, slot);
        self.slots[slot] = Some(Slot { fragment, bytes, used: self.tick, payload: build() });
        Some(Admission { slot, evicted, hit: false })
    }

    /// What is parked in a slot.
    pub fn payload(&self, slot: usize) -> Option<&T> {
        self.slots.get(slot)?.as_ref().map(|s| &s.payload)
    }

    /// Which fragment a slot holds.
    pub fn fragment_at(&self, slot: usize) -> Option<u32> {
        self.slots.get(slot)?.as_ref().map(|s| s.fragment)
    }

    /// Whether a fragment is resident, without touching it.
    #[must_use]
    pub fn is_resident(&self, fragment: u32) -> bool {
        self.of_fragment.contains_key(&fragment)
    }

    /// How many slots are in use.
    #[must_use]
    pub fn len(&self) -> usize {
        self.of_fragment.len()
    }

    /// Whether nothing is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.of_fragment.is_empty()
    }

    /// Bytes the resident set occupies on the device.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// `(hits, misses, evictions)` — the three numbers a run report carries about the slot table.
    #[must_use]
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    /// Drops everything, which is what a device loss or an epoch boundary does.
    pub fn clear(&mut self) {
        self.slots.iter_mut().for_each(|slot| *slot = None);
        self.of_fragment.clear();
        self.resident_bytes = 0;
    }

    /// Marks a slot as most recently used.
    fn touch(&mut self, slot: usize) {
        self.tick += 1;
        if let Some(Some(entry)) = self.slots.get_mut(slot) {
            entry.used = self.tick;
        }
    }

    fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_none)
    }

    /// The occupied slot with the smallest `used` tick; ties go to the lower slot index, which
    /// makes the eviction order a function of the request sequence alone (D §7).
    fn least_recently_used(&self) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|s| (s.used, i)))
            .min()
            .map(|(_, i)| i)
    }

    fn evict(&mut self, slot: usize) -> u32 {
        let entry = self.slots[slot].take().expect("only an occupied slot is evicted");
        self.of_fragment.remove(&entry.fragment);
        self.resident_bytes -= entry.bytes;
        self.evictions += 1;
        entry.fragment
    }
}

impl<T> Default for SlotTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{BUDGET, SLOTS, SlotTable};

    /// The default table is D §6.3's: 32 slots, 400 MB, and a fragment stays put until one of the
    /// two limits forces it out.
    #[test]
    fn the_table_is_thirty_two_slots_and_four_hundred_megabytes() {
        let mut table: SlotTable<u32> = SlotTable::new();
        assert!(table.is_empty());
        for f in 0..u32::try_from(SLOTS).expect("32 slots") {
            let admission = table.admit(f, 8 * 1024 * 1024, || f).expect("8 MB fits");
            assert!(!admission.hit && admission.evicted.is_empty(), "{admission:?}");
        }
        assert_eq!(table.len(), SLOTS);
        assert_eq!(table.resident_bytes(), 32 * 8 * 1024 * 1024);
        assert!(table.resident_bytes() < BUDGET);

        // The thirty-third eviction is the least recently used, which is fragment 0.
        let admission = table.admit(99, 8 * 1024 * 1024, || 99).expect("still fits");
        assert_eq!(admission.evicted, vec![0]);
        assert!(!table.is_resident(0) && table.is_resident(99));
        assert_eq!(table.payload(admission.slot), Some(&99));
        assert_eq!(table.fragment_at(admission.slot), Some(99));
    }

    /// The byte budget bites before the slot count when the fragments are large, and eviction
    /// stops as soon as the newcomer fits.
    #[test]
    fn the_budget_evicts_as_many_as_it_needs_and_no_more() {
        // Ten slots, 100 bytes: four fragments of 25 fill it on bytes, not on slots.
        let mut table: SlotTable<u32> = SlotTable::with_capacity(10, 100);
        for f in 0..4 {
            assert!(table.admit(f, 25, || f).is_some());
        }
        assert_eq!((table.len(), table.resident_bytes()), (4, 100));

        // A fifth of 25 evicts exactly one; a sixth of 50 evicts exactly two.
        let one = table.admit(4, 25, || 4).expect("one eviction is enough");
        assert_eq!(one.evicted, vec![0]);
        let two = table.admit(5, 50, || 5).expect("two evictions are enough");
        assert_eq!(two.evicted, vec![1, 2]);
        assert_eq!(table.resident_bytes(), 100);
        assert_eq!(table.len(), 3, "3, 4 and 5 are resident");
        assert!(table.is_resident(3) && table.is_resident(4) && table.is_resident(5));

        // A fragment larger than the whole budget is refused rather than emptying the table.
        assert!(table.admit(6, 101, || 6).is_none());
        assert_eq!(table.len(), 3, "and nothing was evicted for it");
    }

    /// A hit uploads nothing and makes the fragment the most recently used, which is what the
    /// 3×3 block schedule of D §5 is built to exploit.
    #[test]
    fn a_hit_uploads_nothing_and_renews_the_fragment() {
        let mut table: SlotTable<u32> = SlotTable::with_capacity(3, 1000);
        let mut built = 0;
        let admit = |table: &mut SlotTable<u32>, f: u32, built: &mut u32| {
            table.admit(f, 10, || {
                *built += 1;
                f
            })
        };
        admit(&mut table, 0, &mut built);
        admit(&mut table, 1, &mut built);
        admit(&mut table, 2, &mut built);
        assert_eq!(built, 3);

        // Ask for 0 again: a hit, no upload, and now 1 is the least recently used.
        let hit = admit(&mut table, 0, &mut built).expect("resident");
        assert!(hit.hit && hit.evicted.is_empty());
        assert_eq!(built, 3, "a hit builds nothing");
        let next = admit(&mut table, 3, &mut built).expect("room after one eviction");
        assert_eq!(next.evicted, vec![1], "0 was renewed, so 1 is the oldest");
        assert_eq!(table.stats(), (1, 4, 1));

        // `get` renews too, without admitting.
        assert!(table.get(2).is_some());
        assert!(table.get(1).is_none());
        let last = admit(&mut table, 4, &mut built).expect("room");
        assert_eq!(last.evicted, vec![0], "2 was renewed by `get`, so 0 is the oldest");

        table.clear();
        assert!(table.is_empty() && table.resident_bytes() == 0);
    }

    /// A one-slot table is legal and always evicts, and a zero-slot request is rounded up to one
    /// rather than deadlocking the admission loop.
    #[test]
    fn a_degenerate_table_still_answers() {
        let mut table: SlotTable<u32> = SlotTable::with_capacity(0, 1000);
        assert!(table.admit(1, 10, || 1).is_some());
        let second = table.admit(2, 10, || 2).expect("one slot, so it evicts");
        assert_eq!(second.evicted, vec![1]);
        assert_eq!(table.len(), 1);
    }
}
