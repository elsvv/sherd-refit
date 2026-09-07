//! D §5's shared `MatchData` cache: one entry per `(fragment, t, reg_points)`.
//!
//! R §4.2 builds a pair from both fragments' arrays *at the pair's own* `t_pair`, and D §5 asks
//! for a `Mutex<LruCache<(FragId, t_bits), Arc<MatchData>>>` of 64 entries "so a fragment
//! recomputed at `t_pair` for one pair serves the next". [`MatchCache`] is that cache, and the
//! block schedule of [`pipeline`](crate::pipeline) is what feeds it: a job is all the pairs
//! between two blocks of three fragments, so the six fragments a job touches are asked for
//! repeatedly.
//!
//! # What it can and cannot remove
//!
//! R §1.2 sets `t_pair = min(t_A, t_B)`. Of a pair's two builds exactly one is the fragment that
//! *is* the minimum — `MatchData::at` borrows its cached arrays and rebuilds nothing — and the
//! other is a rebuild of R §3.5 at a thickness that belongs to that one partner. Wall thicknesses
//! are `f64` measurements, so every *expensive* key is unique and the cache can never hit on one:
//! E1 counted 190 of 190 distinct rebuild keys on synthetic 20 and 55 of 55 on pot H
//! (`notes/2026-09-07-e1-profile.md` §5). What repeats is the cheap half — the clouds, the two
//! KD-trees and the `f64` widenings every build pays whether or not it redrew the arrays — which
//! E1 bounds at 0.20 % of the run. The cache is therefore correctness-neutral and cheap, and it
//! is worth what §5 of that note says it is worth until R §4.2's rebuild is made partially
//! `t`-independent.
//!
//! # Why it cannot move a result
//!
//! `MatchData::at` is a pure function of `(fragment, t, reg_points)` — no clock, no thread id, no
//! RNG that is not seeded from `Params` — so a hit is bit-identical to the build it replaces.
//! Two threads that miss on the same key at the same time both build; both results are the same
//! bits, and the loser is simply dropped. Nothing below this module can tell whether it was
//! served from the cache, which is why the gate for it is byte-identical outputs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::fragment::Fragment;
use crate::fragment::samples::MatchData;
use crate::types::FragId;

/// D §8's entry count: 64 `MatchData`, which that table prices at ≈ 400 MB for 10 M-face scans.
///
/// A job of the block schedule touches at most `2 · BLOCK = 6` fragments and asks for at most two
/// keys per pair, so 64 holds several jobs' working sets at once on this machine's pool.
pub const CAPACITY: usize = 64;

/// Everything [`MatchData::at`] is a function of.
///
/// D §5 names `(FragId, t_bits)`. The fragment's *address* is carried beside its id because
/// `Fragment::from_mesh_file` leaves `id` at zero — only the collection loader assigns it — and a
/// cache keyed on a field that two live fragments may share would serve one fragment's arrays for
/// another. Two distinct fragments cannot share an address while both are borrowed, so the pair
/// is an identity whatever built them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    /// The fragment's index in the collection (R §2), for readability in a dump.
    id: FragId,
    /// The borrowed fragment's address, which is what actually identifies it.
    addr: usize,
    /// `t.to_bits()`: the wall thickness the arrays are built at, compared as bits so that the
    /// key is an equality on the `f64` and not a tolerance.
    t: u64,
    /// R §1.1's `reg_points`, which changes `pc_reg` and nothing else.
    reg_points: usize,
}

/// What the cache did over a run, for the log line and the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Builds served from an entry that was already there.
    pub hits: u64,
    /// Builds that had to run [`MatchData::at`].
    pub misses: u64,
    /// Entries dropped to keep the cache at [`CAPACITY`].
    pub evictions: u64,
}

impl CacheStats {
    /// Hits over lookups, or `0.0` when nothing was looked up.
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "a count of pairs, far under 2^53")]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// The cache itself: [`CAPACITY`] entries behind one mutex, least-recently-used first out.
///
/// The lifetime is the fragments': a `MatchData` borrows the fragment it describes (and, at the
/// fragment's own `t`, its arrays), so the cache lives inside the matching stage and dies with it.
#[derive(Debug)]
pub struct MatchCache<'a> {
    slots: Mutex<Slots<'a>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// The mutable half, which is only ever touched under the mutex.
#[derive(Debug)]
struct Slots<'a> {
    capacity: usize,
    /// Monotone stamp; the entry with the smallest one is the least recently used.
    clock: u64,
    evictions: u64,
    entries: Vec<Entry<'a>>,
}

#[derive(Debug)]
struct Entry<'a> {
    key: Key,
    used: u64,
    data: Arc<MatchData<'a>>,
}

impl<'a> MatchCache<'a> {
    /// A cache of `capacity` entries; `0` disables it (every lookup is a miss and nothing is
    /// stored), which is what the no-cache paths want without a second code path.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Mutex::new(Slots { capacity, clock: 0, evictions: 0, entries: Vec::new() }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// D §8's cache: [`CAPACITY`] entries.
    #[must_use]
    pub fn sized() -> Self {
        Self::new(CAPACITY)
    }

    /// R §4.2's arrays for `fragment` at `t`, from the cache when they are there.
    ///
    /// The lock is taken twice and never held across the build: a miss releases it, runs
    /// [`MatchData::at`], and takes it again to insert. Two threads missing on one key therefore
    /// both build — the same bits twice, since the build is a pure function of the key — which is
    /// the trade that keeps one slow build from serialising the pool.
    pub fn get_or_build(
        &self,
        fragment: &'a Fragment,
        t: f64,
        reg_points: usize,
    ) -> Arc<MatchData<'a>> {
        let key = Key {
            id: fragment.id,
            addr: std::ptr::from_ref(fragment) as usize,
            t: t.to_bits(),
            reg_points,
        };
        if let Some(hit) = self.lookup(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return hit;
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let built = Arc::new(MatchData::at(fragment, t, reg_points));
        self.insert(key, &built);
        built
    }

    /// What the cache did so far.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let evictions = self.slots.lock().map_or(0, |s| s.evictions);
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions,
        }
    }

    fn lookup(&self, key: Key) -> Option<Arc<MatchData<'a>>> {
        let mut slots = self.slots.lock().ok()?;
        let at = slots.entries.iter().position(|e| e.key == key)?;
        slots.clock += 1;
        let stamp = slots.clock;
        slots.entries[at].used = stamp;
        Some(Arc::clone(&slots.entries[at].data))
    }

    fn insert(&self, key: Key, data: &Arc<MatchData<'a>>) {
        let Ok(mut slots) = self.slots.lock() else { return };
        if slots.capacity == 0 {
            return;
        }
        slots.clock += 1;
        let stamp = slots.clock;
        if let Some(at) = slots.entries.iter().position(|e| e.key == key) {
            // Another thread got there first with the same bits; keep its `Arc` so that a third
            // thread's outstanding handle stays the one everyone shares.
            slots.entries[at].used = stamp;
            return;
        }
        while slots.entries.len() >= slots.capacity {
            let Some(oldest) =
                slots.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i)
            else {
                break;
            };
            slots.entries.swap_remove(oldest);
            slots.evictions += 1;
        }
        slots.entries.push(Entry { key, used: stamp, data: Arc::clone(data) });
    }
}

impl Default for MatchCache<'_> {
    fn default() -> Self {
        Self::sized()
    }
}

#[cfg(test)]
mod tests {
    use super::{CAPACITY, CacheStats, MatchCache};
    use crate::fragment::Fragment;
    use crate::fragment::samples::REG_POINTS;

    fn slab() -> Fragment {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/slab/input/pieceA.ply");
        Fragment::from_mesh_file(path, 200_000).expect("the committed slab fixture")
    }

    /// The same key twice is one build, and the second call is the *same* `Arc`.
    #[test]
    fn a_repeated_key_is_served_from_the_cache() {
        let fragment = slab();
        let cache = MatchCache::sized();
        let reg = REG_POINTS as usize;
        let first = cache.get_or_build(&fragment, fragment.thick, reg);
        let again = cache.get_or_build(&fragment, fragment.thick, reg);
        assert!(std::sync::Arc::ptr_eq(&first, &again), "one build, one Arc");
        assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 1, evictions: 0 });

        // A different `t` is a different entry, and it is a different answer.
        let other = cache.get_or_build(&fragment, fragment.thick * 0.9, reg);
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 1);
        assert!(
            (other.t - fragment.thick * 0.9).abs() < 1e-15,
            "the entry is the arrays at the key's own t"
        );
    }

    /// A hit is bit-identical to the build it replaces — the property the byte-identity gate rests
    /// on.
    #[test]
    fn a_hit_is_the_build_it_replaces() {
        let fragment = slab();
        let cache = MatchCache::sized();
        let reg = REG_POINTS as usize;
        let t = fragment.thick * 0.93;
        let cached = cache.get_or_build(&fragment, t, reg);
        let fresh = crate::fragment::samples::MatchData::at(&fragment, t, reg);
        assert_eq!(cached.brk.p, fresh.brk.p);
        assert_eq!(cached.samples.pf, fresh.samples.pf);
        assert_eq!(cached.pc_reg.p, fresh.pc_reg.p);
        assert_eq!(cached.brk_dih, fresh.brk_dih);
        assert_eq!(cached.margin.p, fresh.margin.p);
    }

    /// Capacity is a bound: the cache never holds more than it was sized for, and the entry it
    /// drops is the one used longest ago.
    #[test]
    fn the_least_recently_used_entry_is_the_one_that_goes() {
        let fragment = slab();
        let cache = MatchCache::new(2);
        let reg = REG_POINTS as usize;
        let a = cache.get_or_build(&fragment, fragment.thick, reg);
        let _b = cache.get_or_build(&fragment, fragment.thick * 0.9, reg);
        // Touch `a` so that the 0.9 entry becomes the oldest, then overflow.
        let a_again = cache.get_or_build(&fragment, fragment.thick, reg);
        assert!(std::sync::Arc::ptr_eq(&a, &a_again));
        let _c = cache.get_or_build(&fragment, fragment.thick * 0.8, reg);
        assert_eq!(cache.stats().evictions, 1);

        let a_third = cache.get_or_build(&fragment, fragment.thick, reg);
        assert!(std::sync::Arc::ptr_eq(&a, &a_third), "the touched entry survived");
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 3);
        assert!((stats.hit_rate() - 0.4).abs() < 1e-12, "{}", stats.hit_rate());
    }

    /// A cache of nothing is a build every time, and it stores nothing.
    #[test]
    fn a_capacity_of_zero_is_no_cache_at_all() {
        let fragment = slab();
        let cache = MatchCache::new(0);
        let reg = REG_POINTS as usize;
        let first = cache.get_or_build(&fragment, fragment.thick, reg);
        let again = cache.get_or_build(&fragment, fragment.thick, reg);
        assert!(!std::sync::Arc::ptr_eq(&first, &again));
        assert_eq!(cache.stats(), CacheStats { hits: 0, misses: 2, evictions: 0 });
        assert!(cache.stats().hit_rate().abs() < 1e-12);
        assert!(CacheStats::default().hit_rate().abs() < 1e-12, "no lookups, no rate");
        assert_eq!(CAPACITY, 64);
    }
}
