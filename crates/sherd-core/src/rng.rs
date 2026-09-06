//! The pipeline's random numbers (D §3, D §7).
//!
//! Every draw the reference makes is listed in R §10; each one is seeded explicitly, from
//! `Params::seed` or from a hard-coded 0 where the reference hard-codes it, so a run is
//! reproducible on any platform.
//!
//! The generator is `ChaCha8Rng`, whose stream is guaranteed across versions and architectures.
//! It is **not** numpy's PCG64: the sequence of samples differs from the reference's, so the
//! sampled point sets differ point by point (PMC-9, D §13.2). Parity is therefore statistical in
//! native mode — the tolerances of D §10.2 — and exact only in injected mode, where the sampled
//! arrays come from the Python fixture.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

/// The generator for a draw with the given seed.
///
/// Call it once per draw site, with the seed R §10 prescribes; never thread one generator
/// through two stages, or the parallel schedule would change the numbers.
#[inline]
pub fn seeded(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::seeded;
    use rand_chacha::rand_core::Rng;

    fn first_words(seed: u64) -> [u32; 4] {
        let mut r = seeded(seed);
        [r.next_u32(), r.next_u32(), r.next_u32(), r.next_u32()]
    }

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        assert_eq!(first_words(0), first_words(0));
        assert_eq!(first_words(12345), first_words(12345));
    }

    #[test]
    fn different_seeds_give_different_streams() {
        assert_ne!(first_words(0), first_words(1));
    }

    #[test]
    fn the_stream_is_the_documented_one() {
        // ChaCha8Rng::seed_from_u64(0), first four u32 words: pinned here so that a crate update
        // that changes the stream cannot pass unnoticed — it would change every native run.
        assert_eq!(first_words(0), [0xa79a_3b6c, 0xb585_f767, 0xbad8_c037, 0x7746_a55f]);
    }
}
