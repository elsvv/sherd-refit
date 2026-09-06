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
//!
//! # One stream per (fragment, draw)
//!
//! The reference builds **one** generator per fragment (`rng_md = rng(p.seed)`) and consumes it
//! through R §3.5's three samplers in order, so the fracture samples depend on how many numbers
//! the surface samples took. This port gives every draw site its own stream instead
//! ([`seeded_for`]), because R §4.2 and R §8 rebuild the match arrays at another `t` and another
//! `surface_points`, and with one shared stream a change to the *first* sampler silently moves the
//! other two. A generator is still built per fragment and never shared between fragments, which
//! is the reference's structure; what is added is the separation *between* the draws.
//!
//! The separation cannot be "the same seed three times": that would hand the surface sampler and
//! the fracture sampler the identical stream of uniforms, and a surface sample and a fracture
//! sample landing on the same face would then land on the *same point of it*. Each [`Draw`]
//! therefore carries a fixed 64-bit tag which is exclusive-or'd into the seed, and
//! `ChaCha8Rng::seed_from_u64` runs its own SplitMix64 over the result, so two draws of the same
//! run are as independent as two seeds are.
//!
//! [`Draw::Thickness`]'s tag is zero, so R §3.2's stream is exactly the one steps B1 and B2
//! measured; the reference hard-codes its seed to 0 anyway (R §10).

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// The draw sites of R §10, each with its own stream.
///
/// The tag is what separates the streams and it is part of the port's results: changing one
/// changes every natively sampled array that draw produces. New draw sites append a new variant
/// with a new tag and never renumber an existing one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Draw {
    /// R §3.2's inward rays: `choice(n_faces, 20000, replace=False)`. Tagged zero, so this stream
    /// is `seeded(0)` itself — the reference hard-codes the seed here, and steps B1 and B2
    /// measured every native number in the tree against exactly this stream.
    Thickness,
    /// R §3.5.1's whole-surface samples: `n` uniforms for the face pick, then `n` for `u`, then
    /// `n` for `v`.
    Surface,
    /// R §3.5.2's fracture samples, drawn the same way.
    Fracture,
    /// R §3.5.6's thinning of the shell margin: one `choice(margin, margin_points,
    /// replace=False)`, and only when the margin is larger than that.
    Margin,
}

impl Draw {
    /// The 64-bit tag mixed into the seed.
    ///
    /// The three sampling tags are the FNV-1a hash of the variant's name, which is a fixed
    /// arbitrary constant chosen by a rule rather than by taste; the value is what matters, not
    /// the rule, and it is pinned by a test.
    pub const fn tag(self) -> u64 {
        match self {
            Self::Thickness => 0,
            Self::Surface => fnv1a(b"surface"),
            Self::Fracture => fnv1a(b"fracture"),
            Self::Margin => fnv1a(b"margin"),
        }
    }
}

/// FNV-1a over a byte string, as a `const fn` so the tags are compile-time constants.
const fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    hash
}

/// The generator for a draw with the given seed.
///
/// Call it once per draw site, with the seed R §10 prescribes; never thread one generator
/// through two stages, or the parallel schedule would change the numbers.
#[inline]
pub fn seeded(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// The generator for one draw site of a run seeded with `seed` (see the module documentation).
#[inline]
pub fn seeded_for(seed: u64, draw: Draw) -> ChaCha8Rng {
    seeded(seed ^ draw.tag())
}

/// A uniform double in `[0, 1)`, built from the generator's next 64-bit word exactly as numpy
/// builds `Generator.random()`: `(word >> 11) · 2⁻⁵³`.
///
/// The word stream is ChaCha8's rather than PCG64's (PMC-9), but the *function of the word* is
/// numpy's own `random_standard_double`, so the draw has the same 53-bit resolution and the same
/// half-open range, and nothing downstream has to know which generator produced it.
#[inline]
pub fn unit_f64(rng: &mut ChaCha8Rng) -> f64 {
    #[allow(clippy::cast_precision_loss, reason = "53 bits is exactly what an f64 mantissa holds")]
    {
        (rng.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// `rng.choice(n, take, replace=False)`: `min(take, n)` distinct indices below `n`, uniformly,
/// in draw order.
///
/// A partial Fisher–Yates shuffle, so the draw is `O(take)` swaps over a pool of `n` and needs no
/// hash set (D §7 keeps unordered containers off every result path). The *sequence* is not
/// numpy's — this is `ChaCha8Rng`, PMC-9 — so in native mode the drawn set differs from the
/// reference's and the comparison is statistical.
pub fn without_replacement(n: usize, take: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let take = take.min(n);
    let mut pool: Vec<u32> = (0..u32::try_from(n).expect("the population fits in u32")).collect();
    for i in 0..take {
        let span = u32::try_from(n - i).expect("the population fits in u32");
        let j = i + uniform_below(rng, span) as usize;
        pool.swap(i, j);
    }
    pool.truncate(take);
    pool
}

/// A uniform integer in `0..n`, by Lemire's multiply-shift with rejection (unbiased).
pub fn uniform_below(rng: &mut ChaCha8Rng, n: u32) -> u32 {
    debug_assert!(n > 0);
    let mut m = u64::from(rng.next_u32()) * u64::from(n);
    #[allow(clippy::cast_possible_truncation, reason = "the low word is the fractional part")]
    let mut low = m as u32;
    if low < n {
        let threshold = n.wrapping_neg() % n;
        while low < threshold {
            m = u64::from(rng.next_u32()) * u64::from(n);
            #[allow(clippy::cast_possible_truncation, reason = "the low word is the fraction")]
            {
                low = m as u32;
            }
        }
    }
    #[allow(clippy::cast_possible_truncation, reason = "the high word is below n")]
    {
        (m >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::{Draw, seeded, seeded_for, uniform_below, unit_f64, without_replacement};
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

    /// The tags are results, not conveniences: pinned, distinct, and zero for the wall so that
    /// R §3.2's stream is the one steps B1 and B2 measured.
    #[test]
    fn the_draw_tags_are_pinned_and_distinct() {
        assert_eq!(Draw::Thickness.tag(), 0);
        assert_eq!(Draw::Surface.tag(), 0x1826_0d59_cf7e_151c);
        assert_eq!(Draw::Fracture.tag(), 0x5f49_d730_66d9_516b);
        assert_eq!(Draw::Margin.tag(), 0x56b5_a72b_50ec_d75b);

        let tags = [Draw::Thickness, Draw::Surface, Draw::Fracture, Draw::Margin].map(Draw::tag);
        for i in 0..tags.len() {
            for j in (i + 1)..tags.len() {
                assert_ne!(tags[i], tags[j], "tags {i} and {j}");
            }
        }
        assert_eq!(first_words(0), {
            let mut r = seeded_for(0, Draw::Thickness);
            [r.next_u32(), r.next_u32(), r.next_u32(), r.next_u32()]
        });
    }

    /// Two draws of the same run must not hand out the same numbers — which is the whole reason
    /// the tags exist (see the module documentation).
    #[test]
    fn two_draws_of_one_run_are_independent() {
        let mut surface = seeded_for(0, Draw::Surface);
        let mut fracture = seeded_for(0, Draw::Fracture);
        let mut margin = seeded_for(0, Draw::Margin);
        for _ in 0..64 {
            let draws = [unit_f64(&mut surface), unit_f64(&mut fracture), unit_f64(&mut margin)];
            let mut bits: Vec<u64> = draws.iter().map(|u| u.to_bits()).collect();
            bits.sort_unstable();
            bits.dedup();
            assert_eq!(bits.len(), 3, "{draws:?}");
        }
    }

    /// How many uniforms the resolution test draws.
    const N: usize = 100_000;

    /// numpy's `random()` in range and resolution: `[0, 1)` with a 53-bit mantissa.
    #[test]
    fn the_uniform_draw_is_numpys_construction() {
        let mut rng = seeded(0);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for _ in 0..N {
            let u = unit_f64(&mut rng);
            assert!((0.0..1.0).contains(&u), "{u}");
            // 53 bits: every draw is an exact multiple of 2^-53.
            assert!((u * 9_007_199_254_740_992.0).fract().abs() < f64::EPSILON, "{u}");
            lo = lo.min(u);
            hi = hi.max(u);
            sum += u;
        }
        #[allow(clippy::cast_precision_loss, reason = "a sample count")]
        let mean = sum / N as f64;
        assert!((mean - 0.5).abs() < 0.01, "{mean}");
        assert!(lo < 0.001 && hi > 0.999, "{lo} {hi}");
    }

    #[test]
    fn a_draw_without_replacement_is_distinct_capped_and_seeded() {
        let mut rng = seeded(0);
        let a = without_replacement(1000, 20_000, &mut rng);
        assert_eq!(a.len(), 1000, "more asked for than there are");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 1000, "no index twice");

        let mut rng = seeded(0);
        let b = without_replacement(1000, 100, &mut rng);
        assert_eq!(b.len(), 100);
        assert_eq!(b, a[..100], "a prefix of the same shuffle");

        let mut rng = seeded(1);
        assert_ne!(without_replacement(1000, 100, &mut rng), b, "another seed, another draw");
        assert!(without_replacement(0, 10, &mut seeded(0)).is_empty());
        assert!(without_replacement(10, 0, &mut seeded(0)).is_empty());
    }

    #[test]
    fn the_integer_draw_is_uniform_and_in_range() {
        let mut rng = seeded(7);
        let mut counts = [0_u32; 5];
        for _ in 0..50_000 {
            let k = uniform_below(&mut rng, 5);
            assert!(k < 5);
            counts[k as usize] += 1;
        }
        assert!(counts.iter().all(|&c| (9_000..11_000).contains(&c)), "{counts:?}");
        assert_eq!(uniform_below(&mut rng, 1), 0);
    }
}
