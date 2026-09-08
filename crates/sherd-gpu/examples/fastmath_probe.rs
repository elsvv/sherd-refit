//! Does this adapter's compiler keep a compensated sum, or fold it away? (Task W §2, V6-D2.)
//!
//! Metal compiles every shader with fast math on and wgpu does not turn it off (E7 §4), which the
//! team accepted. Fast math is entitled to rewrite `(a + b) - a` into `b`, and every compensated
//! summation — Kahan, Neumaier, Knuth's two-sum, a two-float accumulator — is built out of exactly
//! that expression. Whether the compiler *takes* the licence decides whether a `f64`-emulated
//! accumulation inside `icp.wgsl` is possible at all, so it is measured here rather than assumed.
//!
//! One workgroup, one lane, six sums of the same 200 001 terms — `1.0` and 200 000 of `1e-8`,
//! whose exact sum is `1.002` and whose plain `f32` sum in order is `1.0` exactly, because every
//! term is below the running sum's ULP. A scheme that works reports ≈ 1.002; a scheme the compiler
//! folded reports 1.0 and a compensation of exactly zero.
//!
//! ```text
//! cargo run --release -p sherd-gpu --example fastmath_probe
//! ```
//!
//! Measured on Apple M2 Pro, Metal, wgpu 30.0.1 (task W): schemes 1–4 all report a compensation of
//! **exactly zero** — including the one whose split is forced through an integer bit mask, which
//! the compiler still cancels algebraically, and including (measured separately) a version routed
//! through workgroup storage and a barrier. Scheme 5, whose every subtraction is written
//! `fma(-1.0, b, a)`, recovers the whole residual: `fma` is an intrinsic the rewriter leaves alone
//! and `-1.0 · b` is exact, so `fma(-1.0, b, a)` is `a - b` correctly rounded and nothing else.
//! Scheme 6 is the single step beside it — the exact error of one product, which survives as it is
//! written, and the exact error of one sum, which agrees with the `fma` form where the optimiser
//! happens not to have folded it.
use sherd_gpu::buffers::Dispatch;
use sherd_gpu::{Gpu, buffers, device::AdapterChoice, shader::Kernel};

const SRC: &str = r"
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// The top 11 mantissa bits, through an integer mask — a value the algebraic rewriter has no
// identity for, and the strongest barrier available without touching memory.
fn mask_hi(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) & 0xffffe000u); }
// `a - b`, as the intrinsic.
fn xsub(a: f32, b: f32) -> f32 { return fma(-1.0, b, a); }

@compute @workgroup_size(1)
fn main() {
    let n = arrayLength(&data);

    // 0: the plain sum, in order.
    var plain = 0.0;
    for (var i = 0u; i < n; i = i + 1u) { plain = plain + data[i]; }
    out[0] = plain;

    // 1: Kahan, the form `icp.wgsl`'s `search()` carries.
    var s = 0.0; var c = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let t = data[i] - c; let nx = s + t; c = (nx - s) - t; s = nx;
    }
    out[1] = s; out[2] = c;

    // 2: Neumaier, which needs no magnitude assumption.
    var ns = 0.0; var nc = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let v = data[i]; let t = ns + v;
        if (abs(ns) >= abs(v)) { nc = nc + ((ns - t) + v); } else { nc = nc + ((v - t) + ns); }
        ns = t;
    }
    out[3] = ns + nc; out[4] = nc;

    // 3: Knuth's two-sum into a two-float accumulator.
    var hi = 0.0; var lo = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let v = data[i];
        let sm = hi + v;
        let bv = sm - hi;
        lo = lo + ((hi - (sm - bv)) + (v - bv));
        let t2 = sm + lo;
        lo = lo - (t2 - sm);
        hi = t2;
    }
    out[5] = hi; out[6] = lo;

    // 4: the same, with the split forced through the integer mask.
    var mh = 0.0; var ml = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let v = data[i];
        let sm = mh + v;
        let b = mask_hi(sm);
        let r = sm - b;
        ml = ml + (((mh - b) - r) + v);
        let t2 = sm + ml;
        let b2 = mask_hi(t2);
        ml = ml - ((t2 - b2) - (sm - b2));
        mh = t2;
    }
    out[7] = mh; out[8] = ml;

    // 5: the same, with every subtraction written as an `fma`.
    var fh = 0.0; var fl = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let v = data[i];
        let sm = fh + v;
        let bv = xsub(sm, fh);
        fl = fl + (xsub(fh, xsub(sm, bv)) + xsub(v, bv));
        let t2 = sm + fl;
        fl = xsub(fl, xsub(t2, sm));
        fh = t2;
    }
    out[9] = fh; out[10] = fl;

    // 6: the exact error of one product and of one sum, spelled out both ways. The two operands
    // are scaled by a runtime `1.0` so that nothing here is a compile-time constant.
    let a = 1.0000001 * data[0];
    let b = 3.0000002 * data[0];
    let pr = a * b;
    out[11] = fma(a, b, -pr);
    let sm = a + b;
    out[12] = (sm - a) - b;
    out[13] = xsub(xsub(sm, a), b);
}
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = Gpu::open(&AdapterChoice::Default)?;
    println!("adapter: {}", gpu.entry());

    let mut data = vec![1.0_f32];
    data.extend(std::iter::repeat_n(1e-8_f32, 200_000));
    let exact: f64 = data.iter().map(|&x| f64::from(x)).sum();

    let kernel = Kernel::build(&gpu, "fastmath probe", SRC, "main", 0, &[true, false]);
    let input = buffers::upload(&gpu, "data", &data);
    let out = buffers::output(&gpu, "out", 14 * 4);
    let bind = kernel.bind(&gpu, &[input.as_entire_binding(), out.as_entire_binding()]);
    kernel.dispatch(&gpu, &bind, Dispatch::for_workgroups(1))?;
    let got: Vec<f32> = buffers::read_back(&gpu, "fastmath probe", &out, 14)?;

    println!("  exact, in f64          {exact:.10}");
    println!("0 plain f32              {:.10}", got[0]);
    println!("1 kahan                  {:.10}   c   = {:e}", got[1], got[2]);
    println!("2 neumaier               {:.10}   c   = {:e}", got[3], got[4]);
    println!("3 two-sum, two-float     {:.10}   lo  = {:e}", got[5], got[6]);
    println!("4 masked split           {:.10}   lo  = {:e}", got[7], got[8]);
    println!("5 fma-written two-sum    {:.10}   lo  = {:e}", got[9] + got[10], got[10]);
    println!(
        "6 one step: product error {:e}, sum error plain {:e}, sum error as fma {:e}",
        got[11], got[12], got[13],
    );
    Ok(())
}
