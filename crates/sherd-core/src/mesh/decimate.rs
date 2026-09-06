//! Quadric decimation to the adaptive face budget (R §3.3).
//!
//! ```text
//! target = clip( 600 · ΣA0 / t², 50000, target_faces )
//! if n_faces(F0) > target:  decimate; remove degenerate, duplicated, unreferenced
//! ```
//!
//! The budget is what keeps the working mesh at roughly twelve edges across the wall whatever the
//! scanner's resolution was: `600` faces per `t²` of surface, floored at 50 000 so a small
//! fragment is not starved, capped at the CLI's `--target-faces` so a huge one cannot blow the
//! memory budget. It is computed from the *original* component's area and from the thickness
//! measured on that same mesh (R §3.2), before anything is decimated.
//!
//! **The decimator is `meshopt`, not Open3D (PMC-2).** Experiment E1
//! (`docs/superpowers/notes/2026-09-06-e1-decimation.md`) measured Open3D's
//! `simplify_quadric_decimation` against `meshopt` and `baby_shark` on fourteen benchmark meshes:
//! `meshopt` with `SimplifyOptions::Regularize` hits the face target to within one triangle,
//! reproduces Open3D's `closed_enough` verdict on 14 of 14, keeps `res` inside ±10 % and the
//! segmentation agreement at 0.980–0.993 (gate ≥ 0.97), and does it twenty times faster (1.73 s
//! against 34.70 s). `Regularize` is what makes it pass — plain `meshopt` misses the `res` gate on
//! 13 of 14 meshes, because it sizes triangles adaptively and leaves the flat shell coarse.
//!
//! `LockBorder`, `ErrorAbsolute` and `Prune` stay off, and `target_error` is `1e9`, i.e. no error
//! cap, which is what Open3D does.
//!
//! The collapse order is not Open3D's, so the working mesh is *not* the reference's mesh and never
//! will be: native-mode parity for this stage is statistical (D §10.2, D §13.1). What `meshopt`
//! does give is that **no vertex ever moves** — it returns an index buffer over the original
//! vertices — so the `f64` coordinates the readers produced survive decimation exactly, where
//! Open3D and `baby_shark` both recompute positions from the quadric.

use meshopt::{SimplifyOptions, VertexDataAdapter};

use super::Mesh;
use super::clean::{
    remove_degenerate_triangles, remove_duplicated_vertices, remove_unreferenced_vertices,
};

/// Working-mesh faces per `t²` of surface — about twelve edges across the wall.
pub const FACES_PER_T2: f64 = 600.0;
/// The floor of the budget: a small fragment still gets 50 000 faces.
pub const MIN_FACES: usize = 50_000;
/// `target_error` handed to `meshopt`: large enough that the face count is always the binding
/// constraint, which is how Open3D's decimator behaves.
const NO_ERROR_CAP: f32 = 1e9;

/// R §3.3's adaptive face budget: `int(clip(600 · area / t², 50000, target_faces))`.
///
/// numpy's `clip` applies the lower bound first and the upper bound second, so a `target_faces`
/// below [`MIN_FACES`] wins over the floor — asking for 20 000 faces gives 20 000, not 50 000.
/// The conversion to an integer truncates, as Python's `int()` does.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the value is clamped into 0..=target_faces before the cast, as int(np.clip(...)) is"
)]
pub fn face_budget(total_area: f64, thick: f64, target_faces: usize) -> usize {
    let raw = FACES_PER_T2 * total_area / (thick * thick);
    let clipped = raw.max(MIN_FACES as f64).min(target_faces as f64);
    clipped as usize
}

/// Decimates the mesh to `target` triangles when it has more, and cleans up after itself.
///
/// Returns `true` when the decimator ran. The guard is the reference's: at or below the budget the
/// mesh is handed on untouched, which matters because `meshopt` would still weld a few vertices
/// and `baby_shark` would collapse two hundred edges.
///
/// After a collapse pass the cleanup runs in R §3.3's order — degenerate triangles, duplicated
/// vertices, unreferenced vertices — which is *not* the order R §3.1 uses on load.
pub fn decimate(m: &mut Mesh, target: usize) -> bool {
    if m.f.len() <= target {
        return false;
    }
    m.f = simplify(&m.v, &m.f, target);
    remove_degenerate_triangles(m);
    remove_duplicated_vertices(m);
    remove_unreferenced_vertices(m);
    true
}

/// The `meshopt` call itself: collapses edges until the face count reaches `target`.
///
/// The returned triangles index the *input* vertices; no vertex moves and none is added, so the
/// caller keeps its `f64` coordinates and only has to drop what is no longer referenced.
pub fn simplify(v: &[[f64; 3]], f: &[[u32; 3]], target: usize) -> Vec<[u32; 3]> {
    #[allow(clippy::cast_possible_truncation, reason = "meshopt reads positions as f32 by design")]
    let positions: Vec<f32> =
        v.iter().flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0)
        .expect("3 f32 per vertex is a stride-12 offset-0 buffer");
    let indices = meshopt::simplify(
        f.as_flattened(),
        &adapter,
        target * 3,
        NO_ERROR_CAP,
        SimplifyOptions::Regularize,
        None,
    );
    indices.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

#[cfg(test)]
mod tests {
    use super::{FACES_PER_T2, MIN_FACES, decimate, face_budget, simplify};
    use crate::mesh::Mesh;
    use std::collections::HashSet;

    #[test]
    fn the_budget_is_clipped_at_both_ends_and_truncated() {
        // 600 · 1000 / 2² = 150 000, inside the window.
        assert_eq!(face_budget(1000.0, 2.0, 200_000), 150_000);
        // Below the floor.
        assert_eq!(face_budget(1.0, 2.0, 200_000), MIN_FACES);
        // Above the cap.
        assert_eq!(face_budget(1e6, 2.0, 200_000), 200_000);
        // numpy's clip applies the floor first, so a cap below the floor still wins.
        assert_eq!(face_budget(1.0, 2.0, 20_000), 20_000);
        // int() truncates rather than rounds: 600 · 100.9998 / 1 = 60599.88 -> 60599.
        assert_eq!(face_budget(100.9998, 1.0, 200_000), 60_599);
        // A thin wall asks for more faces than the cap allows.
        assert_eq!(face_budget(1000.0, 0.1, 200_000), 200_000);
        assert!((FACES_PER_T2 - 600.0).abs() < f64::EPSILON);
    }

    /// A grid of `n × n` quads on the unit square, split into triangles: 2·(n−1)² faces.
    fn grid(n: usize) -> Mesh {
        let mut v = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                #[allow(clippy::cast_precision_loss, reason = "small grid")]
                v.push([i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64, 0.0]);
            }
        }
        let mut f = Vec::new();
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                let idx = |a: usize, b: usize| u32::try_from(a * n + b).unwrap();
                f.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                f.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        Mesh::new(v, f)
    }

    #[test]
    fn a_mesh_at_or_below_the_budget_is_left_exactly_alone() {
        let m0 = grid(20); // 722 faces
        let mut m = m0.clone();
        assert!(!decimate(&mut m, m0.f.len()), "equal to the target is not above it");
        assert_eq!(m, m0);
        assert!(!decimate(&mut m, 10_000));
        assert_eq!(m, m0);
    }

    #[test]
    fn decimation_reaches_the_target_without_moving_a_vertex() {
        let m0 = grid(60); // 6962 faces
        let mut m = m0.clone();
        assert!(decimate(&mut m, 2000));
        assert!(
            m.f.len() <= 2000 && m.f.len() >= 1900,
            "meshopt reached {} faces, target 2000",
            m.f.len()
        );

        // Every surviving vertex is bit-identical to one of the input vertices: `meshopt` returns
        // an index buffer and never recomputes a position.
        let original: HashSet<[u64; 3]> =
            m0.v.iter().map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]).collect();
        for p in &m.v {
            assert!(
                original.contains(&[p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]),
                "decimation moved a vertex to {p:?}"
            );
        }
        assert!(m.v.len() < m0.v.len(), "unreferenced vertices must be dropped");
        m.validate(std::path::Path::new("grid")).expect("indices stay in range");
    }

    #[test]
    fn simplify_is_deterministic() {
        let m = grid(60);
        let a = simplify(&m.v, &m.f, 2000);
        let b = simplify(&m.v, &m.f, 2000);
        assert_eq!(a, b);
    }
}
