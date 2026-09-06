//! Cleaning a freshly read mesh (R §3.1 step 2).
//!
//! Three passes, in the reference's order, each one Open3D's:
//!
//! 1. [`remove_duplicated_vertices`] — merge vertices whose coordinates are *exactly* equal;
//! 2. [`remove_degenerate_triangles`] — drop triangles with a repeated index (the merge in step 1
//!    creates them);
//! 3. [`remove_unreferenced_vertices`] — drop vertices no surviving triangle names.
//!
//! Order matters, because each pass changes what the next one sees, and the counts recorded as
//! `n_orig_vertices` / `n_orig_faces` (R §3.1 step 3) are the ones after all three.
//!
//! Every pass keeps the surviving elements in their original relative order and keeps the *first*
//! of a set of duplicates, which is what makes the port's vertex and face order the reference's.
//! The reference never removes duplicate *triangles* — `remove_duplicated_triangles` appears
//! nowhere in `sherd_refit` — so neither does this module.
//!
//! Filled in by plan step S2.

use std::collections::HashMap;

use super::Mesh;

/// Merges vertices with exactly equal coordinates, keeping the first of each set, and returns how
/// many vertices went away.
///
/// "Exactly" is Open3D's `std::tuple<double, double, double>` equality, and this reproduces its
/// two edge cases: `-0.0` merges with `0.0` (they compare equal), and a vertex with a `NaN`
/// coordinate merges with nothing, not even a bit-identical copy of itself (`NaN != NaN`).
///
/// Colours follow the vertex that survives.
pub fn remove_duplicated_vertices(m: &mut Mesh) -> usize {
    let n_old = m.v.len();
    let mut first_at: HashMap<[u64; 3], u32> = HashMap::with_capacity(n_old);
    let mut old_to_new = vec![0_u32; n_old];
    let mut k = 0_usize;
    for i in 0..n_old {
        let p = m.v[i];
        // A NaN coordinate makes the tuple unequal to every other, this one included.
        let new_index = if p[0].is_nan() || p[1].is_nan() || p[2].is_nan() {
            None
        } else {
            let key = [key_bits(p[0]), key_bits(p[1]), key_bits(p[2])];
            match first_at.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => Some(*e.get()),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(u32::try_from(k).expect("vertex count fits in u32"));
                    None
                }
            }
        };
        if let Some(j) = new_index {
            old_to_new[i] = j;
        } else {
            m.v[k] = p;
            if let Some(c) = &mut m.colors {
                c[k] = c[i];
            }
            old_to_new[i] = u32::try_from(k).expect("vertex count fits in u32");
            k += 1;
        }
    }
    if k == n_old {
        return 0;
    }
    m.v.truncate(k);
    if let Some(c) = &mut m.colors {
        c.truncate(k);
    }
    for t in &mut m.f {
        for idx in t {
            *idx = old_to_new[*idx as usize];
        }
    }
    n_old - k
}

/// `-0.0` and `0.0` compare equal as doubles, so they must hash the same.
#[inline]
fn key_bits(x: f64) -> u64 {
    if x == 0.0 { 0.0_f64.to_bits() } else { x.to_bits() }
}

/// Drops triangles that name the same vertex twice, and returns how many went away.
///
/// This is Open3D's `RemoveDegenerateTriangles`: a purely topological test on the indices, run
/// *after* the vertex merge, which is what turns a sliver into a degenerate triangle in the first
/// place. A triangle of zero area whose three indices are distinct survives, exactly as it does in
/// the reference.
pub fn remove_degenerate_triangles(m: &mut Mesh) -> usize {
    let n_old = m.f.len();
    m.f.retain(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0]);
    n_old - m.f.len()
}

/// Drops vertices no triangle refers to, and returns how many went away.
///
/// Order is preserved and the triangles are remapped; colours follow their vertex.
pub fn remove_unreferenced_vertices(m: &mut Mesh) -> usize {
    let n_old = m.v.len();
    let mut referenced = vec![false; n_old];
    for t in &m.f {
        for &i in t {
            referenced[i as usize] = true;
        }
    }
    let mut old_to_new = vec![u32::MAX; n_old];
    let mut k = 0_usize;
    for i in 0..n_old {
        if referenced[i] {
            m.v[k] = m.v[i];
            if let Some(c) = &mut m.colors {
                c[k] = c[i];
            }
            old_to_new[i] = u32::try_from(k).expect("vertex count fits in u32");
            k += 1;
        }
    }
    if k == n_old {
        return 0;
    }
    m.v.truncate(k);
    if let Some(c) = &mut m.colors {
        c.truncate(k);
    }
    for t in &mut m.f {
        for idx in t {
            *idx = old_to_new[*idx as usize];
        }
    }
    n_old - k
}

/// The three passes of R §3.1 step 2, in the reference's order.
///
/// This is what `sherd_refit.fragment.load_mesh` does to every mesh it reads, and every count and
/// checksum the parity fixtures carry for the load stage is taken after it.
pub fn clean(m: &mut Mesh) {
    remove_duplicated_vertices(m);
    remove_degenerate_triangles(m);
    remove_unreferenced_vertices(m);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these tests assert exact coordinates on purpose")]

    use super::{
        clean, remove_degenerate_triangles, remove_duplicated_vertices,
        remove_unreferenced_vertices,
    };
    use crate::mesh::Mesh;

    fn tetra() -> Mesh {
        Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
    }

    #[test]
    fn a_clean_mesh_is_left_alone() {
        let mut m = tetra();
        let before = m.clone();
        clean(&mut m);
        assert_eq!(m, before);
    }

    #[test]
    fn duplicates_merge_onto_the_first_occurrence_and_keep_its_colour() {
        let mut m = Mesh {
            v: vec![[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [1.0, 2.0, 3.0]],
            f: vec![[0, 1, 2], [2, 1, 0]],
            colors: Some(vec![[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        };
        assert_eq!(remove_duplicated_vertices(&mut m), 1);
        assert_eq!(m.v, vec![[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]);
        assert_eq!(m.colors, Some(vec![[1, 1, 1], [2, 2, 2]]));
        assert_eq!(m.f, vec![[0, 1, 0], [0, 1, 0]]);
    }

    #[test]
    fn minus_zero_merges_with_zero_and_nan_merges_with_nothing() {
        let mut m = Mesh::new(vec![[0.0, 0.0, 0.0], [-0.0, 0.0, 0.0]], vec![]);
        assert_eq!(remove_duplicated_vertices(&mut m), 1);
        assert_eq!(m.v, vec![[0.0, 0.0, 0.0]]);

        let nan = f64::NAN;
        let mut m = Mesh::new(vec![[nan, 0.0, 0.0], [nan, 0.0, 0.0]], vec![]);
        assert_eq!(remove_duplicated_vertices(&mut m), 0);
        assert_eq!(m.v.len(), 2);
    }

    #[test]
    fn a_repeated_index_is_degenerate_but_zero_area_alone_is_not() {
        let mut m = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![[0, 1, 2], [0, 1, 1], [1, 1, 1], [2, 1, 0]],
        );
        // [0,1,2] and [2,1,0] are collinear, so their area is zero, but their indices differ.
        assert_eq!(remove_degenerate_triangles(&mut m), 2);
        assert_eq!(m.f, vec![[0, 1, 2], [2, 1, 0]]);
    }

    #[test]
    fn unreferenced_vertices_go_and_the_rest_keep_their_order() {
        let mut m = Mesh {
            v: vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            f: vec![[3, 1, 0]],
            colors: Some(vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        };
        assert_eq!(remove_unreferenced_vertices(&mut m), 1);
        assert_eq!(m.v, vec![[0.0; 3], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        assert_eq!(m.colors, Some(vec![[0, 0, 0], [1, 1, 1], [3, 3, 3]]));
        assert_eq!(m.f, vec![[2, 1, 0]]);
    }

    #[test]
    fn the_three_passes_compose_the_way_the_reference_orders_them() {
        // Vertex 4 duplicates vertex 0, which makes face [0, 4, 1] degenerate; vertex 5 is
        // unreferenced from the start. Only running the passes in R §3.1's order removes all
        // three: a degenerate-first pass would keep face [0, 4, 1].
        let mut m = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [9.0, 9.0, 9.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [0, 4, 1]],
        );
        clean(&mut m);
        assert_eq!(m, tetra());
    }
}
