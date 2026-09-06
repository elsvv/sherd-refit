//! Edge adjacency and `closed_enough` (R §3.1, R §3.3).
//!
//! Three structures are built from the same raw list of undirected edges:
//!
//! * [`unique_edges`] — every undirected edge once, with how many faces use it. `res`
//!   ([`super::geometry::median_edge`]) is a median over these, and [`closed_enough`] counts the
//!   ones a number of faces other than two uses.
//! * [`face_adjacency`] — the `(fa, fb)` pairs of faces that share an edge, which the
//!   segmentation of R §3.4 votes over and grows the shell across (phase 1b).
//! * [`vertex_adjacency`] — the per-vertex neighbour lists Taubin smoothing averages over
//!   (R §3.3.1); Open3D calls the same thing `adjacency_list_`.
//!
//! An edge is identified by its two vertex indices in ascending order, exactly as
//! `sherd_refit.fragment.closed_enough` and `sherd_refit.geometry.face_adjacency` sort theirs
//! (`np.sort(E, 1)`). The reference then packs the pair into one integer
//! (`e0 · (F.max() + 1) + e1`); this module packs it into a `u64` instead, which is the same map
//! from a pair to a key for every mesh a `u32` index buffer can describe.

use super::Mesh;

/// One undirected edge as a key: the smaller index in the high half, the larger in the low half.
///
/// Ordering by this key is ordering by `(min, max)`, which is `np.lexsort((key[:, 1], key[:, 0]))`
/// in the reference.
#[inline]
fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    (u64::from(lo) << 32) | u64::from(hi)
}

/// The two vertex indices of a key, ascending.
#[inline]
fn edge_of(key: u64) -> [u32; 2] {
    #[allow(clippy::cast_possible_truncation, reason = "the halves are u32 by construction")]
    [(key >> 32) as u32, (key & 0xffff_ffff) as u32]
}

/// The `3·m` undirected edges of a face list, in the reference's order.
///
/// The reference builds `np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])`: the three
/// edge slots are *outer*, the faces inner. Anything that has to reproduce the reference's tie
/// order — [`face_adjacency`] — depends on that, so it is the order this function emits.
fn edge_keys(f: &[[u32; 3]]) -> Vec<u64> {
    let mut keys = Vec::with_capacity(f.len() * 3);
    for slot in [(0, 1), (1, 2), (2, 0)] {
        for t in f {
            keys.push(edge_key(t[slot.0], t[slot.1]));
        }
    }
    keys
}

/// Every undirected edge of the mesh once, ascending, with the number of faces that use it.
///
/// This is `np.unique(np.sort(E, 1), axis=0, return_counts=True)` of R §3.3.2. A triangle that
/// names the same vertex twice would contribute a self-edge; R §3.1 removes those triangles
/// before anything here runs.
pub fn unique_edges(f: &[[u32; 3]]) -> (Vec<[u32; 2]>, Vec<u32>) {
    let mut keys = edge_keys(f);
    keys.sort_unstable();
    let mut edges = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < keys.len() {
        let key = keys[i];
        let mut j = i + 1;
        while j < keys.len() && keys[j] == key {
            j += 1;
        }
        edges.push(edge_of(key));
        counts.push(u32::try_from(j - i).unwrap_or(u32::MAX));
        i = j;
    }
    (edges, counts)
}

/// The fraction of boundary edges R §3.3.2 tolerates in a "watertight" fragment.
pub const MAX_BOUNDARY_FRACTION: f64 = 0.002;

/// R §3.3.2: is the mesh closed enough for a signed distance to be trusted?
///
/// Returns `(watertight, n_boundary)`, where `n_boundary` counts the unique edges used by a
/// number of faces *other than two* — so it counts non-manifold edges as well as open ones, which
/// is what the reference's `(counts != 2).sum()` does. A fragment that fails this gets no
/// penetration test (R §6.4).
///
/// Decimation opens a few triangle-sized holes on a scan, which is why the test is a fraction and
/// not zero: `n_boundary ≤ 0.002 · n_unique_edges`.
pub fn closed_enough(f: &[[u32; 3]]) -> (bool, usize) {
    closed_enough_with(f, MAX_BOUNDARY_FRACTION)
}

/// [`closed_enough`] with the reference's `max_boundary_fraction` argument exposed.
pub fn closed_enough_with(f: &[[u32; 3]], max_boundary_fraction: f64) -> (bool, usize) {
    let (_, counts) = unique_edges(f);
    let n_boundary = counts.iter().filter(|&&c| c != 2).count();
    #[allow(clippy::cast_precision_loss, reason = "edge counts are far below 2^53")]
    let watertight = n_boundary as f64 <= max_boundary_fraction * counts.len() as f64;
    (watertight, n_boundary)
}

/// Pairs of faces that share an edge (`sherd_refit.geometry.face_adjacency`).
///
/// One entry per *consecutive* pair in the reference's sorted edge list, so an edge used by two
/// faces yields one pair and a non-manifold edge used by three yields two — the same chain the
/// reference produces, in the same order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FaceAdjacency {
    /// First face of each pair.
    pub fa: Vec<u32>,
    /// Second face of each pair.
    pub fb: Vec<u32>,
    /// The shared edge, its two vertex indices ascending.
    pub edge: Vec<[u32; 2]>,
}

impl FaceAdjacency {
    /// Number of adjacent face pairs.
    #[inline]
    pub fn len(&self) -> usize {
        self.fa.len()
    }

    /// True when no two faces share an edge.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fa.is_empty()
    }
}

/// Builds the face-pair adjacency of R §3.4.
///
/// The reference sorts its `3·m` edge rows with `np.lexsort` — a *stable* sort by the edge key —
/// and pairs every two consecutive rows that carry the same key. Stability is what fixes the
/// pairing when three or more faces share an edge, so this function sorts by `(key, position)`
/// with the position being the index in [`edge_keys`]'s slot-major order, which is the same
/// permutation.
pub fn face_adjacency(f: &[[u32; 3]]) -> FaceAdjacency {
    let n = f.len();
    let keys = edge_keys(f);
    let mut order: Vec<u32> = (0..u32::try_from(keys.len()).expect("3·m fits in u32")).collect();
    order.sort_unstable_by_key(|&i| (keys[i as usize], i));

    let mut out = FaceAdjacency::default();
    for w in order.windows(2) {
        let (i, j) = (w[0] as usize, w[1] as usize);
        if keys[i] == keys[j] {
            #[allow(clippy::cast_possible_truncation, reason = "i < 3·m and m fits in u32")]
            {
                out.fa.push((i % n) as u32);
                out.fb.push((j % n) as u32);
            }
            out.edge.push(edge_of(keys[i]));
        }
    }
    out
}

/// The per-vertex neighbour lists of R §3.3.1, in compressed-row form.
///
/// Open3D's `ComputeAdjacencyList` stores the same thing as a `std::vector<std::unordered_set>`;
/// the neighbours here are sorted ascending instead, which fixes the order the Taubin sums are
/// accumulated in (D §7 forbids unordered containers on a result path). The only consequence is
/// round-off: see [`super::taubin`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexAdjacency {
    offsets: Vec<u32>,
    neighbours: Vec<u32>,
}

impl VertexAdjacency {
    /// Number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// True when the mesh has no vertex.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The neighbours of one vertex, ascending and without repeats.
    #[inline]
    pub fn neighbours(&self, v: usize) -> &[u32] {
        &self.neighbours[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

/// Builds the vertex adjacency of a face list.
///
/// Every triangle contributes its three edges in both directions; duplicates (an edge shared by
/// two faces) are merged, so a neighbour appears once however many faces reach it.
pub fn vertex_adjacency(n_vertices: usize, f: &[[u32; 3]]) -> VertexAdjacency {
    let mut degree = vec![0_u32; n_vertices + 1];
    for t in f {
        for k in 0..3 {
            degree[t[k] as usize + 1] += 2;
        }
    }
    let mut offsets = degree;
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    let mut fill = offsets.clone();
    let mut flat = vec![0_u32; offsets[n_vertices] as usize];
    for t in f {
        for k in 0..3 {
            let (u, v) = (t[k], t[(k + 1) % 3]);
            flat[fill[u as usize] as usize] = v;
            fill[u as usize] += 1;
            flat[fill[v as usize] as usize] = u;
            fill[v as usize] += 1;
        }
    }

    // Sort and deduplicate each vertex's list in place, then compact.
    let mut neighbours = Vec::with_capacity(flat.len());
    let mut compact = vec![0_u32; n_vertices + 1];
    for v in 0..n_vertices {
        let (lo, hi) = (offsets[v] as usize, offsets[v + 1] as usize);
        let slice = &mut flat[lo..hi];
        slice.sort_unstable();
        let mut last: Option<u32> = None;
        for &w in slice.iter() {
            if last != Some(w) {
                neighbours.push(w);
                last = Some(w);
            }
        }
        compact[v + 1] = u32::try_from(neighbours.len()).expect("neighbour count fits in u32");
    }
    neighbours.shrink_to_fit();
    VertexAdjacency { offsets: compact, neighbours }
}

/// [`vertex_adjacency`] for a whole mesh.
pub fn mesh_vertex_adjacency(m: &Mesh) -> VertexAdjacency {
    vertex_adjacency(m.v.len(), &m.f)
}

#[cfg(test)]
mod tests {
    use super::{
        FaceAdjacency, closed_enough, closed_enough_with, face_adjacency, unique_edges,
        vertex_adjacency,
    };

    /// A closed tetrahedron: four faces, six edges, every edge used twice.
    fn tetra() -> Vec<[u32; 3]> {
        vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    }

    #[test]
    fn a_closed_mesh_has_every_edge_twice_and_three_neighbours_per_face() {
        let f = tetra();
        let (edges, counts) = unique_edges(&f);
        assert_eq!(edges.len(), 6);
        assert!(counts.iter().all(|&c| c == 2));
        assert_eq!(edges, vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);

        let adj = face_adjacency(&f);
        assert_eq!(adj.len(), 6, "six shared edges, one pair each");
        let mut degree = [0_usize; 4];
        for i in 0..adj.len() {
            degree[adj.fa[i] as usize] += 1;
            degree[adj.fb[i] as usize] += 1;
        }
        assert_eq!(degree, [3, 3, 3, 3], "every face of a closed mesh has three neighbours");

        assert_eq!(closed_enough(&f), (true, 0));
    }

    #[test]
    fn an_open_mesh_counts_its_boundary_edges() {
        // One triangle: three edges, each used once.
        let f = vec![[0, 1, 2]];
        assert_eq!(closed_enough(&f), (false, 3));
        assert!(face_adjacency(&f).is_empty());

        // Two triangles sharing an edge: five edges, four of them on the boundary.
        let f = vec![[0, 1, 2], [2, 1, 3]];
        assert_eq!(closed_enough(&f), (false, 4));
        // (1, 0), not (0, 1): face 1 reaches the shared edge in its *first* slot and face 0 in its
        // second, and the reference's stable sort keeps slot order. `sherd_refit.geometry
        // .face_adjacency(np.array([[0,1,2],[2,1,3]]))` returns exactly this.
        assert_eq!(
            face_adjacency(&f),
            FaceAdjacency { fa: vec![1], fb: vec![0], edge: vec![[1, 2]] }
        );
    }

    #[test]
    fn the_watertight_test_is_a_fraction_not_a_count() {
        // A closed tetrahedron with one face removed: 6 edges, 3 of them boundary.
        let mut f = tetra();
        f.pop();
        let (watertight, n_boundary) = closed_enough(&f);
        assert!(!watertight);
        assert_eq!(n_boundary, 3);
        // The same mesh judged with a fraction that admits it.
        assert_eq!(closed_enough_with(&f, 0.5), (true, 3));
    }

    #[test]
    fn a_non_manifold_edge_counts_as_a_boundary_and_pairs_in_a_chain() {
        // Three triangles hinged on the edge (0, 1).
        let f = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        let (edges, counts) = unique_edges(&f);
        assert_eq!(edges[0], [0, 1]);
        assert_eq!(counts[0], 3);
        let (watertight, n_boundary) = closed_enough(&f);
        assert!(!watertight);
        assert_eq!(n_boundary, 7, "the hinge plus six free edges");

        let adj = face_adjacency(&f);
        assert_eq!((adj.fa.clone(), adj.fb.clone()), (vec![0, 1], vec![1, 2]));
        assert!(adj.edge.iter().all(|&e| e == [0, 1]));
    }

    #[test]
    fn vertex_neighbours_are_sorted_and_unique() {
        let f = tetra();
        let adj = vertex_adjacency(4, &f);
        assert_eq!(adj.len(), 4);
        for v in 0..4 {
            let n: Vec<u32> = adj.neighbours(v).to_vec();
            let expected: Vec<u32> = (0..4).filter(|&w| w != u32::try_from(v).unwrap()).collect();
            assert_eq!(n, expected, "vertex {v}");
        }

        // A vertex no triangle names has no neighbours, and does not shift the others.
        let adj = vertex_adjacency(6, &f);
        assert_eq!(adj.len(), 6);
        assert!(adj.neighbours(4).is_empty());
        assert!(adj.neighbours(5).is_empty());
        assert_eq!(adj.neighbours(0), &[1, 2, 3]);
    }

    #[test]
    fn an_empty_face_list_yields_empty_structures() {
        let f: Vec<[u32; 3]> = Vec::new();
        assert_eq!(unique_edges(&f), (Vec::new(), Vec::new()));
        assert_eq!(closed_enough(&f), (true, 0), "no edges, no boundary");
        assert!(face_adjacency(&f).is_empty());
        assert!(vertex_adjacency(0, &f).is_empty());
    }
}
