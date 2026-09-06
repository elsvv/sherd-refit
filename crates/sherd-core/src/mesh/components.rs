//! Connected components and the largest-component rule (R §3.1 step 4).
//!
//! A scan often carries specks of stray geometry — a chip of the table, a second sherd caught in
//! the same frame — and only the biggest piece is the fragment. The reference calls Open3D's
//! `cluster_connected_triangles`, which connects two triangles when they **share an edge**: two
//! triangles meeting at a single vertex are *not* connected, and that is deliberate, because a
//! scan's stray specks often touch the fragment at one point.
//!
//! Filled in by plan step S2.

use super::{Mesh, clean::remove_unreferenced_vertices};

/// The edge-connected clustering of a mesh's triangles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Clusters {
    /// Cluster index of every triangle, in triangle order.
    pub labels: Vec<u32>,
    /// Number of triangles in each cluster, indexed by cluster.
    pub counts: Vec<usize>,
}

impl Clusters {
    /// Number of clusters.
    #[inline]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// True when the mesh had no triangle at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// The cluster with the most triangles, ties going to the lowest cluster index.
    ///
    /// The tie-break is the reference's: `np.argmax` returns the first maximum, and clusters are
    /// numbered in the order their first triangle appears, so a tie keeps the component that owns
    /// the earliest triangle.
    pub fn largest(&self) -> Option<u32> {
        let mut best: Option<(usize, u32)> = None;
        for (i, &c) in self.counts.iter().enumerate() {
            let i = u32::try_from(i).expect("cluster count fits in u32");
            if best.is_none_or(|(bc, _)| c > bc) {
                best = Some((c, i));
            }
        }
        best.map(|(_, i)| i)
    }
}

/// Clusters the triangles by shared edge (Open3D's `cluster_connected_triangles`).
///
/// Clusters are numbered in the order their first triangle appears, so the numbering — and with it
/// the tie-break in [`Clusters::largest`] — does not depend on how the union–find happens to
/// resolve.
pub fn cluster_connected_triangles(m: &Mesh) -> Clusters {
    let n = m.f.len();
    let mut parent: Vec<u32> = (0..u32::try_from(n).expect("face count fits in u32")).collect();

    // (edge key, triangle) for all 3·n directed edges, sorted so that the triangles sharing an
    // edge end up adjacent. Sorting rather than hashing keeps the result independent of any hasher
    // and of the iteration order of a map.
    let mut edges: Vec<(u64, u32)> = Vec::with_capacity(3 * n);
    for (i, t) in m.f.iter().enumerate() {
        let i = u32::try_from(i).expect("face count fits in u32");
        edges.push((edge_key(t[0], t[1]), i));
        edges.push((edge_key(t[1], t[2]), i));
        edges.push((edge_key(t[2], t[0]), i));
    }
    edges.sort_unstable();
    let mut start = 0;
    while start < edges.len() {
        let mut end = start + 1;
        while end < edges.len() && edges[end].0 == edges[start].0 {
            end += 1;
        }
        for k in start + 1..end {
            union(&mut parent, edges[start].1, edges[k].1);
        }
        start = end;
    }

    let mut labels = vec![u32::MAX; n];
    let mut counts = Vec::new();
    let mut root_label = vec![u32::MAX; n];
    for (i, label) in labels.iter_mut().enumerate() {
        let r = find(&mut parent, u32::try_from(i).expect("face count fits in u32")) as usize;
        if root_label[r] == u32::MAX {
            root_label[r] = u32::try_from(counts.len()).expect("cluster count fits in u32");
            counts.push(0);
        }
        *label = root_label[r];
        counts[root_label[r] as usize] += 1;
    }
    Clusters { labels, counts }
}

/// Keeps only the largest edge-connected component and drops the vertices that lose their
/// triangles; returns the number of components the mesh had.
///
/// Exactly `sherd_refit.fragment.largest_component`: with a single component (or none) the mesh is
/// left untouched, including its unreferenced vertices, because the reference's `if len(counts) >
/// 1` guard skips the whole body.
pub fn largest_component(m: &mut Mesh) -> usize {
    let clusters = cluster_connected_triangles(m);
    if clusters.len() > 1 {
        let keep = clusters.largest().expect("a non-empty clustering has a largest cluster");
        let mut i = 0;
        m.f.retain(|_| {
            let k = clusters.labels[i] == keep;
            i += 1;
            k
        });
        remove_unreferenced_vertices(m);
    }
    clusters.len()
}

#[inline]
fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    (u64::from(lo) << 32) | u64::from(hi)
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let grand = parent[parent[x as usize] as usize];
        parent[x as usize] = grand;
        x = grand;
    }
    x
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        // Point the higher root at the lower one: the roots stay meaningful, and the labelling
        // pass below does not depend on which way this went anyway.
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "these tests assert exact coordinates on purpose")]

    use super::{cluster_connected_triangles, largest_component};
    use crate::mesh::Mesh;

    /// Two tetrahedra: the first four faces, then a second, smaller piece of two faces.
    fn two_pieces() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
                [5.0, 6.0, 5.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [4, 5, 6], [4, 6, 5]],
        )
    }

    #[test]
    fn triangles_sharing_only_a_vertex_are_not_connected() {
        // Two triangles that meet at vertex 0 and nowhere else.
        let m = Mesh::new(
            vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            vec![[0, 1, 2], [0, 3, 4]],
        );
        let c = cluster_connected_triangles(&m);
        assert_eq!(c.len(), 2);
        assert_eq!(c.counts, vec![1, 1]);
        assert_eq!(c.labels, vec![0, 1]);
    }

    #[test]
    fn clusters_are_numbered_by_their_first_triangle() {
        let m = two_pieces();
        let c = cluster_connected_triangles(&m);
        assert_eq!(c.labels, vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(c.counts, vec![4, 2]);
        assert_eq!(c.largest(), Some(0));
    }

    #[test]
    fn the_largest_component_survives_with_its_vertices_renumbered() {
        let mut m = two_pieces();
        assert_eq!(largest_component(&mut m), 2);
        assert_eq!(m.n_faces(), 4);
        assert_eq!(m.n_vertices(), 4);
        assert_eq!(m.f, vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]);
        assert_eq!(m.v[3], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_tie_keeps_the_component_with_the_earliest_triangle() {
        let mut m = Mesh::new(
            vec![
                [0.0; 3],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0; 3],
                [6.0, 5.0, 5.0],
                [5.0, 6.0, 5.0],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );
        assert_eq!(largest_component(&mut m), 2);
        assert_eq!(m.f, vec![[0, 1, 2]]);
        assert_eq!(m.v, vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    #[test]
    fn a_single_component_is_left_untouched_including_its_stray_vertices() {
        let mut m =
            Mesh::new(vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0; 3]], vec![[0, 1, 2]]);
        assert_eq!(largest_component(&mut m), 1);
        assert_eq!(m.n_vertices(), 4, "the reference's `len(counts) > 1` guard skips the body");
    }

    #[test]
    fn an_empty_mesh_clusters_into_nothing() {
        let mut m = Mesh::default();
        let c = cluster_connected_triangles(&m);
        assert!(c.is_empty());
        assert_eq!(c.largest(), None);
        assert_eq!(largest_component(&mut m), 0);
    }
}
