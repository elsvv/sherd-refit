//! Breaklines: the closed curves where fracture meets shell (R §3.5.5), thinned to a `brk_voxel`
//! grid with sorted, deterministic representatives (PMC-4), with the shell and fracture normals
//! and the dihedral angle carried per point. They drive the hypotheses (§5.1), the coarse score
//! (§5.2) and the seam test (§6.2). Filled in in phase 1b.
