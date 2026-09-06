//! Shell/fracture segmentation of the working mesh (R §3.4): the opposite-wall test by cone
//! rays, the `good`/`near` masks, majority filtering over the face adjacency, island removal and
//! the reference pass. This is the stage the gates watch most closely — ≥ 0.995 area-weighted
//! agreement injected, ≥ 0.97 native (D §10.2). Filled in in phase 1b.
