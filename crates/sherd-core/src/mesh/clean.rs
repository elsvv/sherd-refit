//! Cleaning a freshly read mesh (R §3.1): merge duplicate vertices, drop degenerate triangles,
//! drop duplicate triangles, drop unreferenced vertices — in Open3D's order, because each step
//! changes what the next one sees. Filled in by plan step S3.
