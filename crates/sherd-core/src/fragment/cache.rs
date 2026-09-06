//! The fragment cache `<out>/cache/<name>.sherd` (R §3.7, D §4.2).
//!
//! A `safetensors` file: the working mesh, the labels, the match arrays and (later) the features
//! as tensors, plus a string metadata map carrying `format`, `cache_version`, `algo_ref`,
//! `core_version`, the source's path/size/mtime/optional sha256, `target_faces`, `thick`,
//! `thick_mode`, `res`, `watertight`, the counts and the `md_params` JSON. Loading is an mmap and
//! a header parse; a mismatch of `md_params` alone recomputes only the match arrays. The Python
//! package can read the file with `safetensors.numpy.load_file`, which is what makes the
//! transition possible. Filled in by plan step S4.
