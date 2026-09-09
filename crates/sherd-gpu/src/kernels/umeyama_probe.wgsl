
// The test entry point for `umeyama.wgsl`, and the only reason that file is separate.
//
// `umeyama_rotation` is a pure function of nine floats, so it can be asked a question directly
// instead of only through a rung: one workgroup per case, nine words of covariance in (row-major,
// `sigma[r][c]` at `9k + 3r + c`), nine words of rotation out (row-major, at `9(k + cases) + 3i +
// j`). `icp::umeyama_rotations` is the host half; `tests/adapter.rs` is the caller.
//
// It is compiled only by that helper — the production pipelines are built from `icp.wgsl`'s two
// entry points and never see this file.

@group(0) @binding(0) var<storage, read_write> probe_io: array<f32>;

@compute @workgroup_size(1)
fn umeyama_probe(@builtin(workgroup_id) wg: vec3<u32>) {
    let cases = arrayLength(&probe_io) / 18u;
    let k = wg.x;
    if (k >= cases) {
        return;
    }
    let base = k * 9u;
    let rot = umeyama_rotation(
        vec3<f32>(probe_io[base + 0u], probe_io[base + 3u], probe_io[base + 6u]),
        vec3<f32>(probe_io[base + 1u], probe_io[base + 4u], probe_io[base + 7u]),
        vec3<f32>(probe_io[base + 2u], probe_io[base + 5u], probe_io[base + 8u]),
    );
    let out = (cases + k) * 9u;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            // `rot` is column-major in WGSL: `rot[j][i]` is row `i`, column `j`.
            probe_io[out + i * 3u + j] = rot[j][i];
        }
    }
}
