//! OFF reader (R §3.1) — the one own reader in this module.
//!
//! No crate on crates.io reads OFF (E2 checked `mesh-loader`, `tobj`, `meshx` and searched the
//! index), so this is the file below: the `OFF` magic, the counts line, `n` vertex rows, `m` face
//! rows each `k i0 i1 … [r g b [a]]`, `#` comments, and a fan triangulation for `k > 3`.
//!
//! What was measured against Open3D 0.19 rather than guessed, since Open3D's OFF reader is a
//! hand-written extraction loop with quirks of its own:
//!
//! * coordinates are parsed as **`f32`** — Open3D reads them into `float`, so a file carrying more
//!   than seven significant digits is not read at `f64` precision by the reference either;
//! * a `COFF` colour component is read as an **integer over 255**, so `1.0` in a colour column is
//!   read as `1`, i.e. `1/255`. That is Open3D's behaviour, quirk and all, and it is what its own
//!   writer round-trips: Open3D writes `COFF` with four integer colour columns;
//! * a polygon of more than three corners is fan-triangulated — Open3D returns the same two
//!   triangles for a quad;
//! * tokens after a face's corner list (per-face colours) are ignored.
//!
//! Two deviations, both supersets of what Open3D accepts, so no file it reads is read differently
//! here: a `COFF` row may carry three colour components instead of four (Open3D rejects the whole
//! file), and a `#` comment or a blank line is accepted anywhere, not only in the header. One
//! restriction: a row must be a row. Open3D reads the file as one whitespace-separated token
//! stream and would accept a vertex split across two lines; here the row boundary is what tells a
//! three-column `COFF` colour from a four-column one.
//!
//! Filled in by plan step S2.

use std::path::Path;

use crate::error::{Error, Result};
use crate::io::quantize_color;
use crate::mesh::Mesh;

/// Reads an OFF or COFF file.
pub fn read(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    parse(&text, path)
}

/// Parses OFF text; `path` only names the file in error messages.
pub fn parse(text: &str, path: &Path) -> Result<Mesh> {
    let fail = |what: String| Error::read(path, what);
    let mut rows =
        text.lines().map(|l| l.split('#').next().unwrap_or("")).filter(|l| !l.trim().is_empty());

    let first = rows.next().ok_or_else(|| fail("empty file".into()))?;
    let mut header = first.split_whitespace();
    let colored = match header.next() {
        Some("OFF") => false,
        Some("COFF") => true,
        Some(other) => return Err(fail(format!("expected `OFF` or `COFF`, found `{other}`"))),
        None => return Err(fail("empty file".into())),
    };
    // The three counts usually sit on their own line, but `OFF 8 6 0` is legal too.
    let mut counts: Vec<usize> = Vec::with_capacity(3);
    for w in header {
        counts.push(w.parse().map_err(|_| fail(format!("`{w}` is not a count")))?);
    }
    while counts.len() < 3 {
        let row = rows.next().ok_or_else(|| fail("file ends before the counts".into()))?;
        for w in row.split_whitespace() {
            counts.push(w.parse().map_err(|_| fail(format!("`{w}` is not a count")))?);
        }
    }
    let (n_vertices, n_faces) = (counts[0], counts[1]);

    let mut mesh = Mesh {
        v: Vec::with_capacity(n_vertices),
        f: Vec::with_capacity(n_faces),
        colors: colored.then(|| Vec::with_capacity(n_vertices)),
    };
    for i in 0..n_vertices {
        let row = rows.next().ok_or_else(|| fail(format!("file ends at vertex {i}")))?;
        let mut w = row.split_whitespace();
        let mut p = [0.0_f64; 3];
        for c in &mut p {
            let t = w.next().ok_or_else(|| fail(format!("vertex {i} has fewer than 3 columns")))?;
            // Open3D reads OFF coordinates into a `float`; matching that keeps the port's
            // vertices the reference's, bit for bit.
            let v: f32 =
                t.parse().map_err(|_| fail(format!("vertex {i}: `{t}` is not a number")))?;
            *c = f64::from(v);
        }
        mesh.v.push(p);
        if let Some(colors) = &mut mesh.colors {
            let mut rgb = [0_u8; 3];
            for c in &mut rgb {
                let t =
                    w.next().ok_or_else(|| fail(format!("vertex {i} has no colour columns")))?;
                *c = quantize_color(f64::from(leading_integer(t)));
            }
            colors.push(rgb);
        }
    }
    for i in 0..n_faces {
        let row = rows.next().ok_or_else(|| fail(format!("file ends at face {i}")))?;
        let mut w = row.split_whitespace();
        let t = w.next().ok_or_else(|| fail(format!("face {i} is empty")))?;
        let k: usize = t.parse().map_err(|_| fail(format!("face {i}: `{t}` is not a count")))?;
        let mut corners = Vec::with_capacity(k);
        for _ in 0..k {
            let t =
                w.next().ok_or_else(|| fail(format!("face {i} lists fewer than {k} corners")))?;
            corners.push(
                t.parse::<u32>().map_err(|_| fail(format!("face {i}: `{t}` is not an index")))?,
            );
        }
        for j in 2..corners.len() {
            mesh.f.push([corners[0], corners[j - 1], corners[j]]);
        }
    }
    mesh.validate(path)?;
    Ok(mesh)
}

/// The integer prefix of a token, which is what Open3D's extraction into an `int` reads: `1.0`
/// becomes `1`, `255` stays `255`, anything else is `0`.
fn leading_integer(w: &str) -> i32 {
    let end = w
        .char_indices()
        .position(|(i, c)| !(c.is_ascii_digit() || (i == 0 && (c == '-' || c == '+'))))
        .unwrap_or(w.len());
    w[..end].parse::<i32>().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, reason = "the f32 parse is the property under test")]

    use super::{leading_integer, parse};
    use std::path::Path;

    fn off(text: &str) -> crate::mesh::Mesh {
        parse(text, Path::new("<test>.off")).expect("the fixture parses")
    }

    #[test]
    fn comments_and_blank_lines_are_ignored_and_a_quad_fans() {
        let m = off("OFF\n# a comment\n\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n");
        assert_eq!(m.n_vertices(), 4);
        assert_eq!(m.f, vec![[0, 1, 2], [0, 2, 3]]);
        assert!(!m.has_colors());
    }

    #[test]
    fn the_counts_may_share_the_magic_line() {
        let m = off("OFF 3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n");
        assert_eq!((m.n_vertices(), m.n_faces()), (3, 1));
    }

    #[test]
    fn coff_colours_are_integers_over_255_and_the_alpha_column_is_ignored() {
        let m = off("COFF\n3 1 0\n0 0 0 255 0 0 255\n1 0 0 0 255 0 255\n0 1 0 128 128 128 255\n\
                     3 0 1 2\n");
        assert_eq!(m.colors, Some(vec![[255, 0, 0], [0, 255, 0], [128, 128, 128]]));
    }

    #[test]
    fn a_coff_row_may_omit_the_alpha_column() {
        let m = off("COFF\n3 1 0\n0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 1 2 3\n3 0 1 2\n");
        assert_eq!(m.colors, Some(vec![[255, 0, 0], [0, 255, 0], [1, 2, 3]]));
        assert_eq!(m.f, vec![[0, 1, 2]]);
    }

    #[test]
    fn a_float_colour_column_reads_as_its_integer_part_the_way_open3d_reads_it() {
        assert_eq!(leading_integer("1.0"), 1);
        assert_eq!(leading_integer("255"), 255);
        assert_eq!(leading_integer("x"), 0);
        let m = off("COFF\n3 1 0\n0 0 0 1.0 0.0 0.0 1.0\n1 0 0 0.0 1.0 0.0 1.0\n\
                     0 1 0 0.5 0.5 0.5 1.0\n3 0 1 2\n");
        assert_eq!(m.colors, Some(vec![[1, 0, 0], [0, 1, 0], [0, 0, 0]]));
    }

    #[test]
    fn coordinates_are_parsed_the_way_open3d_parses_them() {
        // Open3D reads OFF coordinates into a `float`; 0.12345678901234567 comes back as the f32.
        let m = off("OFF\n3 1 0\n0.12345678901234567 1 2\n1 0 0\n0 1 0\n3 0 1 2\n");
        assert_eq!(m.v[0][0], f64::from(0.123_456_79_f32));
    }

    #[test]
    fn per_face_colours_after_the_corner_list_are_ignored() {
        let m = off("OFF\n4 2 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2 255 0 0\n3 0 2 3 0 255 0\n");
        assert_eq!(m.f, vec![[0, 1, 2], [0, 2, 3]]);
    }

    #[test]
    fn a_bad_magic_and_a_truncated_body_are_errors() {
        assert!(parse("PLY\n1 0 0\n", Path::new("x.off")).is_err());
        assert!(parse("OFF\n3 1 0\n0 0 0\n", Path::new("x.off")).is_err());
        assert!(parse("", Path::new("x.off")).is_err());
    }
}
