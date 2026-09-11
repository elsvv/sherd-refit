//! `transforms.csv`, `joins.csv` and `matrices/<name>.txt`: the placement as plain text, for a
//! spreadsheet, a script, or a program's "apply transformation" dialog.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use nalgebra::Matrix4;

use super::{BOM, Export, JoinRow, number, write_text};
use crate::error::Result;

/// One row per fragment: its matrix row-major, its group, whether it was assembled.
pub const TRANSFORMS_CSV: &str = "transforms.csv";
/// One row per pair [`Export::joins`] lists.
pub const JOINS_CSV: &str = "joins.csv";
/// One `<name>.txt` per assembled fragment: the matrix as four lines of four numbers.
pub const MATRICES_DIR: &str = "matrices";

/// Writes the three, and returns the files in the order they were written.
pub fn write_tables(out_dir: &Path, ex: &Export<'_>, joins: &[JoinRow]) -> Result<Vec<PathBuf>> {
    let mut written = vec![
        write_text(&out_dir.join(TRANSFORMS_CSV), &transforms_csv(ex))?,
        write_text(&out_dir.join(JOINS_CSV), &joins_csv(ex, joins))?,
    ];
    let assembled = ex.assembled();
    for (n, name) in ex.names.iter().enumerate() {
        if assembled[n] {
            let path = out_dir.join(MATRICES_DIR).join(format!("{name}.txt"));
            written.push(write_text(&path, &matrix_text(&ex.poses[n]))?);
        }
    }
    Ok(written)
}

/// `transforms.csv`: `fragment,source_file,group,assembled,m00,…,m33`, the matrix row by row, the
/// fragments in `transforms.json`'s group order — the largest group first, each in its own order.
pub fn transforms_csv(ex: &Export<'_>) -> String {
    let assembled = ex.assembled();
    let mut out = format!("{BOM}fragment,source_file,group,assembled");
    for r in 0..4 {
        for c in 0..4 {
            let _ = write!(out, ",m{r}{c}");
        }
    }
    out.push('\n');
    let mut listed = vec![false; ex.names.len()];
    let order =
        ex.groups.iter().enumerate().flat_map(|(k, g)| g.iter().map(move |&n| (Some(k), n)));
    let rest = (0..ex.names.len()).map(|n| (None, u32::try_from(n).unwrap_or(u32::MAX)));
    for (group, n) in order.chain(rest) {
        let n = n as usize;
        if n >= listed.len() || listed[n] {
            continue;
        }
        listed[n] = true;
        let _ = write!(
            out,
            "{},{},{},{}",
            field(&ex.names[n]),
            field(&ex.source_name(n)),
            group.map_or_else(String::new, |k| k.to_string()),
            u8::from(assembled[n]),
        );
        for r in 0..4 {
            for c in 0..4 {
                let _ = write!(out, ",{}", number(ex.poses[n][(r, c)]));
            }
        }
        out.push('\n');
    }
    out
}

/// `joins.csv`: one row per pair of [`Export::joins`], in its order.
pub fn joins_csv(ex: &Export<'_>, joins: &[JoinRow]) -> String {
    let mut out = format!(
        "{BOM}fragment_a,fragment_b,band,in_assembly,group,score,seam_t,tight,gap_t,gap_limit_t,\
         normal_agreement,penetration,review_image\n"
    );
    for j in joins {
        let _ = writeln!(
            out,
            "{},{},{},{},{},{:.3},{:.2},{:.3},{:.4},{:.4},{:.3},{:.4},{}",
            field(&ex.names[j.a as usize]),
            field(&ex.names[j.b as usize]),
            j.band,
            u8::from(j.in_assembly),
            j.group.map_or_else(String::new, |k| k.to_string()),
            j.score,
            j.seam,
            j.tight,
            j.gap,
            j.gap_limit,
            j.cont_n,
            j.pen,
            j.review.as_deref().map_or_else(String::new, field),
        );
    }
    out
}

/// A matrix as four lines of four space-separated numbers — what CloudCompare's "Apply
/// transformation → ASCII file" reads and MeshLab's "Matrix: Set/Copy Transformation" pastes.
pub fn matrix_text(m: &Matrix4<f64>) -> String {
    let mut out = String::new();
    for r in 0..4 {
        let row: Vec<String> = (0..4).map(|c| number(m[(r, c)])).collect();
        out.push_str(&row.join(" "));
        out.push('\n');
    }
    out
}

/// RFC 4180 quoting, for the rare name with a comma or a quote in it.
fn field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{field, matrix_text};
    use nalgebra::Matrix4;

    #[test]
    fn a_matrix_reads_back_row_by_row() {
        let mut m = Matrix4::identity();
        m[(0, 3)] = -560.789_876_567_390_4;
        m[(1, 0)] = 0.25;
        let text = matrix_text(&m);
        let back: Vec<Vec<f64>> = text
            .lines()
            .map(|l| l.split(' ').map(|x| x.parse().expect("a number")).collect())
            .collect();
        for (r, row) in back.iter().enumerate() {
            for (c, x) in row.iter().enumerate() {
                assert_eq!(x.to_bits(), m[(r, c)].to_bits(), "({r}, {c})");
            }
        }
        assert_eq!(text.lines().next(), Some("1 0 0 -560.7898765673904"));
    }

    #[test]
    fn only_a_name_that_needs_it_is_quoted() {
        assert_eq!(field("FY234007"), "FY234007");
        assert_eq!(field("a,b"), "\"a,b\"");
        assert_eq!(field("say \"x\""), "\"say \"\"x\"\"\"");
    }
}
