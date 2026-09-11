//! `viewer.html`: [`scene.glb`](super::scene) in any browser, the fragment's name under the cursor.
//!
//! One self-contained file — the page, its WebGL2 renderer and the scene itself, base64 inside it
//! — so it opens by double-click, offline, from wherever it is copied, with nothing installed. The
//! page reads the GLB it carries as any glTF reader would; the side panel (the groups, the joins,
//! their bands and their review images) comes from a JSON block written beside it.
//!
//! What it does: hovering a fragment shows its name, group and joins; a click selects it and lists
//! its joins, each partner a link; a double click flies to it; `L` writes every fragment's name on
//! the scene at once; `C` cycles the colours (the scan's own, one per fragment, one per group); a
//! group can be hidden or framed; `viewer.html#fragment=<name>` opens with that fragment selected.

use std::path::{Path, PathBuf};

use serde_json::json;

use super::scene::Scene;
use super::{Export, JoinRow, write_text};
use crate::error::Result;
use crate::{CORE_VERSION, GIT_COMMIT};

/// The file, under the output directory.
pub const VIEWER_FILE: &str = "viewer.html";

const TEMPLATE: &str = include_str!("viewer.html");
const TITLE: &str = "__SHERD_TITLE__";
const META: &str = "__SHERD_META__";
const GLB: &str = "__SHERD_GLB__";

/// Writes `viewer.html` around `scene`.
pub fn write_viewer(
    out_dir: &Path,
    ex: &Export<'_>,
    joins: &[JoinRow],
    scene: &Scene,
) -> Result<PathBuf> {
    write_text(&out_dir.join(VIEWER_FILE), &page(ex, joins, scene))
}

/// The page: the template with the title, the side panel's JSON and the scene filled in.
pub fn page(ex: &Export<'_>, joins: &[JoinRow], scene: &Scene) -> String {
    let filled =
        TEMPLATE.replace(TITLE, &escape_html(ex.collection)).replace(META, &meta(ex, joins, scene));
    let (head, tail) =
        filled.split_once(GLB).expect("the template carries the scene's placeholder");
    let data = base64(&scene.bytes);
    let mut out = String::with_capacity(head.len() + data.len() + tail.len());
    out.push_str(head);
    out.push_str(&data);
    out.push_str(tail);
    out
}

/// The side panel's data: the fragments, the groups and the joins, indices by
/// [`FragId`](crate::types::FragId). Safe inside a `<script>` element: `<`, `>` and `&` can only
/// occur inside JSON strings, where their `\u` escapes mean the same.
pub fn meta(ex: &Export<'_>, joins: &[JoinRow], scene: &Scene) -> String {
    let group_of = ex.group_of();
    let assembled = ex.assembled();
    let fragments: Vec<_> = ex
        .names
        .iter()
        .enumerate()
        .map(|(n, name)| {
            json!({
                "name": name,
                "source": ex.source_name(n),
                "group": (group_of[n] != usize::MAX).then_some(group_of[n]),
                "assembled": assembled[n],
                "colours": scene.coloured.get(n).copied().unwrap_or(false),
            })
        })
        .collect();
    let groups: Vec<_> = ex
        .groups
        .iter()
        .enumerate()
        .map(|(k, g)| json!({ "index": k, "members": g, "assembled": g.len() > 1 }))
        .collect();
    let joins: Vec<_> = joins
        .iter()
        .map(|j| {
            json!({
                "a": j.a,
                "b": j.b,
                "band": j.band,
                "in_assembly": j.in_assembly,
                "score": round(j.score, 3),
                "seam": round(j.seam, 2),
                "tight": round(j.tight, 3),
                "gap": round(j.gap, 4),
                "review": j.review,
            })
        })
        .collect();
    let value = json!({
        "title": ex.collection,
        "generator": format!("sherd-refit {CORE_VERSION} ({GIT_COMMIT})"),
        "thickness": ex.thickness,
        "tiers": ex.tiers,
        "faces": scene.faces,
        "fragments": fragments,
        "groups": groups,
        "joins": joins,
    });
    value
        .to_string()
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

/// RFC 4648 base64, padded.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let at = |i: usize| usize::from(chunk.get(i).copied().unwrap_or(0));
        let n = (at(0) << 16) | (at(1) << 8) | at(2);
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> shift) & 63]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn round(x: f64, digits: i32) -> f64 {
    let scale = 10_f64.powi(digits);
    (x * scale).round() / scale
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::{GLB, META, TEMPLATE, TITLE, base64, escape_html};

    #[test]
    fn base64_is_rfc_4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"M"), "TQ==");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(&[0xff, 0xfe, 0xfd, 0x00]), "//79AA==");
    }

    #[test]
    fn the_template_has_one_place_for_each_thing() {
        assert_eq!(TEMPLATE.matches(GLB).count(), 1);
        assert_eq!(TEMPLATE.matches(META).count(), 1);
        assert!(TEMPLATE.matches(TITLE).count() >= 1);
        assert_eq!(escape_html("<a & 'b'>"), "&lt;a &amp; &#39;b&#39;&gt;");
    }
}
