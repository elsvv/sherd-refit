//! «Открыть в Blender» (A §9.2): the reviewed assembly as a Python script Blender runs.
//!
//! The app never speaks to Blender. It writes a file and starts `blender --python <file>`, which
//! is the only interface Blender offers that survives its own versions, and the only one that
//! leaves a museum colleague something to keep: the script is a plain text record of what stood
//! where, readable and re-runnable without the app.
//!
//! Everything here is a pure function of the assembly, the input snapshot and two choices. That
//! is what makes a golden test possible — the script is compared as text, and `blender_golden.py`
//! is real Python that `python3 -m py_compile` reads. Blender itself is not installed on the
//! machine this was written on, so the golden file and the compile are the whole of the
//! verification; A §11's list of what is left for a person names the rest.
//!
//! The script's own words are Russian, as the window's are: the person reading Blender's console
//! is the person who pressed the button, and the group labels («Группа 0») are the same ones the
//! «Сборка» inspector shows. i18next has nothing to do here — this text is generated, not shown.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::protocol::AssemblyDto;
use crate::snapshot::InputSnapshot;

/// Which mesh of a fragment goes into Blender (A §9.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// The original scan from the input folder — what the museum measured, tens of megabytes
    /// apiece.
    Full,
    /// `fragments/<name>.glb`, the display mesh the window itself draws: a laptop opens a group
    /// of twenty of these and would not open twenty originals.
    Display,
}

/// How much of the assembly is opened (A §9.2): the group the inspector has, or all of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every group of two or more. A group of one is a fragment nothing was found for, and a
    /// Blender file of fifty unrelated sherds on a row is not what «вся сборка» means.
    All,
    /// One group by its index in [`AssemblyDto::groups`] — the same number the window prints as
    /// «Группа N», whatever its size, because the inspector asked for that one.
    Group(usize),
}

/// [`Resolution`] as the window asks for it (A §9.2's two menu items).
///
/// A mirror and not serde on [`Resolution`] itself, for [`ScopeDto`]'s reason.
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionDto {
    /// «в полном разрешении».
    Full,
    /// «облегчённые модели».
    Display,
}

/// [`Scope`] as the window asks for it: «вся сборка» from the export menu, or «эта группа» from
/// the group inspector.
///
/// A mirror rather than serde and `ts-rs` on [`Scope`] itself, because the two are not the same
/// thing: [`Scope`] is an argument of a pure function that knows nothing of any window, and this
/// is a shape on a wire, tagged the way [`crate::protocol::ExportWhat`] is so that the window
/// reads one style of union throughout. Nothing of this ever reaches the engine's process — «в
/// Blender» is the shell's own work (A §9.2) — which is why it lives here and not in
/// [`crate::protocol`].
#[cfg_attr(feature = "ts", derive(ts_rs::TS), ts(export))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScopeDto {
    /// Every group of two or more.
    All,
    /// The one group the inspector has open.
    Group {
        /// Its index in [`AssemblyDto::groups`], as the window prints it.
        #[cfg_attr(feature = "ts", ts(type = "number"))]
        index: usize,
    },
}

impl From<ResolutionDto> for Resolution {
    /// One name for one name.
    fn from(dto: ResolutionDto) -> Self {
        match dto {
            ResolutionDto::Full => Self::Full,
            ResolutionDto::Display => Self::Display,
        }
    }
}

impl From<ScopeDto> for Scope {
    /// One name for one name. An index the assembly does not have is not refused here: it names
    /// no group, [`groups_of`] finds none, and the script is written with nothing in it — which
    /// is a Blender file that says «объектов: 0» rather than a command that failed.
    fn from(dto: ScopeDto) -> Self {
        match dto {
            ScopeDto::All => Self::All,
            ScopeDto::Group { index } => Self::Group(index),
        }
    }
}

/// One fragment as the script places it.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderFragment {
    /// The fragment's name (R §2): what the object is called in Blender.
    pub name: String,
    /// The original scan, under the run's input folder.
    pub source: PathBuf,
    /// The display mesh, `fragments/<name>.glb`.
    pub display: PathBuf,
    /// R §8's pose for this fragment in its group's frame, row-major — exactly the matrix the
    /// window draws it with, so Blender shows what «Сборка» shows.
    pub pose: [[f64; 4]; 4],
}

/// One assembled group as the script builds it: a Blender collection, and an empty to move it by.
#[derive(Clone, Debug, PartialEq)]
pub struct BlenderGroup {
    /// Its index in [`AssemblyDto::groups`], which is the number the window shows.
    pub index: usize,
    /// What the collection is called — «Группа N».
    pub label: String,
    /// Its fragments, in the assembly's order.
    pub fragments: Vec<BlenderFragment>,
}

/// The groups of `assembly` that `scope` asks for, with every path resolved (A §9.2).
///
/// The snapshot is the authority on file names, not the input folder as it is now: it is what the
/// run recorded, and a fragment whose scan has since gone is simply left out rather than written
/// into the script as a path that will fail to import. A fragment with no pose is left out for
/// the same reason — there is nowhere to put it — and a group that loses all of its fragments
/// this way is dropped, because an empty collection in Blender says nothing.
pub fn groups_of(
    assembly: &AssemblyDto,
    snapshot: &InputSnapshot,
    input: &Path,
    fragments_dir: &Path,
    scope: Scope,
) -> Vec<BlenderGroup> {
    let wanted = |index: usize, members: usize| match scope {
        Scope::All => members > 1,
        Scope::Group(only) => index == only,
    };
    let mut groups = Vec::new();
    for (index, group) in assembly.groups.iter().enumerate() {
        if !wanted(index, group.members.len()) {
            continue;
        }
        let fragments: Vec<BlenderFragment> = group
            .members
            .iter()
            .filter_map(|name| {
                let stamp = snapshot.files.iter().find(|f| &f.name == name)?;
                Some(BlenderFragment {
                    name: name.clone(),
                    source: input.join(&stamp.file),
                    display: fragments_dir.join(format!("{name}.glb")),
                    pose: *assembly.poses.get(name)?,
                })
            })
            .collect();
        if fragments.is_empty() {
            continue;
        }
        groups.push(BlenderGroup { index, label: format!("Группа {index}"), fragments });
    }
    groups
}

/// glTF's Y-up undone (A §9.2).
///
/// Blender's glTF importer rotates the vertices themselves — `(x, y, z)` of the file becomes
/// `(x, −z, y)` of the object — while our `fragments/*.glb` hold the scan's own coordinates with
/// no conversion applied, because the window draws them in the engine's frame. So an object
/// imported from one of ours carries `C·v` where we wanted `v`, and the pose is post-multiplied
/// by `C⁻¹`, which is this: the whole correction, once, on the object's `matrix_world`.
///
/// A `Full` import needs none of it — `wm.ply_import` and its two siblings are given the axes
/// explicitly and move nothing.
const GLTF_TO_SCAN: [[f64; 4]; 4] =
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// The comment every generated script opens with, up to the line that names the assembly.
const HEADER: &str = r##"# Сборка sherd-refit в Blender.
#
# Файл написан приложением — «Открыть в Blender». Он ничего не удаляет и ничего
# не трогает из того, что уже есть в файле: всё складывается в новую коллекцию
# с именем сборки, по коллекции на группу, по объекту на фрагмент.
#
# Позы — те же матрицы, что показывает «Сборка». Масштаб не меняется: единицы
# остаются единицами скана.
#
#     blender --python open_in_blender.py
#
"##;

/// What the script imports, and the version it insists on.
const PRELUDE: &str = r##"import os

import bpy
from mathutils import Matrix, Vector

# Операторов `wm.ply_import` и `wm.stl_import` в Blender 3.x нет вовсе, поэтому
# версия проверяется до первого импорта (A §9.2).
MIN_BLENDER = (4, 0, 0)

"##;

/// Why there is a `TITLE` at all.
const TITLE_NOTE: &str = "# Имя верхней коллекции: всё, что делает скрипт, лежит внутри неё.\n";

/// [`GLTF_TO_SCAN`]'s reason, for the person reading the script instead of this file.
const AXIS_NOTE: &str = r##"# Импортёр glTF переводит Y-up в Z-up прямо в вершинах: (x, y, z) -> (x, -z, y).
# Наши GLB хранят координаты скана как есть, поэтому поза домножается справа на
# обратное преобразование — иначе группа ляжет набок.
"##;

/// The row's spacing, which is the one number of the layout a person might want to change.
const GAP_NOTE: &str = r##"# Зазор между группами в ряду — доля ширины предыдущей группы.
GAP = 0.2

"##;

/// The Python that builds the scene (A §9.2). Pure: the same input gives the same text, which is
/// what `blender_golden.py` holds and what the golden test compares against.
///
/// The script is data plus a fixed body: `TITLE`, `GROUPS` and the axis constant are written
/// here, everything that runs is [`BODY`]. A reader who wants to know what will happen to their
/// file reads the body once; a reader who wants to know what goes where reads the data.
pub fn script(title: &str, groups: &[BlenderGroup], resolution: Resolution) -> String {
    let mut out = String::new();
    // A title with a newline in it would end the comment and start a line of Python: the comment
    // gets one line whatever the shell passed, and `TITLE` itself is an escaped literal.
    let single_line: String = title.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    out.push_str(&format!("{HEADER}# «{single_line}»\n\n{PRELUDE}"));
    out.push_str(&format!("{TITLE_NOTE}TITLE = {}\n\n", py_str(title)));
    out.push_str(&format!(
        "{AXIS_NOTE}GLTF_TO_SCAN = Matrix((\n{}\n))\n\n",
        py_rows(&GLTF_TO_SCAN, "    ")
    ));
    out.push_str(GAP_NOTE);
    if groups.is_empty() {
        out.push_str("GROUPS = []\n");
    } else {
        out.push_str("GROUPS = [\n");
        for group in groups {
            out.push_str(&py_group(group, resolution));
        }
        out.push_str("]\n");
    }
    out.push_str(BODY);
    out
}

/// Everything in the script that runs (A §9.2): the same text in every script, so that the part
/// which differs from export to export is the data above it and nothing else.
const BODY: &str = r##"

def _say(message):
    """Консоль Blender — единственное место, где скрипт может говорить подробно."""
    print("[sherd-refit] " + message)


def _popup(lines):
    """То же самое окном, когда окно есть: фоновый Blender обходится консолью."""
    manager = getattr(bpy.context, "window_manager", None)
    if manager is None:
        return

    def draw(self, _context):
        for line in lines:
            self.layout.label(text=line)

    try:
        manager.popup_menu(draw, title="sherd-refit", icon="INFO")
    except Exception as error:
        _say(str(error))


def _import(fragment):
    """Читает один файл и возвращает объекты, которых до него не было."""
    path = fragment["path"]
    if not os.path.isfile(path):
        raise IOError("файла нет: " + path)
    before = set(bpy.data.objects)
    kind = fragment["importer"]
    if kind == "ply":
        bpy.ops.wm.ply_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif kind == "obj":
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif kind == "stl":
        bpy.ops.wm.stl_import(filepath=path, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    return [obj for obj in bpy.data.objects if obj not in before]


def _adopt(objects, collection):
    """Переносит импортированное в коллекцию группы, куда бы импортёр его ни положил."""
    for obj in objects:
        for other in list(obj.users_collection):
            if other is not collection:
                other.objects.unlink(obj)
        if collection not in obj.users_collection:
            collection.objects.link(obj)


def _place(fragment, objects):
    """Имя фрагмента и его поза. Масштаб не трогаем: единицы остаются сканными."""
    pose = Matrix(fragment["matrix"])
    if fragment["importer"] == "gltf":
        pose = pose @ GLTF_TO_SCAN
    for obj in objects:
        obj.name = fragment["name"]
        # Вложенным объектам поза не нужна: они едут за своим корнем.
        if obj.parent not in objects:
            obj.matrix_world = pose


def _width(objects):
    """Ширина группы по X в мировых координатах: по ней считается сдвиг следующей."""
    edges = []
    for obj in objects:
        for corner in getattr(obj, "bound_box", ()):
            edges.append((obj.matrix_world @ Vector(corner)).x)
    return max(edges) - min(edges) if edges else 0.0


def build():
    """Собирает сцену и рассказывает, что получилось."""
    if bpy.app.version < MIN_BLENDER:
        message = "Нужен Blender %s или новее, а это %s." % (
            ".".join(str(part) for part in MIN_BLENDER),
            ".".join(str(part) for part in bpy.app.version),
        )
        _say(message)
        _popup([message])
        return

    view_layer = bpy.context.view_layer
    # Импортёры кладут объекты в активную коллекцию. Пусть это будет коллекция
    # сцены, а не чья-то чужая и, может быть, выключенная.
    view_layer.active_layer_collection = view_layer.layer_collection

    top = bpy.data.collections.new(TITLE)
    bpy.context.scene.collection.children.link(top)

    failed = []
    total = 0
    offset = 0.0
    for group in GROUPS:
        collection = bpy.data.collections.new(group["label"])
        top.children.link(collection)
        placed = []
        for fragment in group["fragments"]:
            try:
                objects = _import(fragment)
            except Exception as error:
                failed.append(fragment["name"] + " — " + str(error))
                continue
            if not objects:
                failed.append(fragment["name"] + " — импортёр ничего не добавил")
                continue
            _adopt(objects, collection)
            _place(fragment, objects)
            placed.extend(objects)
        total += len(placed)
        # Ширина считается до переноса: после него мировая матрица объекта
        # зависит от родителя, а тот пересчитывается не сразу.
        width = _width(placed)
        if len(GROUPS) > 1:
            empty = bpy.data.objects.new(group["label"], None)
            empty.empty_display_type = "PLAIN_AXES"
            collection.objects.link(empty)
            for obj in placed:
                if obj.parent is None:
                    obj.parent = empty
            empty.location.x = offset
            offset += width * (1.0 + GAP)

    _say("объектов: %d" % total)
    _say("масштаб не менялся: единицы — те же, что в сканах")
    lines = ["Объектов: %d." % total, "Масштаб не менялся: единицы те же, что в сканах."]
    if failed:
        _say("не удалось прочитать файлов: %d" % len(failed))
        for line in failed:
            _say("  " + line)
        lines.append("Не удалось прочитать: %d (подробности в консоли)." % len(failed))
    _popup(lines)


build()
"##;

/// One group as its entry of `GROUPS`.
fn py_group(group: &BlenderGroup, resolution: Resolution) -> String {
    let mut out =
        format!("    {{\n        \"label\": {},\n        \"fragments\": [\n", py_str(&group.label));
    for fragment in &group.fragments {
        let (path, importer) = importer_of(fragment, resolution);
        out.push_str("            {\n");
        out.push_str(&format!("                \"name\": {},\n", py_str(&fragment.name)));
        out.push_str(&format!("                \"path\": {},\n", py_path(path)));
        out.push_str(&format!("                \"importer\": {},\n", py_str(importer)));
        out.push_str(&format!(
            "                \"matrix\": (\n{}\n                ),\n",
            py_rows(&fragment.pose, "                    ")
        ));
        out.push_str("            },\n");
    }
    out.push_str("        ],\n    },\n");
    out
}

/// A 4×4 as Python rows, one per line, each indented by `indent` and each with its comma.
fn py_rows(rows: &[[f64; 4]; 4], indent: &str) -> String {
    rows.iter()
        .map(|row| {
            let cells: Vec<String> = row.iter().map(|value| py_float(*value)).collect();
            format!("{indent}({}),", cells.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// One number as Python reads it.
///
/// Rust's `{:?}` is the shortest text that round-trips, and every shape it produces — `1.0`,
/// `-0.0`, `1e-13` — is a Python float literal too. The three that are not are written as calls,
/// because a pose that arrived as `NaN` from a file on disk must not silently become a zero.
fn py_float(value: f64) -> String {
    if value.is_nan() {
        return "float(\"nan\")".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() { "float(\"inf\")" } else { "float(\"-inf\")" }
            .to_owned();
    }
    format!("{value:?}")
}

/// A path as a Python string literal.
///
/// Lossy on purpose: a path the OS will not give back as UTF-8 cannot be written into a UTF-8
/// source file at all, and a replacement character in a name is a file that fails to import and
/// is named in the script's own summary — which is better than no script.
fn py_path(path: &Path) -> String {
    py_str(&path.to_string_lossy())
}

/// One string as a double-quoted Python literal, with backslashes, quotes and control characters
/// escaped.
///
/// Not a raw string, although `r"…"` is what a Windows path wants to look like: a raw string
/// cannot hold a `"` and cannot end in a backslash, and both are things a folder a museum picked
/// may do. An ordinary literal with `\` doubled survives `C:\scans\керамика\a.ply`, a quote in a
/// name and a non-ASCII name alike — Python 3 source is UTF-8, so the name itself stays readable.
fn py_str(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if other.is_control() => {
                out.push_str(&format!("\\u{:04x}", u32::from(other)));
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Which of Blender's importers reads this fragment, and from which file.
///
/// `.off` is the case the spec calls out: the engine reads it and Blender has no importer for it
/// at all, so such a fragment goes in as its display mesh whatever the reviewer chose. Anything
/// whose extension is not one of the three Blender reads natively takes the same way out — the
/// display mesh always exists, because `Prepare` wrote one for every fragment of the collection.
fn importer_of(fragment: &BlenderFragment, resolution: Resolution) -> (&Path, &'static str) {
    if resolution == Resolution::Display {
        return (&fragment.display, "gltf");
    }
    let extension = fragment
        .source
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "ply" => (&fragment.source, "ply"),
        "obj" => (&fragment.source, "obj"),
        "stl" => (&fragment.source, "stl"),
        _ => (&fragment.display, "gltf"),
    }
}

/// Blender's executable, if this machine has one (A §9.2).
///
/// The setting wins when it points at a file, and **falls through** when it does not: a path a
/// colleague typed a year ago and an application they have since moved must not turn «Открыть в
/// Blender» into a permanent failure when the standard install is right there. A path that is
/// wrong and a setting that is empty come to the same thing, which is «look in the usual places».
///
/// Finding nothing is not an error either. The script is written first and the shell offers it in
/// its folder, so the export is never lost to a missing application.
pub fn find_blender(override_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = override_path
        && path.is_file()
    {
        return Some(path.to_owned());
    }
    candidates().into_iter().find(|path| path.is_file())
}

/// Where Blender installs itself, best first.
///
/// One list for every platform and no `cfg`: `/Applications/Blender.app` cannot exist on Windows
/// and `C:\Program Files` cannot exist on macOS, so a list that names all of them answers the
/// same on each — and it is a list that compiles and is read on the machine in front of you
/// rather than on one of the three. `blender` on `PATH` closes it, which is Linux's answer and,
/// on a Mac with a package manager, one more place worth looking.
fn candidates() -> Vec<PathBuf> {
    const BUNDLE: &str = "Blender.app/Contents/MacOS/Blender";
    let mut found = vec![Path::new("/Applications").join(BUNDLE)];
    for home in ["HOME", "USERPROFILE"].iter().filter_map(std::env::var_os) {
        found.push(Path::new(&home).join("Applications").join(BUNDLE));
    }
    let windows = std::env::var_os("ProgramFiles")
        .map_or_else(|| PathBuf::from(r"C:\Program Files"), PathBuf::from);
    found.extend(blender_foundation(&windows.join("Blender Foundation")));
    found.extend(on_path("blender"));
    found
}

/// Every `Blender Foundation\Blender <version>\blender.exe` of a Windows install, newest first.
///
/// Sorted by the version as numbers and not as text: as text `Blender 10.0` comes before
/// `Blender 4.5`, and it is the newer of the two, so the text order would answer this question
/// right by accident today and wrong the moment a version number reaches two digits. The path
/// breaks a tie, so that a folder nobody can parse does not make the answer depend on the order
/// the file system happened to list things in.
fn blender_foundation(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut versions: Vec<(Vec<u32>, PathBuf)> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let version = name.strip_prefix("Blender ")?;
            let parts: Vec<u32> = version.split('.').filter_map(|part| part.parse().ok()).collect();
            Some((parts, entry.path().join("blender.exe")))
        })
        .collect();
    versions.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    versions.into_iter().map(|(_, path)| path).collect()
}

/// `name` on every entry of `PATH`, in `PATH`'s own order.
fn on_path(name: &str) -> Vec<PathBuf> {
    std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).map(|dir| dir.join(name)).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::protocol::GroupDto;
    use crate::snapshot::FileStamp;

    /// The assembly the golden file is written from: two groups, an `.off` source that has to go
    /// in as its display mesh even at full resolution, a folder with a space and «керамика» in
    /// it, a quote inside a file name, and a Windows path with its backslashes.
    fn golden_groups() -> Vec<BlenderGroup> {
        let scans = "/Volumes/Архив/сканы керамика 2024";
        let fragments = "/Users/museum/Мои \"рабочие\" файлы/karas/fragments";
        // Built and not written out: `sqrt` and a division are correctly rounded by IEEE 754 on
        // every machine, so the sixteen digits they print are the same in CI as here — which a
        // `cos` would not be, and which a literal of sixteen digits could not be read.
        let half_root_three = 3.0_f64.sqrt() / 2.0;
        let third = 1.0_f64 / 3.0;
        vec![
            BlenderGroup {
                index: 0,
                label: "Группа 0".to_owned(),
                fragments: vec![
                    BlenderFragment {
                        name: "FY234008".to_owned(),
                        source: PathBuf::from(format!("{scans}/FY234008.ply")),
                        display: PathBuf::from(format!("{fragments}/FY234008.glb")),
                        pose: [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                    BlenderFragment {
                        name: "FY234012".to_owned(),
                        source: PathBuf::from(format!("{scans}/кусок \"12\".off")),
                        display: PathBuf::from(format!("{fragments}/FY234012.glb")),
                        pose: [
                            [half_root_three, -0.5, 0.0, 12.5],
                            [0.5, half_root_three, 0.0, -3.25],
                            [0.0, 0.0, 1.0, 0.125],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                ],
            },
            BlenderGroup {
                index: 2,
                label: "Группа 2".to_owned(),
                fragments: vec![
                    BlenderFragment {
                        name: "KP-7".to_owned(),
                        source: PathBuf::from(r"C:\Сканы\керамика\KP-7.STL"),
                        display: PathBuf::from(r"C:\ws\fragments\KP-7.glb"),
                        pose: [
                            [-1.0, 0.0, 0.0, 1e-13],
                            [0.0, 1.0, 0.0, -0.0],
                            [0.0, 0.0, -1.0, 250.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                    BlenderFragment {
                        name: "KP-8".to_owned(),
                        source: PathBuf::from(format!("{scans}/KP-8.obj")),
                        display: PathBuf::from(format!("{fragments}/KP-8.glb")),
                        pose: [
                            [1.0, 0.0, 0.0, -101.0625],
                            [0.0, third, -2.0 * third, 0.0],
                            [0.0, 2.0 * third, third, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                ],
            },
        ]
    }

    /// A §9.2's script, whole, as text — the only way to see it without Blender, which is not on
    /// this machine (A §11). `blender_golden.py` is real Python and `python3 -m py_compile` reads
    /// it in the same step of the plan that runs this test.
    #[test]
    fn the_script_is_the_golden_file() {
        let produced = script("karas — сборка", &golden_groups(), Resolution::Full);
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/blender_golden.py");
            std::fs::write(&path, &produced).expect("the golden file sits beside the module");
            // `include_str!` was baked in when this test was compiled, so comparing now would
            // still be comparing against the previous text. Rewriting it *is* this run.
            return;
        }
        assert_eq!(
            produced,
            include_str!("blender_golden.py"),
            "the script changed; read the difference and, if it is right, \
             UPDATE_GOLDEN=1 cargo test -p sherd-app-core --lib blender"
        );
    }

    /// Full resolution is about the *sources*: an `.off` has no Blender importer at all and goes
    /// in as its display mesh regardless, while «облегчённые модели» sends every fragment that
    /// way — and a `.STL` in capitals is still an STL.
    #[test]
    fn an_off_source_and_every_display_fragment_go_in_as_gltf() {
        let full = script("t", &golden_groups(), Resolution::Full);
        assert_eq!(full.matches(r#""importer": "gltf""#).count(), 1, "{full}");
        assert!(full.contains(r#""importer": "stl""#));
        assert!(full.contains(r#""path": "C:\\Сканы\\керамика\\KP-7.STL""#), "{full}");
        // The `.off` went in as its display mesh, and that path has a quote in it.
        assert!(full.contains(r#"Мои \"рабочие\" файлы"#), "{full}");

        let display = script("t", &golden_groups(), Resolution::Display);
        assert_eq!(display.matches(r#""importer": "gltf""#).count(), 4);
        assert!(display.contains(r#""path": "C:\\ws\\fragments\\KP-7.glb""#));
        // Nothing of the fixed body depends on the choice.
        assert!(display.ends_with(BODY));
    }

    /// A §9.2's two scopes. «Вся сборка» is the groups, not the leftovers: a group of one is a
    /// fragment nothing was found for. «Эта группа» is whichever one the inspector has open,
    /// whatever its size, and keeps its number.
    #[test]
    fn a_scope_takes_the_groups_it_asked_for_and_nothing_it_cannot_place() {
        let group = |members: &[&str]| GroupDto {
            members: members.iter().map(|&m| m.to_owned()).collect(),
            refined: false,
        };
        let eye = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let assembly = AssemblyDto {
            groups: vec![group(&["a", "b"]), group(&["c"]), group(&["d", "gone"])],
            poses: ["a", "b", "c", "d", "gone"].iter().map(|n| ((*n).to_owned(), eye)).collect(),
            ..AssemblyDto::default()
        };
        let stamp = |name: &str, file: &str| FileStamp {
            name: name.to_owned(),
            file: file.to_owned(),
            size: 1,
            mtime_ms: 0,
        };
        let snapshot = InputSnapshot {
            files: vec![
                stamp("a", "a.ply"),
                stamp("b", "b.obj"),
                stamp("c", "c.ply"),
                stamp("d", "d.off"),
            ],
            ..InputSnapshot::default()
        };
        let input = Path::new("/in");
        let display = Path::new("/ws/fragments");

        let all = groups_of(&assembly, &snapshot, input, display, Scope::All);
        assert_eq!(all.iter().map(|g| g.index).collect::<Vec<_>>(), [0, 2], "the singleton is out");
        assert_eq!(all[0].label, "Группа 0");
        assert_eq!(all[0].fragments[1].source, Path::new("/in/b.obj"));
        assert_eq!(all[0].fragments[1].display, Path::new("/ws/fragments/b.glb"));
        // `gone` has a pose and no scan: it is left out rather than written in as a path that
        // would fail to import.
        assert_eq!(all[1].fragments.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["d"]);

        let one = groups_of(&assembly, &snapshot, input, display, Scope::Group(1));
        assert_eq!(one.len(), 1);
        assert_eq!((one[0].index, one[0].label.as_str()), (1, "Группа 1"));
        assert!(groups_of(&assembly, &snapshot, input, display, Scope::Group(9)).is_empty());

        // A group whose every fragment is gone is no collection at all.
        let orphan = AssemblyDto {
            groups: vec![group(&["gone", "also-gone"])],
            poses: BTreeMap::new(),
            ..AssemblyDto::default()
        };
        assert!(groups_of(&orphan, &snapshot, input, display, Scope::All).is_empty());
    }

    /// [`GLTF_TO_SCAN`] is the whole of the reasoning that could not be checked in Blender, so it
    /// is checked here: the importer turns the file's `(x, y, z)` into the object's `(x, −z, y)`,
    /// and this matrix, applied to what the importer produced, gives the scan's own coordinates
    /// back.
    #[test]
    fn the_gltf_constant_undoes_what_the_importer_did_to_the_vertices() {
        let (x, y, z) = (1.5_f64, -2.25_f64, 7.0_f64);
        let imported = [x, -z, y, 1.0];
        let back: Vec<u64> = GLTF_TO_SCAN
            .iter()
            .map(|row| {
                let cell: f64 = row.iter().zip(imported).map(|(a, b)| a * b).sum();
                cell.to_bits()
            })
            .collect();
        assert_eq!(back, [x, y, z, 1.0].map(f64::to_bits));
    }

    /// A path a museum picked is not a path a generator may assume anything about (A §10: no
    /// panic, and nothing silently mangled). Every one of these has to come out of Python as the
    /// bytes it went in as.
    #[test]
    fn a_path_survives_being_written_into_python() {
        assert_eq!(py_str(r"C:\Сканы\a.ply"), r#""C:\\Сканы\\a.ply""#);
        assert_eq!(py_str("кусок \"12\".off"), r#""кусок \"12\".off""#);
        assert_eq!(py_str("two\nlines\tand a tab"), r#""two\nlines\tand a tab""#);
        assert_eq!(py_str("bell\u{7}"), r#""bell\u0007""#);
        // Every float shape the poses can take is a Python literal, the three that are not
        // literals included.
        assert_eq!(py_float(1.0), "1.0");
        assert_eq!(py_float(-0.0), "-0.0");
        assert_eq!(py_float(1e-13), "1e-13");
        assert_eq!(py_float(f64::NAN), r#"float("nan")"#);
        assert_eq!(py_float(f64::NEG_INFINITY), r#"float("-inf")"#);
        // A title with a newline in it stays one comment line and one string.
        let text = script("a\nb", &[], Resolution::Full);
        assert!(text.contains("# «a b»\n"), "{text}");
        assert!(text.contains(r#"TITLE = "a\nb""#), "{text}");
        assert!(text.contains("GROUPS = []"), "{text}");
    }

    /// A §9.2: the setting is where Blender is, until it is not — and then the usual places, not
    /// a failure. A path that names a file is taken as it stands.
    #[test]
    fn a_setting_that_points_at_nothing_falls_through_to_the_usual_places() {
        let missing = std::env::temp_dir().join("sherd-no-blender-here");
        std::fs::remove_file(&missing).ok();
        assert_ne!(find_blender(Some(&missing)), Some(missing.clone()));

        let real = std::env::temp_dir().join("sherd-fake-blender");
        std::fs::write(&real, b"#!/bin/sh\n").unwrap();
        assert_eq!(find_blender(Some(&real)), Some(real.clone()));
        std::fs::remove_file(&real).ok();

        // The Windows list is sorted by version and not by name: `Blender 10.0` is the newest.
        let root = std::env::temp_dir().join("sherd-blender-foundation");
        std::fs::remove_dir_all(&root).ok();
        for name in ["Blender 4.5", "Blender 10.0", "Blender 4.10", "Notes"] {
            std::fs::create_dir_all(root.join(name)).unwrap();
        }
        let found: Vec<String> = blender_foundation(&root)
            .iter()
            .filter_map(|path| path.parent()?.file_name()?.to_str().map(str::to_owned))
            .collect();
        assert_eq!(found, ["Blender 10.0", "Blender 4.10", "Blender 4.5"]);
    }
}
