//! `README.txt`: what is in the folder, for the person who opens it.
//!
//! In Russian, as the museum's own documentation is, and written per run: it lists the files this
//! run wrote and no others, names every group's fragments, and states the one convention every
//! matrix file shares. UTF-8 with a byte-order mark, so that Windows Notepad reads the Cyrillic.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::{BOM, Export, JoinRow, write_text};
use crate::error::Result;
use crate::{CORE_VERSION, GIT_COMMIT};

/// The file, under the output directory.
pub const README_FILE: &str = "README.txt";

/// Writes `README.txt` for a run that wrote `written`; a path outside `out_dir` — `--measure`'s
/// file, say — is left out.
pub fn write_readme(
    out_dir: &Path,
    ex: &Export<'_>,
    joins: &[JoinRow],
    written: &[PathBuf],
) -> Result<PathBuf> {
    let mut files: Vec<String> = written
        .iter()
        .filter_map(|p| p.strip_prefix(out_dir).ok())
        // `<out>/../x` strips to `../x`, which is not in the folder: only plain names count.
        .filter(|p| p.components().all(|c| matches!(c, std::path::Component::Normal(_))))
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .collect();
    files.push(README_FILE.to_owned());
    let cache = out_dir.join("cache").is_dir();
    write_text(&out_dir.join(README_FILE), &readme(ex, joins, &files, cache))
}

/// The text, for a run that wrote `files` (relative, `/`-separated) and kept a cache or not.
pub fn readme(ex: &Export<'_>, joins: &[JoinRow], files: &[String], cache: bool) -> String {
    let mut s = String::from(BOM);
    let assembled: Vec<&Vec<u32>> = ex.groups.iter().filter(|g| g.len() > 1).collect();
    let in_groups: usize = assembled.iter().map(|g| g.len()).sum();
    let singles: Vec<&str> = ex
        .groups
        .iter()
        .filter(|g| g.len() == 1)
        .map(|g| ex.names[g[0] as usize].as_str())
        .collect();
    let n = ex.names.len();

    let _ = writeln!(s, "СБОРКА ФРАГМЕНТОВ: {}", ex.collection);
    let _ = writeln!(s, "sherd-refit {CORE_VERSION} ({GIT_COMMIT})");
    s.push('\n');
    let _ = writeln!(
        s,
        "{n} {}. Собрано {} {} ({in_groups} {}), не собрано {}.",
        plural(n, "фрагмент", "фрагмента", "фрагментов"),
        assembled.len(),
        plural(assembled.len(), "группа", "группы", "групп"),
        plural(in_groups, "фрагмент", "фрагмента", "фрагментов"),
        singles.len(),
    );
    let used = joins.iter().filter(|j| j.in_assembly).count();
    if ex.tiers {
        let confirmed = joins.iter().filter(|j| j.band == "confirmed").count();
        let _ = writeln!(
            s,
            "Стыков: подтверждённых {confirmed}, вероятных {}; сборка построена по {used}. Все — в joins.csv.",
            joins.len() - confirmed,
        );
    } else {
        let _ = writeln!(
            s,
            "Стыков принято {}; сборка построена по {used}. Все — в joins.csv.",
            joins.len()
        );
    }
    let _ = writeln!(
        s,
        "Толщина стенки (медиана по коллекции): {:.2} — в единицах исходных файлов.",
        ex.thickness
    );
    s.push('\n');
    s.push_str(
        "Номер фрагмента — это имя его исходного файла без расширения. Под этим именем фрагмент\n\
         стоит везде: в viewer.html, в scene.glb, в placed/, в таблицах и в отчёте.\n\n",
    );

    s.push_str("С ЧЕГО НАЧАТЬ\n");
    if files.iter().any(|f| f == "viewer.html") {
        s.push_str(
            "  Откройте viewer.html двойным щелчком (Chrome, Edge, Firefox или Safari; интернет не\n\
             \x20 нужен). Наведите курсор на фрагмент — появится его номер. Щелчок — подробности и\n\
             \x20 стыки, двойной щелчок — приблизить, клавиша L — подписать все фрагменты сразу.\n",
        );
    } else {
        s.push_str("  Откройте report.md (подробный отчёт) и preview_<k>.png (картинки групп).\n");
    }
    s.push('\n');

    s.push_str("ГРУППЫ\n");
    for (k, g) in ex.groups.iter().enumerate().filter(|(_, g)| g.len() > 1) {
        let names: Vec<&str> = g.iter().map(|&m| ex.names[m as usize].as_str()).collect();
        let head = format!(
            "  Группа {k} — {} {}: ",
            g.len(),
            plural(g.len(), "фрагмент", "фрагмента", "фрагментов")
        );
        wrap(&mut s, &head, &names);
    }
    if !singles.is_empty() {
        let head = format!(
            "  Не собраны — {} {}: ",
            singles.len(),
            plural(singles.len(), "фрагмент", "фрагмента", "фрагментов")
        );
        wrap(&mut s, &head, &singles);
    }
    if assembled.is_empty() {
        s.push_str("  Ни одной группы: уверенных стыков не нашлось (см. вероятные в joins.csv).\n");
    }
    s.push('\n');

    files_section(&mut s, files, cache);
    s.push_str(MATRICES);
    s
}

/// `ФАЙЛЫ В ЭТОЙ ПАПКЕ`: one row per kind of file the run wrote, a directory with its count.
fn files_section(s: &mut String, files: &[String], cache: bool) {
    s.push_str("ФАЙЛЫ В ЭТОЙ ПАПКЕ\n");
    let mut kinds: BTreeMap<usize, (String, usize)> = BTreeMap::new();
    for f in files {
        let (kind, rank) = kind_of(f);
        kinds.entry(rank).or_insert_with(|| (kind, 0)).1 += 1;
    }
    if cache {
        kinds.entry(rank_of("cache/")).or_insert_with(|| ("cache/".to_owned(), 0));
    }
    for (kind, count) in kinds.values() {
        let name = if kind.ends_with('/') && *count > 0 {
            format!("{kind} ({count} {})", plural(*count, "файл", "файла", "файлов"))
        } else if *count > 1 {
            format!("{kind} ({count})")
        } else {
            kind.clone()
        };
        let _ = writeln!(s, "  {name}");
        let about = describe(kind);
        if !about.is_empty() {
            let _ = writeln!(s, "      {about}");
        }
    }
    s.push('\n');
}

/// The matrix convention and how to use it, the same for every run.
const MATRICES: &str = "\
МАТРИЦЫ: КАК ПРИМЕНИТЬ В ДРУГОЙ ПРОГРАММЕ
  Матрица M фрагмента (4×4) переводит координаты его ИСХОДНОГО файла в положение в сборке:
      [x' y' z' 1] = M · [x y z 1]   (точки — столбцы)
  Поворот — левый верхний блок 3×3, сдвиг — последний столбец (m03, m13, m23), нижняя строка
  0 0 0 1. Масштаба нет. Единицы — те же, что в исходных файлах. Одна и та же матрица записана
  в transforms.json, transforms.csv, matrices/<имя>.txt и в узле фрагмента в scene.glb.

  Каждая группа отцентрована в начале координат, поэтому две группы, открытые вместе, лягут
  друг на друга: открывайте по одной группе. В viewer.html и scene.glb группы разнесены.

  Geomagic Wrap, MeshLab, CloudCompare, Blender:
    • проще всего — импортировать файлы одной группы из placed/: они уже стоят на своих
      местах, в полном разрешении и с цветом, и каждый станет отдельным объектом с именем
      файла, то есть с номером фрагмента;
    • CloudCompare: Edit → Apply transformation → ASCII file → matrices/<имя>.txt;
    • MeshLab: Filters → Normals, Curvatures and Orientation → Matrix: Set/Copy
      Transformation, вставить четыре строки из matrices/<имя>.txt;
    • Blender: File → Import → glTF 2.0 → scene.glb.

  Python (в том числе встроенный в Geomagic Wrap) — все матрицы из transforms.csv:
      import csv
      with open(r\"ПУТЬ\\transforms.csv\", newline=\"\", encoding=\"utf-8-sig\") as f:
          for row in csv.DictReader(f):
              M = [[float(row[\"m%d%d\" % (r, c)]) for c in range(4)] for r in range(4)]
              # row[\"fragment\"], row[\"source_file\"], row[\"group\"],
              # row[\"assembled\"] == \"1\" — фрагмент собран

  В Excel таблицы открываются через «Данные → Из текста/CSV» (разделитель — запятая).

НЕ СОБРАН — НЕ ЗНАЧИТ «НЕ ПОДХОДИТ»
  Это значит, что программа не нашла стыка, в котором уверена. Вероятные стыки — в joins.csv
  (band = probable) и в report.md; их стоит проверить глазами.
";

/// The row a file is listed under, and where that row goes in the list.
fn kind_of(file: &str) -> (String, usize) {
    let kind = match file.split_once('/') {
        Some((dir, _)) => format!("{dir}/"),
        None if file.starts_with("assembly_")
            && Path::new(file).extension().is_some_and(|e| e.eq_ignore_ascii_case("ply")) =>
        {
            "assembly_<k>.ply".to_owned()
        }
        None if file.starts_with("preview_") && file != "preview_segmentation.png" => {
            "preview_<k>.png".to_owned()
        }
        None => file.to_owned(),
    };
    let rank = rank_of(&kind);
    (kind, rank)
}

const ORDER: [&str; 16] = [
    "viewer.html",
    "scene.glb",
    "report.md",
    "report.json",
    "transforms.json",
    "transforms.csv",
    "joins.csv",
    "matrices/",
    "placed/",
    "assembly_<k>.ply",
    "preview_<k>.png",
    "preview_segmentation.png",
    "review/",
    "measure.json",
    "cache/",
    "README.txt",
];

fn rank_of(kind: &str) -> usize {
    ORDER.iter().position(|k| *k == kind).unwrap_or_else(|| {
        // An unknown file goes after the known ones, in name order.
        ORDER.len()
            + kind
                .bytes()
                .fold(0_usize, |h, b| h.wrapping_mul(31).wrapping_add(usize::from(b)) % 1_000_003)
    })
}

fn describe(kind: &str) -> &'static str {
    match kind {
        "viewer.html" => {
            "просмотр сборки в браузере; наведите курсор на фрагмент — появится его номер"
        }
        "scene.glb" => {
            "та же сцена для 3D-программ (glTF 2.0): облегчённые меши, «группа → фрагмент», у каждого\n      \
             фрагмента его имя и его матрица"
        }
        "report.md" => "подробный отчёт (на английском): фрагменты, группы, стыки и их оценки",
        "report.json" => "то же в машинном виде, со всеми кандидатами",
        "transforms.json" => "матрицы, группы и параметры запуска (JSON)",
        "transforms.csv" => {
            "матрица каждого фрагмента одной строкой (Excel, Python, скрипты Geomagic)"
        }
        "joins.csv" => "найденные стыки: пара, уверенность, вошёл ли в сборку, оценки, картинка",
        "matrices/" => {
            "по файлу на собранный фрагмент: матрица 4×4 текстом (CloudCompare, MeshLab)"
        }
        "placed/" => "собранные фрагменты в полном разрешении, уже на своих местах",
        "assembly_<k>.ply" => {
            "группа одним склеенным мешем (имена фрагментов в нём не сохраняются)"
        }
        "preview_<k>.png" => "группа в четырёх видах; подпись сверху — какой фрагмент каким цветом",
        "preview_segmentation.png" => "все фрагменты в ряд; красным — найденная поверхность излома",
        "review/" => "картинки стыков для проверки глазами, по одной на пару",
        "cache/" => {
            "служебная: предобработка фрагментов для быстрого повторного запуска; можно удалить"
        }
        "README.txt" => "этот файл",
        _ => "",
    }
}

/// `head` and then `names`, comma-separated, wrapped under the head's indent at 100 columns.
fn wrap(s: &mut String, head: &str, names: &[&str]) {
    let indent = "      ";
    let mut line = head.to_owned();
    let mut first = true;
    for name in names {
        let piece = if first { (*name).to_owned() } else { format!(", {name}") };
        if !first && line.chars().count() + piece.chars().count() > 100 {
            line.push(',');
            s.push_str(&line);
            s.push('\n');
            line = format!("{indent}{name}");
        } else {
            line.push_str(&piece);
        }
        first = false;
    }
    s.push_str(&line);
    s.push('\n');
}

/// Russian's three plural forms: 1 фрагмент, 2 фрагмента, 5 фрагментов (and 11–14 take the last).
fn plural(n: usize, one: &'static str, few: &'static str, many: &'static str) -> &'static str {
    let (last, tens) = (n % 10, n % 100);
    if last == 1 && tens != 11 {
        one
    } else if (2..=4).contains(&last) && !(12..=14).contains(&tens) {
        few
    } else {
        many
    }
}

#[cfg(test)]
mod tests {
    use super::{kind_of, plural, readme};
    use crate::export::Export;
    use nalgebra::Matrix4;
    use std::path::PathBuf;

    #[test]
    fn russian_plurals() {
        let f = |n| plural(n, "файл", "файла", "файлов");
        assert_eq!(
            [f(1), f(2), f(5), f(11), f(12), f(21), f(22), f(25), f(111)],
            ["файл", "файла", "файлов", "файлов", "файлов", "файл", "файла", "файлов", "файлов"]
        );
    }

    #[test]
    fn files_are_listed_by_kind_and_groups_by_name() {
        let names: Vec<String> = ["FY1", "FY2", "FY3"].map(String::from).to_vec();
        let sources: Vec<PathBuf> =
            names.iter().map(|n| PathBuf::from(format!("{n}.ply"))).collect();
        let poses = vec![Matrix4::identity(); 3];
        let groups = vec![vec![0, 1], vec![2]];
        let ex = Export {
            collection: "set",
            names: &names,
            sources: &sources,
            poses: &poses,
            groups: &groups,
            candidates: &[],
            used: &[],
            tiers: true,
            review: None,
            thickness: 2.0,
        };
        let files: Vec<String> =
            ["viewer.html", "placed/FY1.ply", "placed/FY2.ply", "preview_0.png", "README.txt"]
                .map(String::from)
                .to_vec();
        let text = readme(&ex, &[], &files, true);
        assert!(text.starts_with('\u{feff}'), "a BOM for Notepad");
        assert!(text.contains("Группа 0 — 2 фрагмента: FY1, FY2"));
        assert!(text.contains("Не собраны — 1 фрагмент: FY3"));
        assert!(text.contains("placed/ (2 файла)"));
        assert!(text.contains("cache/"));
        assert!(!text.contains("scene.glb ("), "only what the run wrote is listed as a file");
        assert_eq!(kind_of("assembly_3.ply").0, "assembly_<k>.ply");
        assert_eq!(kind_of("review/a__b.png").0, "review/");
    }
}
