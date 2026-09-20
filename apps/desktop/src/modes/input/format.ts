import type { TFunction } from "i18next";

import type { Warning } from "../../ipc/bindings/Warning";
import type { Language } from "../../state/ui";

/**
 * How the «Вход» mode writes numbers (A §7.3). Everything here is built by hand rather than with
 * `Intl`: a webview's ICU data is not the same on every machine, and a collection size that reads
 * `9,8 GB` on one and `9.8 ГБ` on another is a bug report nobody can reproduce. The one place
 * `Intl` is still right is a date, which this module has none of.
 */

/** U+202F, the thousands separator Russian typography uses — thin, and never a line break. */
const NARROW_NBSP = " ";

/**
 * A number the engine did not give. Punctuation, identical in both languages, so it needs no
 * translation — and it is honest where a `0` would be a lie about a fragment nobody measured.
 */
const UNKNOWN = "—";

/**
 * Decimal units, as Finder and Explorer count them (1 kB = 1000 B): the file sizes shown here are
 * the ones the OS shows for the very same folder, and matching the OS matters more than matching
 * `ls -l`.
 */
const UNITS: Record<Language, readonly string[]> = {
  ru: ["Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ"],
  en: ["B", "KB", "MB", "GB", "TB", "PB"],
};

/** From which step a tenth is worth showing: bytes and kilobytes are whole, megabytes up are not. */
const FIRST_FRACTIONAL_STEP = 2;

/**
 * The one piece of `Intl` this module does use, and for a question that has nothing to do with
 * language: where a string may be cut. Locale-independent on purpose — the default locale is
 * passed as `undefined` because grapheme boundaries are the same everywhere that matters here.
 */
const GRAPHEMES = new Intl.Segmenter(undefined, { granularity: "grapheme" });

/**
 * One measurement of the inspector's key–value list: one decimal, the way the engine prints its
 * own `{:.1}` rows, or `—` where a fragment carries no number at all.
 */
export function formatDecimal(n: number): string {
  return Number.isFinite(n) ? n.toFixed(1) : UNKNOWN;
}

/**
 * `9.8 ГБ`, `1.1 МБ`, `512 B` — the size of one scan or of the whole folder.
 *
 * The decimal point stays a point in both languages: the numbers beside it (the wall thickness,
 * the resolution) come out of the engine's own `{:.1}` formatting, and one collection listing
 * with two decimal marks in it looks like a fault in the app.
 */
export function formatBytes(n: number, lang: Language): string {
  const units = UNITS[lang];
  let value = Number.isFinite(n) && n > 0 ? n : 0;
  let unit = units[0] ?? "";
  let step = 0;
  for (let i = 1; i < units.length && value >= 1000; i += 1) {
    value /= 1000;
    unit = units[i] ?? unit;
    step = i;
  }
  const text = step >= FIRST_FRACTIONAL_STEP ? value.toFixed(1) : String(Math.round(value));
  return `${text} ${unit}`;
}

/**
 * `1 495 166` in Russian (with a narrow no-break space) and `1,495,166` in English — a face count
 * is read, not computed, and seven unbroken digits cannot be read.
 */
export function formatCount(n: number, lang: Language): string {
  const value = Number.isFinite(n) ? Math.round(n) : 0;
  const separator = lang === "ru" ? NARROW_NBSP : ",";
  const digits = Math.abs(value).toString();
  const grouped = digits.replace(/\B(?=(?:\d{3})+$)/gu, separator);
  return value < 0 ? `−${grouped}` : grouped;
}

/**
 * A fraction as whole percent: `0.1209` is `12 %`. The space before the sign is the Russian rule
 * and does no harm in English, and a whole percent is all the precision «12 % площади» carries.
 */
export function formatPercent(f: number): string {
  return Number.isFinite(f) ? `${String(Math.round(f * 100))} %` : UNKNOWN;
}

/**
 * What one warning means, in words the reviewer can act on (A §7.3).
 *
 * The thickness sentence quotes the collection's median beside the fragment's own wall and says
 * by how much the two differ, because «78.2» alone means nothing without what is normal here. A
 * median that is not a positive number is not a collection anyone measured, so the sentence drops
 * the comparison rather than printing a division by zero.
 */
export function warningText(w: Warning, t: TFunction): string {
  switch (w.warning) {
    case "thickness_outlier": {
      const thickness = formatDecimal(w.thickness);
      if (!Number.isFinite(w.median) || w.median <= 0) {
        return t("warning.thickness_outlier_alone", { thickness });
      }
      const deviation = (w.thickness - w.median) / w.median;
      const percent = `${deviation < 0 ? "−" : "+"}${formatPercent(Math.abs(deviation))}`;
      return t("warning.thickness_outlier", { thickness, median: formatDecimal(w.median), percent });
    }
    case "not_watertight":
      // R §6.4: an open mesh has no inside, so the engine cannot tell a join from a coincidence.
      return t("warning.not_watertight");
  }
}

/**
 * `FZ234001-…-02`, `/Volumes/…/karas_reduced` — a name or a path cut in the middle, because both
 * carry their meaning at the ends: a fragment's serial differs from its neighbour's in the last
 * characters, and a folder is recognised by its own name and not by the volume it sits on. CSS
 * can only ellipsise at one end, so this is done in the string.
 */
export function middleEllipsis(text: string, max: number): string {
  // Grapheme clusters, not UTF-16 units: macOS hands filenames back decomposed, so «карась» can
  // be nine code units, and a cut between a letter and its combining accent renders as a box.
  const parts = Array.from(GRAPHEMES.segment(text), (part) => part.segment);
  if (max < 3 || parts.length <= max) {
    return text;
  }
  const head = Math.ceil((max - 1) / 2);
  const tail = max - 1 - head;
  return `${parts.slice(0, head).join("")}…${parts.slice(parts.length - tail).join("")}`;
}

/**
 * The three sides of a fragment's bounding box, `149.3 × 257.5 × 221.4`. The engine works in the
 * scans' own units and never claims they are millimetres, so neither does this.
 */
export function formatExtent(extent: readonly [number, number, number]): string {
  return extent.map((side) => formatDecimal(side)).join(" × ");
}
