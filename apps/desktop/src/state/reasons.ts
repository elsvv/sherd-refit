import type { TFunction } from "i18next";

import type { CandidateRow } from "../ipc/bindings/CandidateRow";

/**
 * Why the engine did not confirm a join, in words a reviewer can act on (A §8.3).
 *
 * The engine writes its refusals for itself — `tight 0.2767 < 0.35`, `no arm: support 0 < 1 and
 * margin 1.00 < 2` — and they are the most useful thing on the whole screen: they say what to
 * look at before deciding. They are also unreadable to anyone who has not read R §6. This module
 * is the one place that knows both, and it is pure and tested because the alternative is a
 * sentence that quietly stops matching the engine's own wording.
 *
 * **What it relies on**, from `sherd_core::tiers::Thresholds::refusals`: a numeric test is
 * `<name> <value> < <limit>` or `… > <limit>`; a redraw prefixes its own refusal with
 * `redraw <n>: `; the arms are one line beginning `no arm: support <n> < <m> and `. Anything
 * outside those shapes is shown **as the engine wrote it** rather than dropped: a build of the
 * engine that says something new must still say it on the screen.
 */

/** The one line that says why neither distinguishing arm reached this candidate. */
const NO_ARM = "no arm: ";

/**
 * `tight 0.2767 < 0.35`, and the five other scores written the same way — `{name} {value:.4}`
 * against a limit printed as the engine holds it (`0.35`, `5`, `0`).
 */
const COMPARISON = /^(tight|gap|seam|cont_n|pen|slide) (-?[\d.]+(?:e[-+]?\d+)?) [<>] (\S+)$/u;

/** `redraw 1: …` — R §6 run again at this pose on a redrawn collection. */
const REDRAW = /^redraw \d+: (.+)$/u;

/** `determined 1e-3 deg > 1e-6 deg`, printed in exponent form because the range is that small. */
const DETERMINED = /^determined (\S+) deg > (\S+) deg$/u;

/** `redraws accepted 1 < 2`. */
const REDRAWS_ACCEPTED = /^redraws accepted (\d+) < (\d+)$/u;

/** The arms line, split into the support half (always the same shape) and the half that varies. */
const ARMS = /^no arm: support \d+ < \d+ and (.+)$/u;

/** `margin 1.00 < 2` — how much better this placement is than the pair's second one. */
const MARGIN = /^margin ([\d.]+) < (\S+)$/u;

/** `the second placement is 3.69 t away, under 5`. */
const RIVAL_NEAR = /^the second placement is ([\d.]+) t away, under (\S+)$/u;

/** `re-search agreed 1/2 < 2`. */
const RESEARCH = /^re-search agreed (\d+)\/(\d+) < \d+$/u;

/**
 * How many decimals each score is worth reading at. Not one number for all of them: `tight` is a
 * fraction of a surface and two decimals is all anyone can judge, `gap` is thousandths of `t`,
 * and a seam length of `4.0000` reads as false precision.
 */
const DECIMALS: Record<string, number> = { tight: 2, cont_n: 2, gap: 3, seam: 1, slide: 2 };

/** A number the engine printed, or `null` when what stands there is not one (nothing panics). */
function number(text: string | undefined): number | null {
  if (text === undefined) {
    return null;
  }
  const value = Number.parseFloat(text);
  return Number.isFinite(value) ? value : null;
}

/** One score's sentence, or `null` when its value will not read as a number. */
function comparison(name: string, value: string | undefined, limit: string | undefined, t: TFunction): string | null {
  // `pen` is the one score whose number says nothing to a reviewer: any penetration at all is the
  // refusal, and «0.0008» is not a fact anyone can act on.
  if (name === "pen") {
    return t("reason.pen");
  }
  const measured = number(value);
  const decimals = DECIMALS[name];
  if (measured === null || decimals === undefined || limit === undefined) {
    return null;
  }
  return t(`reason.${name}`, { value: measured.toFixed(decimals), limit });
}

/** The half of the arms line that varies, as a sentence; `null` for a shape this build knows not. */
function arm(text: string, t: TFunction): string | null {
  const margin = MARGIN.exec(text);
  if (margin !== null) {
    const measured = number(margin[1]);
    return measured === null ? null : t("reason.arm_margin", { margin: measured.toFixed(1), least: margin[2] });
  }
  const near = RIVAL_NEAR.exec(text);
  if (near !== null) {
    const away = number(near[1]);
    return away === null ? null : t("reason.arm_rival", { away: away.toFixed(1), least: near[2] });
  }
  const research = RESEARCH.exec(text);
  if (research !== null) {
    const of = number(research[2]);
    return of === null ? null : t("reason.arm_research", { agreed: research[1], count: of });
  }
  if (text === "no second placement to beat") {
    return t("reason.arm_no_rival");
  }
  if (text.startsWith("the second placement is one R §6.5 would accept")) {
    return t("reason.arm_rival_accepted");
  }
  return null;
}

/** One refusal as a sentence, or `null` when this build does not know that shape. */
function sentence(text: string, t: TFunction): string | null {
  const redraw = REDRAW.exec(text);
  if (redraw !== null) {
    // The redraw's own refusal, wrapped: `redraw_refusal` reports at most one, so the sentence
    // need not say which of the two draws it was to be unambiguous.
    const what = sentence(redraw[1] ?? "", t);
    return what === null ? null : t("reason.redraw", { what });
  }
  const scored = COMPARISON.exec(text);
  if (scored !== null) {
    return comparison(scored[1] ?? "", scored[2], scored[3], t);
  }
  if (text === "pen: penetration not measurable, a fragment is not watertight" || text === "penetration not measurable") {
    // R §6.4: an open mesh has no inside, so the test did not fail — it could not be run.
    return t("reason.pen_unavailable");
  }
  if (text === "slide: no shared seam to slide along") {
    return t("reason.slide_none");
  }
  if (text === "determined: not probed") {
    return t("reason.determined_unprobed");
  }
  const determined = DETERMINED.exec(text);
  if (determined !== null) {
    return t("reason.determined", { value: determined[1], limit: determined[2] });
  }
  const redraws = REDRAWS_ACCEPTED.exec(text);
  if (redraws !== null) {
    const least = number(redraws[2]);
    return least === null ? null : t("reason.redraws_accepted", { accepted: redraws[1], count: least });
  }
  const arms = ARMS.exec(text);
  if (arms !== null) {
    const half = arm(arms[1] ?? "", t);
    // «support 0 < 1» is the same in every shipped threshold set (`min_support` is 1, and the
    // engine's own test asserts it), so the first half is one sentence and only the second varies.
    return half === null ? null : t("reason.no_arm", { arm: half });
  }
  return null;
}

/** One refusal as a sentence, falling back to the engine's own words. */
function explainOne(text: string, t: TFunction): string {
  return sentence(text, t) ?? text;
}

/**
 * Every refusal of `Evidence::failed`, in the engine's own order, as sentences (A §8.3's list
 * under «Почему не подтверждён сам»).
 */
export function explain(failed: readonly string[], t: TFunction): string[] {
  return failed.map((text) => explainOne(text, t));
}

/** Which arm confirmed a join, in words (A §8.3); an arm this build knows not, as it stands. */
function armWord(name: string, t: TFunction): string {
  switch (name) {
    case "support":
      return t("reason.by_support");
    case "margin":
      return t("reason.by_margin");
    case "research":
      return t("reason.by_research");
    default:
      return name;
  }
}

/**
 * The one sentence the inspector puts after «Почему не подтверждён сам» (A §8.3).
 *
 * The arms line is preferred over every other refusal because it is what «сам» is about: a
 * candidate can score perfectly and still be held back because nothing else agrees with it, and
 * that is the sentence the mock-up shows. A join the engine *did* confirm gets the other half of
 * the block — what confirmed it.
 */
export function headline(row: CandidateRow, t: TFunction): string {
  const failed = row.evidence?.failed ?? [];
  if (failed.length > 0) {
    return explainOne(failed.find((text) => text.startsWith(NO_ARM)) ?? failed[0] ?? "", t);
  }
  if (row.tier === "confirmed") {
    const arm = row.evidence?.arm;
    return arm === undefined ? t("reason.headline_confirmed") : t("reason.headline_arm", { arm: armWord(arm, t) });
  }
  // No refusals and not confirmed: the tier pass never got to this candidate (`evidence: null`),
  // or the run threw the pair out before it (A §8.3's third band).
  return t(row.tier === "rejected" ? "reason.headline_rejected" : "reason.headline_unprobed");
}
