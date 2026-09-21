import type { TFunction } from "i18next";
import { createInstance } from "i18next";
import { beforeAll, describe, expect, it } from "vitest";

import type { CandidateRow } from "../ipc/bindings/CandidateRow";
import ru from "../i18n/ru.json";
import { explain, headline } from "./reasons";

/**
 * The real Russian wording, not a stub `t`: what is being tested is that every shape the engine
 * writes reaches a sentence, and a stub that echoes its key would pass while `ru.json` was empty.
 * A private instance rather than `src/i18n`, because that one reaches for `window` to remember
 * the language and there is no window in a node test.
 */
let t: TFunction;

beforeAll(async () => {
  const i18n = createInstance();
  await i18n.init({
    lng: "ru",
    resources: { ru: { translation: ru } },
    interpolation: { escapeValue: false },
  });
  t = i18n.t.bind(i18n);
});

/** The one sentence `failed` came to, for a header shape that must make exactly one. */
function one(failure: string): string {
  const sentences = explain([failure], t);
  expect(sentences).toHaveLength(1);
  return sentences[0] ?? "";
}

/** A candidate row with nothing but what [`headline`] reads. */
function row(overrides: Partial<CandidateRow> = {}): CandidateRow {
  return {
    a: "FY234111",
    b: "FY234058",
    pose: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    score: 8.1,
    scores: {
      tightA: 0.31,
      tightB: 0.28,
      tight: 0.28,
      gapA: 0.01,
      gapB: 0.012,
      gap: 0.012,
      contactA: 37,
      contactB: 34,
      contact: 34,
      seam: 19.7,
      gap_limit: 0.031,
      tight_delta: 0.01,
      cont: 0.1,
      cont_n: 0.96,
      pen: 0,
      pen_depth: 0,
      brk: 0.62,
      brk_best: 0.78,
    },
    tier: "probable",
    used: false,
    evidence: null,
    ...overrides,
  };
}

/** The evidence of a candidate that failed `failed` and reached no arm. */
function held(failed: string[]): CandidateRow["evidence"] {
  return {
    placements: 1,
    resample_tight_min: 0.24,
    resample_gap_max: 0.038,
    resample_accept: 1,
    support: 0,
    failed,
  };
}

describe("explain (A §8.3: the engine's refusals as sentences)", () => {
  it("says the number and its limit for every score the tiers floor or cap", () => {
    expect(one("tight 0.2767 < 0.35")).toBe("плотное прилегание 0.28 при нужных 0.35");
    expect(one("gap 0.0358 > 0.015")).toBe("зазор 0.036 t при допустимых 0.015 t");
    expect(one("seam 4.0000 < 5")).toBe("шов 4.0 t короче нужных 5 t");
    expect(one("cont_n 0.8485 < 0.9")).toBe("сплошность шва 0.85 при нужных 0.9");
    expect(one("slide 0.7200 > 0.1")).toContain("вдоль шва");
  });

  it("says penetration in words, because its number means nothing to a reviewer", () => {
    expect(one("pen 0.0008 > 0")).toBe("поверхности проникают друг в друга");
    expect(one("pen: penetration not measurable, a fragment is not watertight")).toBe(
      "проникание нельзя проверить: меш одного из фрагментов не замкнут",
    );
  });

  it("marks a refusal that came from a redraw as one, whatever the redraw refused", () => {
    expect(one("redraw 1: pen 0.0006 > 0")).toBe(
      "на повторной выборке точек: поверхности проникают друг в друга",
    );
    expect(one("redraw 2: tight 0.2767 < 0.35")).toBe(
      "на повторной выборке точек: плотное прилегание 0.28 при нужных 0.35",
    );
    expect(one("redraw 1: penetration not measurable")).toBe(
      "на повторной выборке точек: проникание нельзя проверить: меш одного из фрагментов не замкнут",
    );
  });

  it("says what could not be probed at all", () => {
    expect(one("slide: no shared seam to slide along")).toBe(
      "сдвиг вдоль шва не проверить: у пары нет общего шва",
    );
    expect(one("determined: not probed")).toBe("устойчивость позы не проверяли");
  });

  it("keeps the exponents of the determinedness probe, whose whole range is below a millionth", () => {
    expect(one("determined 1e-3 deg > 1e-6 deg")).toBe(
      "поза неустойчива: мельчайшее изменение чисел поворачивает её на 1e-3° при допустимых 1e-6°",
    );
  });

  it("counts the redraws a candidate survived, with the noun in the right form", () => {
    expect(one("redraws accepted 1 < 2")).toBe("устоял на 1 из 2 повторных выборок");
    expect(one("redraws accepted 0 < 1")).toBe("устоял на 0 из 1 повторной выборки");
  });

  it("turns the one line about the arms into the three sentences A §8.3 asks for", () => {
    expect(one("no arm: support 0 < 1 and margin 1.00 < 2")).toBe(
      "ни один другой стык не ставит фрагмент туда же, а вторая поза этой пары почти так же хороша (отрыв 1.0× при нужных 2×)",
    );
    expect(one("no arm: support 0 < 1 and re-search agreed 1/2 < 2")).toBe(
      "ни один другой стык не ставит фрагмент туда же, а повторный поиск пришёл в ту же позу 1 раз из 2",
    );
    expect(one("no arm: support 0 < 1 and the second placement is 3.69 t away, under 5")).toBe(
      "ни один другой стык не ставит фрагмент туда же, а вторая поза стоит слишком близко (3.7 t при нужных 5 t) — это может быть тот же шов со сдвигом",
    );
    expect(one("no arm: support 0 < 1 and no second placement to beat")).toContain(
      "ни один другой стык не ставит фрагмент туда же",
    );
    expect(
      one(
        "no arm: support 0 < 1 and the second placement is one R §6.5 would accept, so beating it is not evidence",
      ),
    ).toContain("ни один другой стык не ставит фрагмент туда же");
  });

  it("shows a string it does not know as it is, rather than dropping the engine's own words", () => {
    expect(one("brk 0.4000 < 0.5")).toBe("brk 0.4000 < 0.5");
    expect(one("no arm: support 0 < 1 and something nobody has written yet")).toBe(
      "no arm: support 0 < 1 and something nobody has written yet",
    );
    expect(one("tight not-a-number < 0.35")).toBe("tight not-a-number < 0.35");
  });

  it("keeps the engine's order and says nothing for a candidate that failed nothing", () => {
    expect(explain(["seam 4.0000 < 5", "tight 0.2767 < 0.35"], t)).toEqual([
      "шов 4.0 t короче нужных 5 t",
      "плотное прилегание 0.28 при нужных 0.35",
    ]);
    expect(explain([], t)).toEqual([]);
  });
});

describe("explain (A §8.4: why an accepted join did not fit)", () => {
  it("says what the placement ran into, not what the pair failed", () => {
    expect(one("penetrates FY234003 (0.004)")).toBe("в этой позе фрагмент вошёл бы внутрь FY234003");
    expect(one("inconsistent with the assembled poses (5.6 deg, 0.44 t)")).toBe(
      "расходится с уже расставленными позами группы: 5.6° и 0.44 t",
    );
  });

  it("keeps a join's two names as the engine wrote them — a name may hold a dash of its own", () => {
    expect(one("inconsistent with stronger join FY234019-FZ234010-02 (3.1 deg, 0.20 t)")).toBe(
      "расходится с более сильным стыком FY234019-FZ234010-02: 3.1° и 0.20 t",
    );
    expect(one("merging the two groups disagrees with join FY234001-FY234006 (9.0 deg, 1.10 t)")).toBe(
      "объединение двух групп расходится со стыком FY234001-FY234006: 9.0° и 1.10 t",
    );
  });

  it("tells the two group merges apart: the one the tool will not do, and the one the evidence does not allow", () => {
    expect(one("would merge two groups (not supported)")).toBe("соединил бы две группы, а этого сборка пока не делает");
    expect(one("would merge two groups (only a confirmed join may)")).toBe(
      "соединил бы две группы — на это имеет право только подтверждённый ядром стык",
    );
  });

  it("names the two vetoes of constraints.json, and leaves a third build's word alone", () => {
    expect(one("refused by constraints.json (`must_not_join`)")).toBe(
      "пара помечена в constraints.json как несоединяемая",
    );
    expect(one("refused by constraints.json (`different_object`)")).toBe(
      "фрагменты отнесены в constraints.json к разным объектам",
    );
    expect(one("refused by constraints.json (`something_new`)")).toBe("refused by constraints.json (`something_new`)");
  });
});

describe("headline (A §8.3: «Почему не подтверждён сам»)", () => {
  it("prefers the line about the arms, which is what «сам» means", () => {
    const held_ = held(["seam 4.0000 < 5", "no arm: support 0 < 1 and margin 1.00 < 2"]);
    expect(headline(row({ evidence: held_ }), t)).toBe(
      "ни один другой стык не ставит фрагмент туда же, а вторая поза этой пары почти так же хороша (отрыв 1.0× при нужных 2×)",
    );
  });

  it("falls back to the first refusal when no arm was even tried", () => {
    expect(headline(row({ evidence: held(["seam 4.0000 < 5"]) }), t)).toBe("шов 4.0 t короче нужных 5 t");
  });

  it("says what confirmed a confirmed join", () => {
    const confirmed = row({
      tier: "confirmed",
      used: true,
      evidence: { placements: 3, resample_tight_min: 0.39, resample_gap_max: 0.014, resample_accept: 3, support: 2, arm: "support" },
    });
    expect(headline(confirmed, t)).toBe("стык подтверждён: независимая опора");
  });

  it("says so for a pair the tier pass never probed, and for one the engine threw out", () => {
    expect(headline(row({ evidence: null }), t)).toBe("ядро не разбирало эту пару подробно");
    expect(headline(row({ tier: "rejected", evidence: null }), t)).toBe("ядро отклонило эту пару");
  });
});
