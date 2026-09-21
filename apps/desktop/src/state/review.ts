import { create } from "zustand";

import { api } from "../ipc";
import type { CommandError, EngineEventPayload, EngineFinishedPayload } from "../ipc/api";
import { toCommandError } from "../ipc/api";
import type { AssemblyDto } from "../ipc/bindings/AssemblyDto";
import type { CandidateRow } from "../ipc/bindings/CandidateRow";
import type { Decision } from "../ipc/bindings/Decision";
import type { Event as EngineEvent } from "../ipc/bindings/Event";
import type { PairDetailDto } from "../ipc/bindings/PairDetailDto";
import { useAssembly } from "./assembly";

/**
 * The reviewer's own state (A §8): which run is being reviewed, what has been decided about its
 * pairs, and what the session has answered.
 *
 * The decisions are the window's and the file is their shadow: undo and redo are an in-memory
 * history here (A §8.1), and every change sends the **whole** list — there is no partial update
 * and no per-decision command, so the file on disk and the list on the screen cannot drift. The
 * assembly that comes back is not kept here: it belongs to [`useAssembly`], which is what the
 * viewer and A §5's `draft` row already read.
 *
 * The pure half — [`applyDecision`], [`bulkAccept`] and the undo stack — is exported and tested
 * on its own (A §11), because that is where the rules of A §8.1 live and none of them needs a
 * store to be true.
 */

/** `decisions.json`'s `version`, as `sherd_app_core::decisions::DECISIONS_VERSION` writes it. */
const DECISIONS_VERSION = 1;

/** The pair the centre pane is showing, and the pose it is showing it in. */
export interface Selection {
  a: string;
  b: string;
  /** `p_a = T · p_b`, row-major — the matrix the candidate carries (R §0). */
  pose: CandidateRow["pose"];
}

/** The three fields undo and redo move between; the store holds them flat. */
export interface Undo {
  /** The decisions as they stand — what `reviewApply` sends and `decisions.json` holds. */
  decisions: Decision[];
  /** Every earlier list, oldest first. */
  past: Decision[][];
  /** What [`undone`] took away, newest first. */
  future: Decision[][];
}

/** Whether a decision and a pair of names are about the same pair (A §8.1: the key is unordered). */
function isPair(decision: Decision, a: string, b: string): boolean {
  return (decision.a === a && decision.b === b) || (decision.a === b && decision.b === a);
}

/** What was decided about a pair, however it is named, or `undefined` for an undecided one. */
export function decisionOf(list: readonly Decision[], a: string, b: string): Decision | undefined {
  return list.find((decision) => isPair(decision, a, b));
}

/**
 * «Подтвердить» on one candidate: the pose travels with the decision, so an accepted join
 * outlives the `match.state` it was found in (A §8.1).
 */
export function accepted(row: CandidateRow, at: string, bulk = false): Decision {
  return { a: row.a, b: row.b, verdict: "accept", pose: row.pose, source: row.tier, bulk, at, carried_from: null };
}

/**
 * «Отклонить»: on the pair as a whole and with no pose, because that is what `must_not_join`
 * means — this pair does not join, in any pose (A §8.2).
 */
export function rejected(row: CandidateRow, at: string): Decision {
  return { a: row.a, b: row.b, verdict: "reject", pose: null, source: row.tier, bulk: false, at, carried_from: null };
}

/**
 * The list with `decision` in it, replacing whatever was decided about that pair before —
 * including a decision that named the two fragments the other way round (A §8.1).
 *
 * The replacement takes the earlier one's place rather than being appended, so the list keeps the
 * order the pairs were first decided in and a reviewer changing their mind does not shuffle the
 * file. The new decision's own names win: it carries *its* candidate's pose, and a pose read
 * against the other order would be its inverse.
 */
export function applyDecision(list: readonly Decision[], decision: Decision): Decision[] {
  const at = list.findIndex((other) => isPair(other, decision.a, decision.b));
  if (at === -1) {
    return [...list, decision];
  }
  return list.map((other, i) => (i === at ? decision : other));
}

/** The list without whatever was decided about a pair; unchanged when nothing was (A §8.1). */
export function clearDecision(list: readonly Decision[], a: string, b: string): Decision[] {
  return list.filter((decision) => !isPair(decision, a, b));
}

/**
 * «Принять все оставшиеся вероятные» (A §8.4), expanded: one `accept` per undecided **probable**
 * pair, each flagged `bulk` so that one undo takes the whole expansion back.
 *
 * A pair that is already decided is skipped — the button is about the rest, and re-accepting a
 * pair the reviewer has just rejected would undo their work silently. Rows are taken in the
 * order they are given and only the first of each pair is used, so the caller's sort (the
 * queue's, best score first) decides which pose an accepted pair gets.
 */
export function bulkAccept(list: readonly Decision[], rows: readonly CandidateRow[], at: string): Decision[] {
  let next = [...list];
  for (const row of rows) {
    if (row.tier !== "probable" || decisionOf(next, row.a, row.b) !== undefined) {
      continue;
    }
    next = applyDecision(next, accepted(row, at, true));
  }
  return next;
}

/**
 * One step of the history: `decisions` becomes the list, what stood there goes on the stack, and
 * the redo is thrown away — a new decision after an undo makes the undone branch unreachable,
 * which is what every editor does.
 */
export function recorded(history: Undo, decisions: Decision[]): Undo {
  return { decisions, past: [...history.past, history.decisions], future: [] };
}

/** One step back; nothing at all when there is nothing to go back to. */
export function undone(history: Undo): Undo {
  const previous = history.past.at(-1);
  if (previous === undefined) {
    return history;
  }
  return {
    decisions: previous,
    past: history.past.slice(0, -1),
    future: [history.decisions, ...history.future],
  };
}

/** One step forward, over what [`undone`] took away. */
export function redone(history: Undo): Undo {
  const [next, ...rest] = history.future;
  if (next === undefined) {
    return history;
  }
  return { decisions: next, past: [...history.past, history.decisions], future: rest };
}

/** The fields of an `assembly` event, named one by one so a new one cannot be dropped silently. */
function assemblyOf(event: Extract<EngineEvent, { event: "assembly" }>): AssemblyDto {
  return { groups: event.groups, poses: event.poses, joins: event.joins, unplaced: event.unplaced };
}

/** The same for a `pair_detail`, which is A §2.2's `PairDetail` flattened onto the event. */
function detailOf(event: Extract<EngineEvent, { event: "pair_detail" }>): PairDetailDto {
  return {
    a: event.a,
    b: event.b,
    contact: event.contact,
    contact_class: event.contact_class,
    seam: event.seam,
    tight: event.tight,
    gap: event.gap,
  };
}

/** What the «Ревью» mode is looking at and what it has decided (A §8). */
export interface ReviewState extends Undo {
  /** The run the session is over, or `null` when none is open. */
  runId: string | null;
  /** Whether the session has said `ready` — until then it can be asked nothing (A §8). */
  ready: boolean;
  /** Whether an answer to the last request is still on its way (A §8.2: «under a second»). */
  pending: boolean;
  /**
   * And whether that request is the slow one: R §9 over the unrefined groups (A §8.4, «≈30 с»).
   * Apart from [`ReviewState.pending`] because the two are reported differently — a reassembly
   * answers before anything could be drawn about it, a refinement is a job to watch in the
   * status line — and neither the draft line nor the status line can tell them apart from the
   * jobs store: the `stage` a session leaves there stands until the session ends.
   */
  refining: boolean;
  /** Decisions the session could not apply because their fragments are gone (A §8.2). */
  dropped: Decision[];
  /** The seam of the selected placement, or `null` while it is being asked for. */
  detail: PairDetailDto | null;
  /** The pair the centre pane is showing. */
  selected: Selection | null;
  /** A refusal worth showing: a command that failed, or the session's own `request_failed`. */
  error: CommandError | null;

  /**
   * Opens a session over a run and loads what was decided about it before (A §8). Entering the
   * mode again over the same run does nothing, which is what the shell answers anyway.
   */
  open(runId: string): Promise<void>;
  /** Closes it and gives the engine's ~3 GB back; closing none is no error. */
  close(): Promise<void>;
  /** Shows a candidate and asks the session for its seam (A §8.3). */
  select(row: CandidateRow | null): void;

  accept(row: CandidateRow): void;
  reject(row: CandidateRow): void;
  /** Un-decides a pair — back to «нет решения», not to «отклонено». */
  clear(a: string, b: string): void;
  acceptAllProbable(rows: readonly CandidateRow[]): void;
  undo(): void;
  redo(): void;
  /** «Сбросить решения»: the draft back to the run as the engine left it, in one undoable step. */
  reset(): void;
  /** «Уточнить позы»: R §9 over the groups the decisions left unrefined (A §8.4). */
  refine(): void;
  /** Puts the «перенесено не всё» notice away (A §8.5). */
  dismissDropped(): void;
  dismissError(): void;

  applyEvent(payload: EngineEventPayload): void;
  applyFinished(payload: EngineFinishedPayload): void;
}

/** What a session starts from, and what closing one leaves behind. */
function blank(): Omit<ReviewState, keyof ReviewActions> {
  return {
    runId: null,
    ready: false,
    decisions: [],
    past: [],
    future: [],
    pending: false,
    refining: false,
    dropped: [],
    detail: null,
    selected: null,
    error: null,
  };
}

/** Everything of [`ReviewState`] that is a function, so [`blank`] can name the rest. */
type ReviewActions = Pick<
  ReviewState,
  | "open"
  | "close"
  | "select"
  | "accept"
  | "reject"
  | "clear"
  | "acceptAllProbable"
  | "undo"
  | "redo"
  | "reset"
  | "refine"
  | "dismissDropped"
  | "dismissError"
  | "applyEvent"
  | "applyFinished"
>;

export const useReview = create<ReviewState>()((set, get) => {
  /** RFC 3339, as `decisions.json` stamps every decision (A §8.1). */
  const now = (): string => new Date().toISOString();

  /** Files the list and asks for a reassembly; the answer is an `assembly` event (A §8.2). */
  const send = async (decisions: Decision[]): Promise<void> => {
    try {
      await api.reviewApply({ version: DECISIONS_VERSION, decisions });
    } catch (e) {
      set({ pending: false, error: toCommandError(e) });
    }
  };

  /** One change of the draft: the history moves, the screen waits, the list goes out whole. */
  const commit = (decisions: Decision[]): void => {
    set((state) => ({ ...recorded(state, decisions), pending: true, error: null }));
    void send(decisions);
  };

  /** Asks the session for a placement's seam (A §8.3); a session that is not ready is asked on `ready`. */
  const askPair = (selection: Selection): void => {
    void (async () => {
      try {
        await api.reviewPair(selection.a, selection.b, selection.pose);
      } catch (e) {
        set({ error: toCommandError(e) });
      }
    })();
  };

  return {
    ...blank(),

    open: async (runId) => {
      if (get().runId === runId) {
        return;
      }
      set({ ...blank(), runId });
      try {
        // The run's own file first: a run continued from a review starts with those decisions
        // already in it (A §8.5), and the session's baseline is the assembly they produced.
        const filed = await api.runDecisions(runId);
        if (get().runId !== runId) {
          return;
        }
        set({ decisions: filed.decisions });
        await api.reviewOpen(runId);
      } catch (e) {
        if (get().runId === runId) {
          set({ error: toCommandError(e) });
        }
      }
    },

    close: async () => {
      if (get().runId === null) {
        return;
      }
      // The screen is left at once: the command returns as soon as the line is written and the
      // slot is freed a moment later by the session's own thread (A §8), so waiting for it would
      // only hold the mode open over a session nobody is looking at.
      set(blank());
      try {
        await api.reviewClose();
      } catch (e) {
        // A refusal here means the worker is still holding the collection's three gigabytes,
        // which is worth saying out loud even though the reviewer has moved on (A §10).
        set({ error: toCommandError(e) });
      }
    },

    select: (row) => {
      if (row === null) {
        set({ selected: null, detail: null });
        return;
      }
      const selection: Selection = { a: row.a, b: row.b, pose: row.pose };
      set({ selected: selection, detail: null });
      if (get().ready) {
        askPair(selection);
      }
    },

    accept: (row) => {
      commit(applyDecision(get().decisions, accepted(row, now())));
    },

    reject: (row) => {
      commit(applyDecision(get().decisions, rejected(row, now())));
    },

    clear: (a, b) => {
      const next = clearDecision(get().decisions, a, b);
      if (next.length !== get().decisions.length) {
        commit(next);
      }
    },

    acceptAllProbable: (rows) => {
      const next = bulkAccept(get().decisions, rows, now());
      if (next.length !== get().decisions.length) {
        commit(next);
      }
    },

    undo: () => {
      const state = get();
      const back = undone(state);
      // The very same object back means there was nothing to undo: no request, and nothing on
      // the screen moves.
      if (back !== state) {
        set({ ...back, pending: true, error: null });
        void send(back.decisions);
      }
    },

    redo: () => {
      const state = get();
      const forward = redone(state);
      if (forward !== state) {
        set({ ...forward, pending: true, error: null });
        void send(forward.decisions);
      }
    },

    reset: () => {
      if (get().decisions.length > 0) {
        commit([]);
      }
    },

    refine: () => {
      set({ pending: true, refining: true, error: null });
      void (async () => {
        try {
          await api.reviewRefine();
        } catch (e) {
          set({ pending: false, refining: false, error: toCommandError(e) });
        }
      })();
    },

    dismissDropped: () => {
      set({ dropped: [] });
    },

    dismissError: () => {
      set({ error: null });
    },

    applyEvent: (payload) => {
      // A run's own `dropped` (A §8.5's carry-over) arrives on the run's job, not the session's,
      // and the notice is the same one — so that event is taken whoever sent it.
      const event = payload.event;
      if (event.event === "dropped") {
        set((state) => ({ dropped: [...state.dropped, ...event.decisions] }));
        return;
      }
      if (payload.job !== "review") {
        return;
      }
      switch (event.event) {
        case "ready": {
          set({ ready: true });
          // The pair the reviewer picked while the match was still loading: asked for now that
          // there is somebody to ask.
          const selection = get().selected;
          if (selection !== null) {
            askPair(selection);
          }
          break;
        }
        case "assembly":
          // A §2.1: the shell has already filed this over `assembly.json`; the window only has
          // to draw it.
          useAssembly.getState().show(assemblyOf(event));
          // The answer to a reassembly *and* to a refinement (A §8.4): whichever was asked for
          // is over.
          set({ pending: false, refining: false });
          break;
        case "pair_detail": {
          const detail = detailOf(event);
          const selection = get().selected;
          // An answer for a pair the reviewer has moved on from is stale, not wrong: dropping it
          // keeps the seam on the screen the seam of the pair on the screen.
          if (selection !== null && selection.a === detail.a && selection.b === detail.b) {
            set({ detail });
          }
          break;
        }
        case "request_failed":
          set({ pending: false, refining: false, error: { kind: "worker", message: event.message } });
          break;
        default:
          // `stage` and `progress` while the match loads and while R §9 runs; the status line
          // reads those from the jobs store, as it does for every other job.
          break;
      }
    },

    applyFinished: (payload) => {
      if (payload.job !== "review" || get().runId === null) {
        return;
      }
      const outcome = payload.outcome;
      // A §10: a session whose `match.state` or collection would not load ends here rather than
      // at the first decision, and the mode shows what it said and leaves. A cancel is the user's
      // own doing and needs no banner.
      const failure = "Failed" in outcome && outcome.Failed.kind !== "cancelled" ? outcome.Failed : null;
      set({
        ...blank(),
        error: failure === null ? null : { kind: failure.kind, message: failure.message },
      });
    },
  };
});
