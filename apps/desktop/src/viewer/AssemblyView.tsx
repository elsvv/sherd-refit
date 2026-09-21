import { type Ref, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import type { AssemblyDto } from "../ipc/bindings/AssemblyDto";
import Meter from "../ui/Meter";
import type { ColourMode, LayoutMode } from "./AssemblyViewer";
import { AssemblyViewer } from "./AssemblyViewer";

/** The two things a pane around the viewer cannot ask for with a prop, because both answer back. */
export interface AssemblyViewHandle {
  /** The current frame as a PNG data URL, for «Снимок PNG»; `null` before the viewer exists. */
  screenshot(): string | null;
  /** Puts one fragment in the middle of the viewport — the inspector's partner rows fly to it. */
  flyTo(name: string): void;
}

/** Everything the «Сборка» mode decides about what the viewport shows. */
export interface AssemblyViewProps {
  /** Every fragment of the run that has a display mesh; the order is what «Цвет: фрагменты» uses. */
  fragments: readonly { name: string; url: string }[];
  assembly: AssemblyDto;
  colourMode: ColourMode;
  layout: LayoutMode;
  /** Which group `single` shows alone. */
  group?: number | undefined;
  /** The groups whose eye is off in the tree. */
  hidden?: ReadonlySet<number> | undefined;
  /** Whether the tray of groups of one is on screen; off by default (A §7.2). */
  unassembled: boolean;
  /** «Разъединить», 0…1. */
  explode: number;
  /** «Подписи» (`L`). */
  labels: boolean;
  selected: string | null;
  /** A property and not a method, so that the pane may hand over a closure of its own freely. */
  onSelect: (name: string | null) => void;
  /** `useFitSignal`'s counter: every increment is one press of `F` or of «Вписать» (A §7.1). */
  fitSignal: number;
  ref?: Ref<AssemblyViewHandle> | undefined;
}

/** Where the tooltip is, and what it is about. */
interface Tip {
  name: string;
  x: number;
  y: number;
  /** Near the right edge the tooltip opens to the left instead, or it would be cut off. */
  flip: boolean;
}

/** How much room the tooltip needs to its right before it flips over to the other side. */
const TIP_ROOM = 200;

/**
 * The React side of the assembly viewer, and the whole of it (A §7.2: `viewer/` is three.js with
 * no React in it). This component owns one [`AssemblyViewer`] for its lifetime and does nothing
 * but tell it what changed — React never sees the scene, and the scene never re-renders React,
 * except for the two things that are words rather than pixels: how much of the collection has
 * arrived, and what is under the cursor.
 *
 * The canvas itself is the viewer's, not React's: the effect hands over an empty `<div>` that
 * React never puts a child into. A canvas React kept across StrictMode's mount → cleanup → mount
 * would come back to the second viewer with the first one's WebGL context on it, lost, and
 * three.js throws on a lost context.
 */
export default function AssemblyView(props: AssemblyViewProps) {
  const { fragments, assembly, colourMode, layout, group, hidden, unassembled } = props;
  const { explode, labels, selected, onSelect, fitSignal, ref } = props;
  const { t } = useTranslation();
  const host = useRef<HTMLDivElement | null>(null);
  const [viewer, setViewer] = useState<AssemblyViewer | null>(null);
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [tip, setTip] = useState<Tip | null>(null);

  // The viewer is made once and keeps its callbacks for its whole life, so what they call has to
  // be read at the moment of the call and not closed over: `onSelect` is a new function on every
  // render of the mode around this one.
  const latest = useRef(onSelect);
  latest.current = onSelect;

  // `fragments` is built by the pane above and is a new array on every render; its *content* is
  // what decides whether 155 meshes have to be fetched again.
  const sources = useMemo(() => fragments.map((row) => `${row.name}\u0000${row.url}`).join("\n"), [fragments]);
  const list = useRef(fragments);
  list.current = fragments;
  const current = useRef(assembly);
  current.current = assembly;

  // One viewer per mounted component. It is kept in state and not in a ref on purpose: the
  // effects below have to run again once it exists, and a ref changing tells React nothing.
  useEffect(() => {
    const element = host.current;
    if (element === null) {
      return undefined;
    }
    const created = new AssemblyViewer(element, {
      onHover: (name, at) => {
        if (name === null || at === null) {
          setTip(null);
          return;
        }
        const width = host.current?.clientWidth ?? 0;
        setTip({ name, x: at.x, y: at.y, flip: at.x > width - TIP_ROOM });
      },
      onSelect: (name) => {
        latest.current(name);
      },
      onProgress: (loaded, total) => {
        setProgress({ loaded, total });
      },
    });
    setViewer(created);
    return () => {
      setViewer(null);
      created.dispose();
    };
  }, []);

  useEffect(() => {
    if (viewer === null) {
      return;
    }
    // The assembly is read from the ref: a new run selected at the same time as a new collection
    // must be loaded once, with both, and not loaded and then replaced.
    void viewer.load({ fragments: list.current.map((row) => ({ ...row })), assembly: current.current });
    // `sources` is in the dependencies and not in the body on purpose: it stands for the content
    // of `fragments`, which is what decides whether the meshes have to be fetched again.
  }, [viewer, sources]);

  useEffect(() => {
    viewer?.setAssembly(assembly);
  }, [viewer, assembly]);

  useEffect(() => {
    viewer?.setColourMode(colourMode);
  }, [viewer, colourMode]);

  useEffect(() => {
    viewer?.setLayout(layout, group);
  }, [viewer, layout, group]);

  useEffect(() => {
    if (viewer === null) {
      return;
    }
    // Every group each time: the set of hidden ones is small, the call is a no-op for a group
    // that already stands as asked, and a diff of two sets would be more code than the loop.
    assembly.groups.forEach((_, index) => {
      viewer.setGroupVisible(index, hidden?.has(index) !== true);
    });
  }, [viewer, assembly, hidden]);

  useEffect(() => {
    viewer?.setUnassembledVisible(unassembled);
  }, [viewer, unassembled]);

  useEffect(() => {
    viewer?.setExplode(explode);
  }, [viewer, explode]);

  useEffect(() => {
    viewer?.setLabels(labels);
  }, [viewer, labels]);

  useEffect(() => {
    viewer?.select(selected);
  }, [viewer, selected]);

  // `fitSignal` is read for its change and not for its value; a new viewer starts fitted anyway.
  useEffect(() => {
    if (fitSignal > 0) {
      viewer?.fit();
    }
  }, [viewer, fitSignal]);

  useImperativeHandle(
    ref,
    () => ({
      screenshot: () => viewer?.screenshot() ?? null,
      flyTo: (name: string) => {
        viewer?.flyTo(name);
      },
    }),
    [viewer],
  );

  // What the tooltip says about the fragment under the cursor: which vessel it belongs to and
  // how many joins hold it there (`main-layout.html`, the `.tip` over the viewport).
  const where = useMemo(() => {
    const of = new Map<string, number>();
    assembly.groups.forEach((row, index) => {
      for (const name of row.members) {
        of.set(name, row.members.length < 2 ? -1 : index);
      }
    });
    const joins = new Map<string, number>();
    for (const join of assembly.joins) {
      joins.set(join.a, (joins.get(join.a) ?? 0) + 1);
      joins.set(join.b, (joins.get(join.b) ?? 0) + 1);
    }
    return { of, joins };
  }, [assembly]);

  const loading = progress.total > 0 && progress.loaded < progress.total;
  const tipGroup = tip === null ? undefined : where.of.get(tip.name);

  return (
    <div ref={host} className="relative h-full w-full overflow-hidden">
      {loading ? (
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 px-3 py-2">
          <Meter
            value={progress.loaded}
            max={progress.total}
            label={t("assembly.loading", { done: progress.loaded, total: progress.total })}
            className="w-full"
          />
          <div className="mt-1 text-[11px] text-viewport-text tabular-nums">
            {t("assembly.loading", { done: progress.loaded, total: progress.total })}
          </div>
        </div>
      ) : null}

      {tip === null ? null : (
        // `pointer-events-none`: the orbit and the picking keep working under a label that is
        // only there to read. It is placed at the pointer and flips over near the right edge.
        <div
          className="pointer-events-none absolute z-10 max-w-[220px] truncate rounded-md border border-border bg-panel px-2 py-1 text-xs text-text shadow-lg"
          style={{
            left: `${String(tip.x + (tip.flip ? -14 : 14))}px`,
            top: `${String(tip.y + 16)}px`,
            transform: tip.flip ? "translateX(-100%)" : undefined,
          }}
        >
          <b>{tip.name}</b>
          <div className="text-muted">
            {tipGroup === undefined || tipGroup < 0
              ? t("assembly.unpaired")
              : t("assembly.group", { n: tipGroup })}
            {" · "}
            {t("assembly.joins", { count: where.joins.get(tip.name) ?? 0 })}
          </div>
        </div>
      )}
    </div>
  );
}
