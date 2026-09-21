import { useVirtualizer } from "@tanstack/react-virtual";
import clsx from "clsx";
import { LayoutGrid, List } from "lucide-react";
import type { KeyboardEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { api } from "../../ipc";
import type { FragmentInfo } from "../../ipc/bindings/FragmentInfo";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import type { InputFilter } from "../../state/ui";
import { useUi } from "../../state/ui";
import { FOCUS_RING } from "../../ui/Button";
import Chip from "../../ui/Chip";
import IconButton from "../../ui/IconButton";
import { formatBytes, middleEllipsis } from "./format";

/** The side of one thumbnail cell, gap included — the plan's 104 px. */
const CELL = 104;
/** One row of the «☰» layout: a small thumbnail, the name and the size on one line. */
const LIST_ROW = 30;
/** Rows drawn above and below the window, so a fast scroll shows cells and not holes. */
const OVERSCAN = 3;
/** How many characters of a name survive in a cell and in a list row before it is cut. */
const GRID_NAME_MAX = 14;
const LIST_NAME_MAX = 34;
/** The path in the header: enough of both ends to recognise the folder in a 360 px pane. */
const PATH_MAX = 44;

/**
 * One scan as the pane shows it: always a file of the input folder, and the prepared row for it
 * once a `Prepare` has produced one.
 *
 * The file comes first on purpose (A §5: «Список файлов виден сразу»). The window must not wait
 * for the engine to say a collection exists — 155 names appear the moment the folder is linked,
 * and each grows a thumbnail when its `fragment_ready` lands.
 */
interface Row {
  name: string;
  size: number;
  info: FragmentInfo | undefined;
  excluded: boolean;
  warnings: number;
}

/** Whether a row belongs in the list under the chosen filter. */
function passes(row: Row, filter: InputFilter): boolean {
  switch (filter) {
    case "all":
      return true;
    case "warnings":
      return row.warnings > 0;
    case "excluded":
      return row.excluded;
  }
}

/** The thumbnail a prepared fragment has, or `null` while it has none yet. */
function thumbnailUrl(view: WorkspaceView, row: Row): string | null {
  return row.info === undefined ? null : api.assetUrl(`${view.fragments_dir}/${row.name}.png`);
}

/**
 * The thumbnail box: `fragments/<name>.png`, or a pulsing placeholder while the fragment has no
 * prepared row. The placeholder pulses because it is a promise the app is keeping — and it stops
 * pulsing for anyone who asked their system not to animate things.
 */
function Thumbnail({ url, title, className }: { url: string | null; title: string; className: string }) {
  return (
    <span className={clsx("block overflow-hidden rounded bg-viewport", className)}>
      {url === null ? (
        <span className="block h-full w-full animate-pulse bg-panel-2 motion-reduce:animate-none" title={title} />
      ) : (
        <img src={url} alt="" loading="lazy" draggable={false} className="h-full w-full object-contain" />
      )}
    </span>
  );
}

/**
 * One cell of the grid, or one row of the list: a real `<button>`, because choosing a fragment is
 * an action and the pane is worked with the keyboard as much as with the mouse (A §7.1).
 */
function Cell({
  row,
  url,
  index,
  selected,
  tabbable,
  grid,
  size,
  onPick,
}: {
  row: Row;
  url: string | null;
  index: number;
  selected: boolean;
  tabbable: boolean;
  grid: boolean;
  size: string;
  onPick: (name: string) => void;
}) {
  const { t } = useTranslation();
  const name = middleEllipsis(row.name, grid ? GRID_NAME_MAX : LIST_NAME_MAX);

  return (
    <button
      type="button"
      data-cell={index}
      aria-pressed={selected}
      tabIndex={tabbable ? 0 : -1}
      title={row.name}
      onClick={() => {
        onPick(row.name);
      }}
      className={clsx(
        "flex h-full min-w-0 items-center rounded-md text-left hover:bg-panel-2",
        grid ? "flex-col gap-1 p-1" : "gap-2 px-1.5",
        // Drawn inside the cell: an outline on the edge of the scroll container would be clipped.
        selected && "outline-2 outline-offset-[-2px] outline-accent",
        row.excluded && "opacity-40",
        FOCUS_RING,
      )}
    >
      <Thumbnail url={url} title={t("input.pending")} className={grid ? "w-full flex-1" : "h-5 w-7 shrink-0"} />
      <span className={clsx("flex min-w-0 items-baseline gap-1", grid ? "w-full justify-center" : "flex-1")}>
        <span className={clsx("truncate text-[11px]", row.excluded && "line-through")}>{name}</span>
        {row.warnings > 0 ? (
          <span className="shrink-0 text-warn" role="img" aria-label={t("counts.warnings", { n: row.warnings })}>
            ⚠
          </span>
        ) : null}
      </span>
      {grid ? null : <span className="shrink-0 text-[10px] text-muted tabular-nums">{size}</span>}
    </button>
  );
}

/**
 * The left pane of the «Вход» mode (A §7.3, part 3 of `states-and-input.html`): where the scans
 * are and how many, the three filters, the grid/list toggle, and the collection itself.
 *
 * The list is virtualised: 155 fragments is nothing, but a museum box of two thousand would put
 * two thousand `<img>`s in the document, and the window has to stay responsive while the engine
 * is using the same machine.
 */
export default function InputLeft({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const filter = useUi((state) => state.inputFilter);
  const layout = useUi((state) => state.inputLayout);
  const selectedName = useUi((state) => state.selectedFragment);

  const scroller = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  // Set when the arrow keys moved the selection, so the cell that is now selected takes the focus
  // as soon as the virtualiser has drawn it. A mouse click needs none of this: it focuses itself.
  const wantsFocus = useRef(false);

  const rows = useMemo<Row[]>(() => {
    const excluded = new Set(view.excluded);
    const prepared = new Map(view.fragments.map((info) => [info.name, info]));
    return view.files.map((file) => {
      const info = prepared.get(file.name);
      return {
        name: file.name,
        size: file.size,
        info,
        excluded: excluded.has(file.name),
        warnings: info?.warnings.length ?? 0,
      };
    });
  }, [view.files, view.fragments, view.excluded]);

  const shown = useMemo(() => rows.filter((row) => passes(row, filter)), [rows, filter]);
  const totalBytes = useMemo(() => rows.reduce((sum, row) => sum + row.size, 0), [rows]);
  const withWarnings = rows.filter((row) => row.warnings > 0).length;
  const excludedCount = rows.filter((row) => row.excluded).length;

  const grid = layout === "grid";
  const columns = grid ? Math.max(1, Math.floor(width / CELL)) : 1;
  const rowHeight = grid ? CELL : LIST_ROW;
  const lines = Math.ceil(shown.length / columns);

  const virtualizer = useVirtualizer<HTMLDivElement, HTMLDivElement>({
    count: lines,
    getScrollElement: () => scroller.current,
    estimateSize: () => rowHeight,
    overscan: OVERSCAN,
  });

  // The pane's width decides how many cells fit, and the layout decides how tall a line is; both
  // change under the user's hand, and neither is something the virtualiser can notice by itself.
  useEffect(() => {
    virtualizer.measure();
  }, [virtualizer, rowHeight, columns]);

  useEffect(() => {
    const element = scroller.current;
    if (element === null) {
      return undefined;
    }
    setWidth(element.clientWidth);
    const observer = new ResizeObserver((entries) => {
      const entry = entries.at(0);
      if (entry !== undefined) {
        setWidth(entry.contentRect.width);
      }
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, []);

  const selectedIndex = shown.findIndex((row) => row.name === selectedName);

  // Focus follows the arrow keys, but the cell to focus may not be drawn yet: the effect runs
  // again after every render the scroll causes, and gives up only once it has found the cell.
  useEffect(() => {
    if (!wantsFocus.current || selectedIndex < 0) {
      return;
    }
    const cell = scroller.current?.querySelector(`[data-cell="${String(selectedIndex)}"]`);
    if (cell instanceof HTMLElement) {
      wantsFocus.current = false;
      cell.focus();
    }
  });

  const pick = useCallback((name: string) => {
    useUi.getState().selectFragment(name);
  }, []);

  const move = (to: number) => {
    const row = shown[to];
    if (row === undefined) {
      return;
    }
    wantsFocus.current = true;
    useUi.getState().selectFragment(row.name);
    virtualizer.scrollToIndex(Math.floor(to / columns));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || shown.length === 0) {
      return;
    }
    let step: number;
    switch (e.key) {
      case "ArrowRight":
        step = 1;
        break;
      case "ArrowLeft":
        step = -1;
        break;
      case "ArrowDown":
        step = columns;
        break;
      case "ArrowUp":
        step = -columns;
        break;
      default:
        return;
    }
    e.preventDefault();
    // With nothing chosen yet, the first arrow key lands on the first fragment rather than on
    // whatever a step away from «no selection» would be.
    if (selectedIndex < 0) {
      move(0);
      return;
    }
    move(Math.min(Math.max(selectedIndex + step, 0), shown.length - 1));
  };

  const filters: { id: InputFilter; label: string }[] = [
    { id: "all", label: t("input.filter_all") },
    { id: "warnings", label: t("input.filter_warnings", { n: withWarnings }) },
    { id: "excluded", label: t("input.filter_excluded", { n: excludedCount }) },
  ];

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border px-2 py-1.5">
        <div className="flex items-center gap-1">
          <p className="min-w-0 flex-1 truncate text-[11px] text-muted" title={view.input.path ?? undefined}>
            {view.input.linked
              ? middleEllipsis(view.input.path ?? t("input.path_unavailable"), PATH_MAX)
              : t("status.empty")}
          </p>
          <IconButton
            label={t("input.layout_grid")}
            pressed={grid}
            onClick={() => {
              useUi.getState().setInputLayout("grid");
            }}
          >
            <LayoutGrid size={13} aria-hidden="true" />
          </IconButton>
          <IconButton
            label={t("input.layout_list")}
            pressed={!grid}
            onClick={() => {
              useUi.getState().setInputLayout("list");
            }}
          >
            <List size={13} aria-hidden="true" />
          </IconButton>
        </div>
        {/* Only for a folder that answers: with none linked, or one that has moved, «0 файлов ·
            0 B» would be an answer about a folder nobody counted. */}
        {view.input.available ? (
          <p className="mt-0.5 text-[11px] text-muted">
            {`${t("input.files", { count: rows.length })} · ${formatBytes(totalBytes, language)}`}
          </p>
        ) : null}

        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          {filters.map((choice) => (
            <Chip
              key={choice.id}
              active={filter === choice.id}
              onClick={() => {
                useUi.getState().setInputFilter(choice.id);
              }}
            >
              {choice.label}
            </Chip>
          ))}
        </div>
      </div>

      {/* The scroll container is always in the document, even with nothing to put in it: it is
          what the width is measured on, and a pane that measured itself at zero would lay the
          whole collection out in one column the moment the fragments arrived. */}
      <div
        ref={scroller}
        // The keys are taken here and not on each cell: the selection moves between cells that
        // may not be drawn, so one handler over the whole list is the only place it can live.
        onKeyDown={onKeyDown}
        // And the container itself is a tab stop, because the cell that would otherwise be the
        // only one is virtualised away as soon as it scrolls out of the window: a keyboard user
        // who scrolled the list would then Tab straight past the collection with no way back
        // into it. Focused here, the arrow keys work exactly as they do from a cell.
        tabIndex={0}
        role="group"
        aria-label={t("input.list")}
        // Its own ring and not `FOCUS_RING`: drawn inside, as the selected cell's outline is,
        // because a ring on the edge of a scroll container is clipped by the container.
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
      >
        {shown.length === 0 ? (
          rows.length === 0 ? null : (
            <p className="px-2 py-3 text-[11px] text-muted">{t("input.nothing_matches")}</p>
          )
        ) : (
          <div className="relative w-full" style={{ height: `${String(virtualizer.getTotalSize())}px` }}>
            {virtualizer.getVirtualItems().map((line) => (
              <div
                key={line.key}
                className={clsx("absolute top-0 left-0 grid w-full gap-1", grid ? "px-1 pt-1" : "px-1")}
                style={{
                  height: `${String(line.size)}px`,
                  transform: `translateY(${String(line.start)}px)`,
                  gridTemplateColumns: `repeat(${String(columns)}, minmax(0, 1fr))`,
                }}
              >
                {shown.slice(line.index * columns, line.index * columns + columns).map((row, column) => {
                  const index = line.index * columns + column;
                  return (
                    <Cell
                      key={row.name}
                      row={row}
                      url={thumbnailUrl(view, row)}
                      index={index}
                      selected={index === selectedIndex}
                      tabbable={selectedIndex < 0 ? index === 0 : index === selectedIndex}
                      grid={grid}
                      size={formatBytes(row.size, language)}
                      onPick={pick}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
