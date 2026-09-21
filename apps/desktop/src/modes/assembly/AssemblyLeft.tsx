import { useVirtualizer } from "@tanstack/react-virtual";
import clsx from "clsx";
import { ChevronDown, ChevronRight, Eye, EyeOff, Focus, Search } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";

import type { AssemblyDto } from "../../ipc/bindings/AssemblyDto";
import { useAssembly } from "../../state/assembly";
import { useUi } from "../../state/ui";
import { FOCUS_RING } from "../../ui/Button";
import { groupColour, UNPAIRED_COLOUR } from "../../viewer/colours";
import type { Row } from "./tree";
import { buildRows, UNPAIRED } from "./tree";

/** One line of the tree, in pixels — the height the virtualiser lays the list out with. */
const ROW = 24;
/** Rows drawn above and below the window, so a fast scroll shows names and not holes. */
const OVERSCAN = 8;

/** Which group of the run a fragment ended in, or `null` for a name the run never placed. */
function groupOf(assembly: AssemblyDto, name: string): number | null {
  const at = assembly.groups.findIndex((group) => group.members.includes(name));
  return at === -1 ? null : at;
}

/** The colour of a row's dot: its group's, or the grey of «без пары» (A §7.2). */
function dotColour(group: number): string {
  return group === UNPAIRED ? UNPAIRED_COLOUR : groupColour(group);
}

/** A group's or the tray's line: a dot, a chevron, what it is, how many, and its two switches. */
function SectionRow({
  label,
  count,
  colour,
  open,
  hidden,
  selected,
  onOpen,
  onSelect,
  onHide,
  onAlone,
}: {
  label: string;
  count: number;
  colour: string;
  open: boolean;
  hidden: boolean;
  selected: boolean;
  onOpen: () => void;
  onSelect: () => void;
  onHide: () => void;
  /** «показать одну»; the tray has no single-group layout of its own, so it passes `null`. */
  onAlone: (() => void) | null;
}) {
  const { t } = useTranslation();
  return (
    <div className={clsx("flex h-full items-center gap-0.5 rounded px-1", selected && "bg-select")}>
      <button
        type="button"
        aria-label={open ? t("assembly.collapse") : t("assembly.expand")}
        aria-expanded={open}
        title={open ? t("assembly.collapse") : t("assembly.expand")}
        onClick={onOpen}
        className={clsx("inline-flex h-4 w-4 shrink-0 items-center justify-center rounded text-muted", FOCUS_RING)}
      >
        {open ? <ChevronDown size={12} aria-hidden="true" /> : <ChevronRight size={12} aria-hidden="true" />}
      </button>
      <button
        type="button"
        aria-pressed={selected}
        onClick={onSelect}
        className={clsx("flex min-w-0 flex-1 items-center gap-1.5 rounded px-1 text-left hover:bg-panel-2", FOCUS_RING)}
      >
        <span
          aria-hidden="true"
          className={clsx("h-2 w-2 shrink-0 rounded-full", hidden && "opacity-30")}
          style={{ background: colour }}
        />
        <span className={clsx("min-w-0 flex-1 truncate text-[11px]", hidden && "text-muted")}>{label}</span>
        <span className="shrink-0 text-[10px] text-muted tabular-nums">{t("assembly.pieces", { count })}</span>
      </button>
      {onAlone === null ? null : (
        <button
          type="button"
          aria-label={t("assembly.show_one")}
          title={t("assembly.show_one")}
          onClick={onAlone}
          className={clsx(
            "inline-flex h-4 w-4 shrink-0 items-center justify-center rounded text-muted hover:text-text",
            FOCUS_RING,
          )}
        >
          <Focus size={12} aria-hidden="true" />
        </button>
      )}
      <button
        type="button"
        aria-label={hidden ? t("assembly.show") : t("assembly.hide")}
        aria-pressed={!hidden}
        title={hidden ? t("assembly.show") : t("assembly.hide")}
        onClick={onHide}
        className={clsx(
          "inline-flex h-4 w-4 shrink-0 items-center justify-center rounded text-muted hover:text-text",
          FOCUS_RING,
        )}
      >
        {hidden ? <EyeOff size={12} aria-hidden="true" /> : <Eye size={12} aria-hidden="true" />}
      </button>
    </div>
  );
}

/**
 * The left pane of the «Сборка» mode (A §7.3, `main-layout.html` variant A): a search box, then
 * the run's groups with their fragments under them and everything that found no partner in one
 * section at the end.
 *
 * What the tree *is* lives in [`buildRows`], which is pure and tested; this draws it, virtualised
 * because a museum box of two thousand sherds is two thousand rows and the window has to stay
 * responsive while the engine is using the same machine.
 *
 * Choosing a row here and choosing a fragment in the viewport are the same act, through
 * [`useUi`]: the row of the fragment the user clicked in 3D opens its group and scrolls itself
 * into view, and a fragment chosen here is the one the viewport lights up.
 */
export default function AssemblyLeft() {
  const { t } = useTranslation();
  const assembly = useAssembly((state) => state.assembly);
  const error = useAssembly((state) => state.error);
  const query = useUi((state) => state.assemblyQuery);
  const open = useUi((state) => state.assemblyOpen);
  const hidden = useUi((state) => state.assemblyHidden);
  const unpaired = useUi((state) => state.assemblyUnpaired);
  const selectedFragment = useUi((state) => state.selectedFragment);
  const selectedGroup = useUi((state) => state.assemblyGroup);

  const scroller = useRef<HTMLDivElement>(null);
  /** The fragment the list has already been scrolled to, so that it is scrolled to only once. */
  const scrolled = useRef<string | null>(null);

  const rows = useMemo<Row[]>(
    () => (assembly === null ? [] : buildRows(assembly, open, query)),
    [assembly, open, query],
  );

  const virtualizer = useVirtualizer<HTMLDivElement, HTMLDivElement>({
    count: rows.length,
    getScrollElement: () => scroller.current,
    estimateSize: () => ROW,
    overscan: OVERSCAN,
  });

  // A fragment chosen in the viewport has to be findable here: its group is opened if it is
  // closed, and the row is scrolled to once it exists. Both are one effect, because opening the
  // group is what makes the row exist — which is also why `rows` is a dependency.
  //
  // Once per selection, though, and [`scrolled`] is what remembers that: the effect runs again
  // on every change to the tree, and without it opening a group at the far end of two thousand
  // rows would snap the list back to whatever is selected.
  useEffect(() => {
    if (selectedFragment === null || assembly === null) {
      scrolled.current = null;
      return;
    }
    if (scrolled.current === selectedFragment) {
      return;
    }
    const at = rows.findIndex((row) => row.kind === "member" && row.name === selectedFragment);
    if (at >= 0) {
      scrolled.current = selectedFragment;
      virtualizer.scrollToIndex(at);
      return;
    }
    const group = groupOf(assembly, selectedFragment);
    if (group !== null) {
      useUi.getState().openAssemblyGroup(assembly.groups[group]?.members.length === 1 ? UNPAIRED : group);
    }
  }, [selectedFragment, assembly, rows, virtualizer]);

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-border px-2 py-1.5">
        <div className="relative">
          <Search
            size={12}
            aria-hidden="true"
            className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-muted"
          />
          <input
            type="text"
            value={query}
            aria-label={t("assembly.search_label")}
            placeholder={t("assembly.search")}
            onChange={(e) => {
              useUi.getState().setAssemblyQuery(e.target.value);
            }}
            className={clsx(
              "h-6 w-full rounded-md border border-border bg-panel pr-2 pl-6 text-[11px] placeholder:text-muted",
              FOCUS_RING,
            )}
          />
        </div>
      </div>

      <div
        ref={scroller}
        // A `group` and not a `tree`: a real ARIA tree owes its reader levels, set sizes and one
        // tab stop with arrow-key movement, and half a tree widget reads worse to a screen reader
        // than a named list of buttons does.
        role="group"
        aria-label={t("assembly.tree")}
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-1 py-1"
      >
        {rows.length === 0 ? (
          // Three emptinesses, and they are not the same: nothing assembled yet, nothing that
          // the search matches, and a file that would not be read — the last of which the
          // centre and A §10's banner also say, and this pane must not contradict them with
          // «Здесь появится сборка».
          <p className="px-1 py-2 text-[11px] text-muted">
            {assembly !== null
              ? t("assembly.nothing_matches")
              : error === null
                ? t("assembly.will_appear")
                : t("assembly.unreadable")}
          </p>
        ) : (
          <div className="relative w-full" style={{ height: `${String(virtualizer.getTotalSize())}px` }}>
            {virtualizer.getVirtualItems().map((item) => {
              const row = rows[item.index];
              if (row === undefined) {
                return null;
              }
              return (
                <div
                  key={item.key}
                  className="absolute top-0 left-0 w-full"
                  style={{ height: `${String(item.size)}px`, transform: `translateY(${String(item.start)}px)` }}
                >
                  {row.kind === "member" ? (
                    <button
                      type="button"
                      aria-pressed={row.name === selectedFragment}
                      title={row.name}
                      onClick={() => {
                        useUi.getState().selectFragment(row.name);
                      }}
                      onDoubleClick={() => {
                        useUi.getState().flyToFragment(row.name);
                      }}
                      className={clsx(
                        "flex h-full w-full items-center gap-1.5 rounded pr-1 pl-[26px] text-left hover:bg-panel-2",
                        row.name === selectedFragment && "bg-select",
                        FOCUS_RING,
                      )}
                    >
                      <span
                        aria-hidden="true"
                        className="h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{ background: dotColour(row.group) }}
                      />
                      <span className="min-w-0 truncate text-[11px]">{row.name}</span>
                    </button>
                  ) : row.kind === "group" ? (
                    <SectionRow
                      label={t("assembly.group_title", { n: row.index })}
                      count={row.count}
                      colour={groupColour(row.index)}
                      open={row.open}
                      hidden={hidden.has(row.index)}
                      selected={selectedGroup === row.index}
                      onOpen={() => {
                        useUi.getState().toggleAssemblyOpen(row.index);
                      }}
                      onSelect={() => {
                        useUi.getState().selectGroup(row.index);
                        useUi.getState().openAssemblyGroup(row.index);
                      }}
                      onHide={() => {
                        useUi.getState().toggleAssemblyHidden(row.index);
                      }}
                      onAlone={() => {
                        useUi.getState().setAssemblyLayout("single", row.index);
                      }}
                    />
                  ) : (
                    <SectionRow
                      label={t("assembly.unpaired_title")}
                      count={row.count}
                      colour={UNPAIRED_COLOUR}
                      open={row.open}
                      // The tray is one switch in the viewer, not a group of it: its eye is the
                      // «Без пары» chip of the toolbar, and both must say the same thing.
                      hidden={!unpaired}
                      selected={false}
                      onOpen={() => {
                        useUi.getState().toggleAssemblyOpen(UNPAIRED);
                      }}
                      onSelect={() => {
                        useUi.getState().toggleAssemblyOpen(UNPAIRED);
                      }}
                      onHide={() => {
                        useUi.getState().setAssemblyUnpaired(!unpaired);
                      }}
                      onAlone={null}
                    />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
