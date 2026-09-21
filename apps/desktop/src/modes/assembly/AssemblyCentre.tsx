import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { useFitSignal } from "../../app/shortcuts";
import { api } from "../../ipc";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useAssembly } from "../../state/assembly";
import { useUi } from "../../state/ui";
import Chip from "../../ui/Chip";
import type { AssemblyViewHandle } from "../../viewer/AssemblyView";
import AssemblyView from "../../viewer/AssemblyView";

/** How the «Разъединить» slider is stepped: a hundredth is finer than the eye can see anyway. */
const EXPLODE_STEP = 0.01;

/**
 * The name the PNG is offered under: the run and the moment, so that two snapshots of one run do
 * not overwrite one another in the downloads folder.
 */
function snapshotName(runId: string | null): string {
  const now = new Date();
  const pad = (n: number): string => String(n).padStart(2, "0");
  const stamp = `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `${runId ?? "assembly"}-${stamp}.png`;
}

/**
 * The centre of the «Сборка» mode (A §7.3): the run's assembly in 3D, and the chips over its
 * top-left corner that decide what is drawn (A §7.2).
 *
 * The viewport itself is [`AssemblyView`]'s, which owns the one `AssemblyViewer` and never lets
 * React near the scene; everything here is the *choice* of what to show, which lives in
 * [`useUi`] so that it survives a trip to the «Вход» mode and back.
 *
 * With nothing assembled there is a line of text and no canvas: a black rectangle over a run
 * that is still going, or one that assembled nothing, says less than one sentence does. And a
 * run that *is* going over an earlier result leaves that result on the screen, marked as the
 * previous one — A §6's overlay sits over it, and what is underneath must not pretend to be new.
 */
export default function AssemblyCentre({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const assembly = useAssembly((state) => state.assembly);
  const runId = useAssembly((state) => state.runId);
  const loading = useAssembly((state) => state.loading);

  const colour = useUi((state) => state.assemblyColour);
  const layout = useUi((state) => state.assemblyLayout);
  const single = useUi((state) => state.assemblySingle);
  const hidden = useUi((state) => state.assemblyHidden);
  const unpaired = useUi((state) => state.assemblyUnpaired);
  const explode = useUi((state) => state.assemblyExplode);
  const labels = useUi((state) => state.assemblyLabels);
  const selected = useUi((state) => state.selectedFragment);
  const fly = useUi((state) => state.assemblyFly);
  const fitSignal = useFitSignal((state) => state.signal);

  const viewer = useRef<AssemblyViewHandle>(null);
  /** Whether the «Разъединить» slider is out; one popover, and only while it is asked for. */
  const [slider, setSlider] = useState(false);

  // The arrangement belongs to the run it was made for: group 2 of one run is not group 2 of
  // another. Told here, where the run being drawn is known, and a no-op while it is the same one.
  useEffect(() => {
    useUi.getState().showRun(runId);
  }, [runId]);

  // «Все группы», «Одна группа» and «Без пары» each put a different *set* of things in front of
  // the camera, and the camera has to meet it: a group that `packDiscs` put off to one side would
  // otherwise be isolated out of frame, and the tray — a second block hung under the first — came
  // up in the first look as a sliver at the bottom edge of a viewport still framed on the groups.
  //
  // Only these. A group's eye takes away something already framed, and «Разъединить» is a slider:
  // re-framing on every tick of a drag would be motion sickness, and a fit also recomputes the
  // direction, which would undo the orbit of anyone who had turned the assembly to look at a
  // seam. «Вписать» and `F` are one press away for both.
  useEffect(() => {
    useFitSignal.getState().requestFit();
  }, [layout, single, unpaired]);

  // The inspector's partner rows and the tree's double click both ask for a fragment to be
  // brought into the middle; the request is cleared so that asking twice moves the camera twice.
  useEffect(() => {
    if (fly === null) {
      return;
    }
    viewer.current?.flyTo(fly.name);
    useUi.getState().clearFly();
  }, [fly]);

  // Every fragment this run placed that also has a display mesh. Built from the workspace's own
  // index rather than from the run's snapshot, because a mesh is what the viewer can load and
  // `fragments/` is where the meshes are (A §2.1: the window never walks the input folder).
  const fragments = useMemo(() => {
    if (assembly === null) {
      return [];
    }
    const wanted = new Set(assembly.groups.flatMap((group) => group.members));
    return view.fragments
      .filter((info) => wanted.has(info.name))
      .map((info) => ({ name: info.name, url: api.assetUrl(`${view.fragments_dir}/${info.name}.glb`) }));
  }, [assembly, view.fragments, view.fragments_dir]);

  if (assembly === null) {
    return (
      <div className="flex h-full w-full items-center justify-center px-6 text-center text-xs text-viewport-text">
        {loading ? t("assembly.reading") : t("assembly.will_appear")}
      </div>
    );
  }

  /**
   * «Снимок PNG». The canvas is drawn without `preserveDrawingBuffer`, so its pixels are readable
   * only in the turn of the event loop they were drawn in — which is why this reads the data URL
   * straight out of the click and awaits nothing before it.
   */
  const shoot = (): void => {
    const url = viewer.current?.screenshot();
    if (url === null || url === undefined) {
      return;
    }
    const link = document.createElement("a");
    link.href = url;
    link.download = snapshotName(runId);
    link.click();
  };

  return (
    <div className="relative h-full w-full">
      <AssemblyView
        ref={viewer}
        fragments={fragments}
        assembly={assembly}
        colourMode={colour}
        layout={layout}
        group={single ?? undefined}
        hidden={hidden}
        unassembled={unpaired}
        explode={explode}
        labels={labels}
        selected={selected}
        onSelect={(name) => {
          useUi.getState().selectFragment(name);
        }}
        fitSignal={fitSignal}
      />

      {/* Bounded on the right as well as on the left, so that a narrow window wraps the chips
          onto a second row instead of pushing «Снимок PNG» out of the viewport. */}
      <div
        className="absolute top-2 right-2 left-2 flex flex-wrap items-start gap-1"
        role="group"
        aria-label={t("assembly.toolbar")}
      >
        <Chip
          onClick={() => {
            useFitSignal.getState().requestFit();
          }}
        >
          {t("assembly.fit")}
        </Chip>
        <Chip
          active={labels}
          onClick={() => {
            useUi.getState().toggleAssemblyLabels();
          }}
        >
          {t("assembly.labels")}
        </Chip>
        <Chip
          title={t("assembly.colour_hint")}
          onClick={() => {
            useUi.getState().cycleAssemblyColour();
          }}
        >
          {t("assembly.colour", { mode: t(`assembly.colour_${colour}`) })}
        </Chip>
        <Chip
          active={layout === "spread"}
          onClick={() => {
            useUi.getState().setAssemblyLayout("spread");
          }}
        >
          {t("assembly.layout_all")}
        </Chip>
        <Chip
          active={layout === "single"}
          onClick={() => {
            useUi.getState().setAssemblyLayout("single");
          }}
        >
          {t("assembly.layout_one")}
        </Chip>
        <Chip
          active={unpaired}
          onClick={() => {
            useUi.getState().setAssemblyUnpaired(!unpaired);
          }}
        >
          {t("assembly.tray")}
        </Chip>

        <div className="relative">
          <Chip
            active={slider || explode > 0}
            aria-expanded={slider}
            onClick={() => {
              setSlider(!slider);
            }}
          >
            {t("assembly.explode")}
          </Chip>
          {slider ? (
            <div className="absolute top-full left-0 z-10 mt-1 flex w-52 items-center gap-2 rounded-md border border-border bg-panel p-2 shadow-lg">
              <input
                type="range"
                min={0}
                max={1}
                step={EXPLODE_STEP}
                value={explode}
                aria-label={t("assembly.explode")}
                onChange={(e) => {
                  useUi.getState().setAssemblyExplode(Number(e.target.value));
                }}
                className="min-w-0 flex-1 accent-accent"
              />
              <span className="w-9 shrink-0 text-right text-[11px] text-muted tabular-nums">
                {t("assembly.explode_value", { percent: `${String(Math.round(explode * 100))}%` })}
              </span>
            </div>
          ) : null}
        </div>

        <Chip onClick={shoot}>{t("assembly.snapshot")}</Chip>
      </div>

      {/* A §5: the assembly on the screen during a run is the *last* one, and saying so is the
          difference between a stale picture and a lie. It is a caption on the picture and sits in
          the bottom-left corner: the top of the viewport is the chips' and the bottom middle is
          A §6's running card, and the first look had this badge lying over «Снимок PNG». */}
      {view.job?.kind === "run" ? (
        <p className="pointer-events-none absolute bottom-2 left-2 rounded-full border border-border bg-panel px-2 py-0.5 text-[10px] text-muted">
          {t("assembly.previous")}
        </p>
      ) : null}
    </div>
  );
}
