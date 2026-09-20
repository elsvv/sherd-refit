import { useTranslation } from "react-i18next";

import { useFitSignal } from "../../app/shortcuts";
import { api } from "../../ipc";
import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useUi } from "../../state/ui";
import Chip from "../../ui/Chip";
import FragmentView from "../../viewer/FragmentView";

/**
 * The centre of the «Вход» mode (A §7.3): one fragment in 3D, and the four switches that decide
 * what is drawn — the scan as it was captured, the fracture surface the engine found in red
 * (R §3.4), the wireframe, and «Вписать».
 *
 * Only a prepared fragment can be shown: the display meshes are what a `Prepare` writes, and the
 * window never loads a source scan (A §2.1). So the two cases where there is nothing to draw —
 * no fragment chosen, and one chosen whose turn in the preparation has not come — each say so in
 * a line of text rather than leaving an empty black rectangle.
 */
export default function InputCentre({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const selected = useUi((state) => state.selectedFragment);
  const fragmentView = useUi((state) => state.fragmentView);
  const wireframe = useUi((state) => state.wireframe);
  const fitSignal = useFitSignal((state) => state.signal);

  const info = selected === null ? undefined : view.fragments.find((row) => row.name === selected);
  const url =
    info === undefined
      ? null
      : api.assetUrl(`${view.fragments_dir}/${info.name}${fragmentView === "seg" ? ".seg" : ""}.glb`);

  if (url === null) {
    return (
      <div className="flex h-full w-full items-center justify-center px-6 text-center text-xs text-viewport-text">
        {selected === null ? t("input.pick_fragment") : t("input.fragment_pending", { name: selected })}
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <FragmentView url={url} wireframe={wireframe} fitSignal={fitSignal} />

      <div className="absolute top-2 left-2 flex gap-1" role="group" aria-label={t("input.toolbar")}>
        <Chip
          active={fragmentView === "scan"}
          onClick={() => {
            useUi.getState().setFragmentView("scan");
          }}
        >
          {t("input.view_scan")}
        </Chip>
        <Chip
          active={fragmentView === "seg"}
          onClick={() => {
            useUi.getState().setFragmentView("seg");
          }}
        >
          {t("input.view_seg")}
        </Chip>
        <Chip
          active={wireframe}
          onClick={() => {
            useUi.getState().setWireframe(!wireframe);
          }}
        >
          {t("input.wireframe")}
        </Chip>
        {/* «Вписать» asks for a fit through the same counter the `F` key bumps; the chip never
            reaches for the viewer, which is not React's to hold (A §7.2). */}
        <Chip
          onClick={() => {
            useFitSignal.getState().requestFit();
          }}
        >
          {t("input.fit")}
        </Chip>
      </div>
    </div>
  );
}
