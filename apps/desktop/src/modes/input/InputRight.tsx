import { useTranslation } from "react-i18next";

import type { WorkspaceView } from "../../ipc/bindings/WorkspaceView";
import { useUi } from "../../state/ui";
import { useWorkspace } from "../../state/workspace";
import Button from "../../ui/Button";
import { formatBytes, formatCount, formatDecimal, formatExtent, formatPercent, warningText } from "./format";

/** One line of the inspector's key–value list. */
interface Field {
  key: string;
  label: string;
  value: string;
}

/**
 * The right pane of the «Вход» mode (A §7.3): what is known about the chosen fragment, what is
 * wrong with it in plain words, and the one decision this mode offers — leaving it out.
 *
 * The file's name and size are known from the moment the folder is linked; everything else is
 * measured by a `Prepare`, so a fragment whose turn has not come shows the two and says the rest
 * is on its way. Excluding works either way: it is a mark in `sherd-workspace.json` and not a
 * fact about a mesh (A §5.1).
 */
export default function InputRight({ view }: { view: WorkspaceView }) {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const selected = useUi((state) => state.selectedFragment);

  const stamp = selected === null ? undefined : view.files.find((file) => file.name === selected);
  if (stamp === undefined) {
    // Either nothing is chosen, or the chosen fragment is no longer in the folder — a file
    // deleted under the window is not an error, it is one fewer row.
    return <p className="px-2 py-3 text-[11px] text-muted">{t("input.nothing_selected")}</p>;
  }

  const info = view.fragments.find((row) => row.name === stamp.name);
  const excluded = view.excluded.includes(stamp.name);

  const fields: Field[] = [
    { key: "file", label: t("input.field_file"), value: stamp.file },
    { key: "size", label: t("input.field_size"), value: formatBytes(stamp.size, language) },
  ];
  if (info !== undefined) {
    const stats = info.stats;
    fields.push(
      { key: "orig_faces", label: t("input.field_orig_faces"), value: formatCount(stats.orig_faces, language) },
      { key: "faces", label: t("input.field_faces"), value: formatCount(stats.faces, language) },
      { key: "thickness", label: t("input.field_thickness"), value: formatDecimal(stats.thickness) },
      { key: "watertight", label: t("input.field_watertight"), value: stats.watertight ? t("input.yes") : t("input.no") },
      {
        key: "fracture",
        label: t("input.field_fracture"),
        value: t("input.fracture_value", { percent: formatPercent(stats.fracture_area_fraction) }),
      },
      { key: "extent", label: t("input.field_extent"), value: formatExtent(stats.extent) },
    );
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-2">
      <h2 className="truncate text-xs font-semibold" title={stamp.name}>
        {stamp.name}
      </h2>

      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-[11px]">
        {fields.map((field) => (
          <div key={field.key} className="contents">
            <dt className="text-muted">{field.label}</dt>
            <dd className="min-w-0 truncate text-right tabular-nums" title={field.value}>
              {field.value}
            </dd>
          </div>
        ))}
      </dl>

      {info === undefined ? (
        <p className="mt-2 text-[11px] text-muted">{t("input.fragment_pending", { name: stamp.name })}</p>
      ) : (
        info.warnings.map((warning) => (
          <p key={warning.warning} className="mt-2 rounded-md border border-warn p-1.5 text-[11px] text-warn">
            {warningText(warning, t)}
          </p>
        ))
      )}

      <div className="mt-3">
        <Button
          onClick={() => {
            void useWorkspace.getState().excludeFragment(stamp.name, !excluded);
          }}
        >
          {excluded ? t("input.include") : t("input.exclude")}
        </Button>
        <p className="mt-1 text-[10px] text-muted">{t("input.exclude_note")}</p>
      </div>
    </div>
  );
}
