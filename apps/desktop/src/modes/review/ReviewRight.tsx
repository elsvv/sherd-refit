import { useTranslation } from "react-i18next";

import { useReview } from "../../state/review";

/**
 * The right pane of the «Ревью» mode (A §8.3): the inspector of the selected join.
 *
 * It says which pair is on the screen and nothing else yet. What belongs here — the scores
 * against the run's own limits, «Почему не подтверждён сам» out of [`explain`] and [`headline`],
 * and the pair's other poses — is the second half of this screen and is built next; the pure
 * parts it needs ([`limitsOf`], [`scoreLines`]) are already in `queue.ts` and tested.
 */
export default function ReviewRight() {
  const { t } = useTranslation();
  const selected = useReview((state) => state.selected);

  return (
    <div className="flex h-full flex-col gap-2 px-2 py-2">
      <h2 className="truncate text-[10px] font-semibold tracking-wide text-muted uppercase">
        {selected === null ? t("review.no_pair") : `${selected.a} – ${selected.b}`}
      </h2>
      <p className="text-[11px] leading-snug text-muted">{t("review.inspector_soon")}</p>
    </div>
  );
}
