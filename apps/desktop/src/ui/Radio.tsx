import clsx from "clsx";

/** One choice of a group. */
export interface RadioOption<T extends string> {
  value: T;
  label: string;
  /** What choosing this costs, in one sentence — the cards of A §7.4 carry one, the pills do not. */
  hint?: string | undefined;
}

export interface RadioProps<T extends string> {
  /**
   * Groups the inputs in the document. Two radio groups on one screen with the same name are one
   * group, so this has to be unique within the dialog that holds them.
   */
  name: string;
  /** What the group is choosing between; rendered as the section's heading. */
  legend: string;
  value: T;
  options: readonly RadioOption<T>[];
  onChange: (value: T) => void;
  /** `card` is A §7.4's three preset cards; `inline` the «Авто · CPU · GPU» row. */
  variant?: "card" | "inline" | undefined;
}

/**
 * A choice of one out of a few (A §7.4's preset and executor). Native `<input type="radio">`s
 * under the paint, because a set of buttons would have to re-implement what the browser already
 * does for a radio group — the arrow keys walk it, Tab enters and leaves it as one stop, and a
 * screen reader says «2 of 3».
 *
 * The ring is drawn on the label through `has-[:focus-visible]`, since the input itself is not on
 * the screen: without it the keyboard would be moving a focus nobody can see (A §7.1).
 */
export default function Radio<T extends string>({ name, legend, value, options, onChange, variant = "card" }: RadioProps<T>) {
  const card = variant === "card";
  return (
    <fieldset>
      <legend className="mb-1.5 text-[10px] tracking-wide text-muted uppercase">{legend}</legend>
      <div className={clsx(card ? "grid grid-cols-3 gap-1.5" : "flex flex-wrap items-center gap-1.5")}>
        {options.map((option) => {
          const chosen = option.value === value;
          return (
            <label
              key={option.value}
              className={clsx(
                "cursor-default border",
                card
                  ? "flex flex-col gap-0.5 rounded-md p-2"
                  : "inline-flex h-6 items-center rounded-full px-2.5 text-xs whitespace-nowrap",
                chosen ? "border-accent bg-select text-text" : "border-border bg-panel text-text hover:bg-panel-2",
                "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent",
              )}
            >
              <input
                type="radio"
                name={name}
                value={option.value}
                checked={chosen}
                onChange={() => {
                  onChange(option.value);
                }}
                className="sr-only"
              />
              <span className={clsx(card && "text-xs font-semibold", chosen && !card && "text-accent")}>
                {option.label}
              </span>
              {card && option.hint !== undefined ? (
                <span className="text-[10px] leading-snug text-muted">{option.hint}</span>
              ) : null}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
