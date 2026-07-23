/**
 * Horizontal proportional bars: label + value, for "top N" breakdowns.
 * The workhorse of the console -- platform split, top events, trade
 * breakdowns all render through this.
 *
 * Three things here are deliberate and were all wrong before:
 *
 * 1. The bar is 10px tall on a visible `bg-surface` track. It used to be
 *    8px of `accent-dim` on a track that measured 1.04:1 against the page,
 *    so neither the bar nor the extent it was measured against could be
 *    seen. A proportional bar you can't see is just decoration next to the
 *    number.
 * 2. The value column is `min-w-[7rem]` and grows, not a fixed `w-16`
 *    (64px). `$52,118,904` is ~100px at 14px monospace; the old fixed
 *    width only survived because the number overflowed its own box.
 * 3. The label is `text-ink-1`, not `text-ink-2`. It is the row's subject.
 *    Its count/unit qualifier (`sublabel`) is what should recede.
 */
export function BarList({
  items,
  formatValue = (n: number) => n.toLocaleString('en-US'),
  formatValueTitle,
  emptyLabel = 'No data in this range',
}: {
  items: Array<{ label: string; value: number; sublabel?: string }>
  formatValue?: (n: number) => string
  /**
   * Hover text for the value cell. Pass alongside an abbreviating
   * `formatValue` so the exact figure stays reachable.
   */
  formatValueTitle?: (n: number) => string
  emptyLabel?: string
}) {
  if (items.length === 0) {
    return <p className="py-4 text-sm text-ink-3">{emptyLabel}</p>
  }

  const max = Math.max(...items.map(i => i.value), 1)

  return (
    <ul>
      {items.map(item => {
        const pct = Math.max((item.value / max) * 100, item.value > 0 ? 2 : 0)
        return (
          /*
           * Grid, not flex, and PROPORTIONAL, not fixed. A fixed label
           * width (w-44, or w-64 at xl) is fine in a full-width section and
           * catastrophic in a three-up row: 256px of label plus 112px of
           * value inside a ~430px column left roughly 30px for the bar, so
           * the bars in "Volume by type" collapsed to invisible stubs. The
           * track keeps a 2.5rem floor so it can never vanish entirely, and
           * the value column sizes to its own content instead of guessing.
           */
          <li
            key={item.label}
            className="row-hover -mx-2 grid grid-cols-[minmax(6rem,14rem)_minmax(3rem,1fr)_auto] items-center gap-3 rounded-sm px-2 py-2"
          >
            <div className="truncate text-sm text-ink-1" title={item.label}>
              {item.label}
              {item.sublabel && (
                <span className="ml-1.5 text-xs text-ink-3">{item.sublabel}</span>
              )}
            </div>
            <div className="relative h-2.5 overflow-hidden rounded-full bg-surface">
              <div
                className="h-full rounded-full bg-accent-bar"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div
              className="numeral whitespace-nowrap text-right text-sm text-ink-1"
              title={formatValueTitle?.(item.value)}
            >
              {formatValue(item.value)}
            </div>
          </li>
        )
      })}
    </ul>
  )
}
