/**
 * Horizontal proportional bars: label + value, for "top N" breakdowns.
 * The workhorse of the console -- platform split, top events, trade
 * breakdowns all render through this. No card chrome; rows are separated
 * by hover state alone, bars are the only "chrome."
 */
export function BarList({
  items,
  formatValue = (n: number) => n.toLocaleString('en-US'),
  emptyLabel = 'No data in this range',
}: {
  items: Array<{ label: string; value: number; sublabel?: string }>
  formatValue?: (n: number) => string
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
          <li
            key={item.label}
            className="row-hover -mx-2 flex items-center gap-3 rounded-sm px-2 py-1.5"
          >
            <div className="w-44 shrink-0 truncate text-sm text-ink-2" title={item.label}>
              {item.label}
              {item.sublabel && (
                <span className="ml-1.5 text-xs text-ink-3">{item.sublabel}</span>
              )}
            </div>
            <div className="relative h-2 min-w-0 flex-1 overflow-hidden rounded-sm bg-surface">
              <div
                className="h-full rounded-sm bg-accent-dim"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="numeral w-16 shrink-0 text-right text-sm text-ink-1">
              {formatValue(item.value)}
            </div>
          </li>
        )
      })}
    </ul>
  )
}
