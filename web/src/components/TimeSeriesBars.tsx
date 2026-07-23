/**
 * The "one real chart" for a page's primary daily series (daily events on
 * Overview, daily active users on Users) -- replaces the decorative KPI-tile
 * sparklines (no axis, no scale, 24px tall) with something an operator can
 * actually read: a fixed-height track (so bar heights resolve against a real
 * pixel value, not an auto-height ancestor -- see the zero-height bug this
 * file exists to avoid reintroducing), a y-axis max label + gridline, and a
 * handful of evenly spaced, readable date ticks instead of just first/last.
 *
 * Inline CSS only, no chart library, matching the rest of the console.
 */
export function TimeSeriesBars({
  data,
  height = 160,
  formatValue = (n: number) => Math.round(n).toLocaleString('en-US'),
  formatDay,
  tickCount = 6,
  emptyLabel = 'No data in this range',
}: {
  data: Array<{ day: string; value: number }>
  height?: number
  formatValue?: (n: number) => string
  formatDay: (day: string) => string
  tickCount?: number
  emptyLabel?: string
}) {
  if (data.length === 0) {
    return <p className="py-4 text-sm text-ink-3">{emptyLabel}</p>
  }

  const max = Math.max(...data.map(d => d.value), 1)

  // Up to `tickCount` evenly spaced x-axis labels (always first + last),
  // rather than a label under every single bar.
  const tickIndices = new Set(
    data.length <= tickCount
      ? data.map((_, i) => i)
      : Array.from({ length: tickCount }, (_, i) =>
          Math.round((i * (data.length - 1)) / (tickCount - 1))
        )
  )

  return (
    <div>
      <div className="flex items-stretch gap-2">
        <div
          className="numeral w-10 shrink-0 pt-0.5 text-right text-[10px] leading-none text-ink-3"
          aria-hidden="true"
        >
          {formatValue(max)}
        </div>
        <div className="relative min-w-0 flex-1" style={{ height }}>
          <div className="absolute inset-x-0 top-0 border-t border-hairline" aria-hidden="true" />
          {/* Baseline. Without it the bars float against the page and you
              can't tell a short bar from a cropped one. */}
          <div className="absolute inset-x-0 bottom-0 border-b border-hairline" aria-hidden="true" />
          <div className="flex h-full items-end gap-[3px]">
            {data.map(d => (
              <div key={d.day} className="group relative h-full min-w-0 flex-1">
                {/*
                 * Capped at 88px and centred in its slot. Uncapped, a 7-day
                 * range on a 1600px page draws seven ~200px-wide blocks --
                 * at that width the chart stops reading as a chart and
                 * becomes a wall of colour. The cap only ever engages on
                 * short ranges; at 30d/90d the bars are naturally thinner
                 * and the slot width wins. Centring keeps each bar aligned
                 * with its own flex-1 x-axis tick below.
                 */}
                <div
                  className="absolute inset-x-0 bottom-0 mx-auto max-w-[88px] rounded-t-sm bg-accent-bar transition-colors group-hover:bg-accent"
                  style={{ height: `${Math.max((d.value / max) * 100, d.value > 0 ? 1.5 : 0)}%` }}
                />
                <span className="pointer-events-none absolute -top-6 left-1/2 z-10 hidden -translate-x-1/2 whitespace-nowrap rounded-sm border border-hairline bg-raised px-1.5 py-0.5 text-[10px] text-ink-1 group-hover:block">
                  {formatDay(d.day)} · {formatValue(d.value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="numeral ml-12 mt-1.5 flex text-[10px] text-ink-3">
        {data.map((d, i) => (
          <span
            key={d.day}
            className="min-w-0 flex-1 truncate text-center first:text-left last:text-right"
          >
            {tickIndices.has(i) ? formatDay(d.day) : ''}
          </span>
        ))}
      </div>
    </div>
  )
}
