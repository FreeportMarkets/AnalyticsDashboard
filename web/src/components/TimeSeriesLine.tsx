/**
 * A single-series line + area chart for a daily time series. Replaces the
 * stacked-bar treatment that rendered as giant blocks at low day counts
 * (7 bars across a wide container = ~200px squares). A line reads cleanly at
 * any width and matches the Streamlit dashboard it's compared against.
 *
 * Pure SVG, no chart library. The line and area are static (SSR-friendly);
 * per-point hover uses invisible full-height columns over the SVG, the same
 * group-hover pattern as TimeSeriesBars, so no client JS is required.
 *
 * The y-axis is anchored at 0 (not min-value) so bar/line heights encode
 * absolute magnitude honestly -- a volume chart that floated its baseline
 * would exaggerate day-to-day swings.
 */

export interface LinePoint {
  day: string
  value: number
  /** Optional richer tooltip lines; falls back to formatValue(value). */
  tooltip?: React.ReactNode
}

export function TimeSeriesLine({
  data,
  height = 160,
  formatValue = (n: number) => Math.round(n).toLocaleString('en-US'),
  formatDay,
  tickCount = 6,
  emptyLabel = 'No data in this range',
}: {
  data: LinePoint[]
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
  // Use a 0..100 viewBox on X and 0..100 on Y; the SVG scales to its box via
  // preserveAspectRatio=none, so we never need pixel widths and it's fully
  // responsive. Y is inverted (0 at bottom).
  const n = data.length
  const x = (i: number) => (n === 1 ? 50 : (i / (n - 1)) * 100)
  const y = (v: number) => 100 - (v / max) * 100

  const linePath = data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(2)} ${y(d.value).toFixed(2)}`).join(' ')
  const areaPath = `${linePath} L ${x(n - 1).toFixed(2)} 100 L ${x(0).toFixed(2)} 100 Z`

  const tickIndices = new Set(
    n <= tickCount
      ? data.map((_, i) => i)
      : Array.from({ length: tickCount }, (_, i) => Math.round((i * (n - 1)) / (tickCount - 1)))
  )

  return (
    <div>
      <div className="flex items-stretch gap-2">
        <div
          className="numeral w-12 shrink-0 pt-0.5 text-right text-[10px] leading-none text-ink-3"
          aria-hidden="true"
        >
          {formatValue(max)}
        </div>
        <div className="relative min-w-0 flex-1" style={{ height }}>
          <div className="absolute inset-x-0 top-0 border-t border-hairline" aria-hidden="true" />
          <div className="absolute inset-x-0 bottom-0 border-b border-hairline" aria-hidden="true" />
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="absolute inset-0 h-full w-full"
            aria-hidden="true"
          >
            <defs>
              <linearGradient id="tsl-area" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-accent)" stopOpacity="0.28" />
                <stop offset="100%" stopColor="var(--color-accent)" stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={areaPath} fill="url(#tsl-area)" />
            {/* vectorEffect keeps the stroke 1.5px at any scale, so
                preserveAspectRatio=none doesn't smear it. */}
            <path
              d={linePath}
              fill="none"
              stroke="var(--color-accent)"
              strokeWidth={1.5}
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
          {/* Invisible hover columns: a dot at the point + a tooltip. */}
          <div className="absolute inset-0 flex">
            {data.map((d, i) => (
              <div key={d.day} className="group relative h-full flex-1">
                <span
                  className="pointer-events-none absolute left-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent opacity-0 transition-opacity group-hover:opacity-100"
                  style={{ top: `${y(d.value)}%` }}
                  aria-hidden="true"
                />
                <span className="pointer-events-none absolute -top-7 left-1/2 z-10 hidden -translate-x-1/2 flex-col whitespace-nowrap rounded-sm border border-hairline bg-raised px-1.5 py-1 text-[10px] text-ink-1 group-hover:flex">
                  <span className="text-ink-2">{formatDay(d.day)}</span>
                  {d.tooltip ?? formatValue(d.value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="numeral ml-14 mt-1.5 flex text-[10px] text-ink-3">
        {data.map((d, i) => (
          <span key={d.day} className="min-w-0 flex-1 truncate text-center first:text-left last:text-right">
            {tickIndices.has(i) ? formatDay(d.day) : ''}
          </span>
        ))}
      </div>
    </div>
  )
}
