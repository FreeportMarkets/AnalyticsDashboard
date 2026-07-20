import { AnimatedNumber } from './AnimatedNumber'

function formatDelta(current: number, previous: number): { label: string; positive: boolean } | null {
  if (previous === 0) {
    if (current === 0) return null
    return { label: 'new', positive: current > 0 }
  }
  const pct = ((current - previous) / previous) * 100
  const sign = pct >= 0 ? '+' : ''
  return { label: `${sign}${pct.toFixed(1)}%`, positive: pct >= 0 }
}

/**
 * Label, big tabular value, optional delta vs. a previous period. No card
 * border -- tiles sit inside a hairline-divided row; the numbers are the
 * design. No sparkline: a 7-point, unlabeled 24px squiggle communicated
 * nothing the delta didn't already say, and it pushed a gap between the
 * value and its delta. The one real trend chart per page now lives below
 * the KPI row (see TimeSeriesBars) instead of being smeared thin across
 * every tile.
 */
export function StatTile({
  label,
  value,
  previousValue,
  format,
}: {
  label: string
  value: number
  previousValue?: number
  format?: (n: number) => string
}) {
  const delta = previousValue !== undefined ? formatDelta(value, previousValue) : null

  return (
    <div className="min-w-0 py-4 first:pt-0 sm:py-0">
      <div className="text-xs uppercase tracking-wide text-ink-2">{label}</div>
      <AnimatedNumber
        value={value}
        format={format}
        className="numeral animate-numeral-in mt-1 block text-3xl font-semibold text-ink-1"
      />
      {delta && (
        <div
          className={`numeral mt-0.5 text-xs ${delta.positive ? 'text-positive' : 'text-negative'}`}
        >
          <span aria-hidden="true">{delta.positive ? '↑' : '↓'}</span> {delta.label}
          <span className="text-ink-3"> vs prior period</span>
        </div>
      )}
    </div>
  )
}
