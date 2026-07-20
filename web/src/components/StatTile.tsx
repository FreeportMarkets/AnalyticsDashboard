import { AnimatedNumber } from './AnimatedNumber'
import { Sparkline } from './Sparkline'

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
 * Label, big tabular value, optional delta vs. a previous period, optional
 * inline sparkline. No card border -- tiles sit inside a hairline-divided
 * row; the numbers are the design.
 */
export function StatTile({
  label,
  value,
  previousValue,
  format,
  sparklineValues,
}: {
  label: string
  value: number
  previousValue?: number
  format?: (n: number) => string
  sparklineValues?: number[]
}) {
  const delta = previousValue !== undefined ? formatDelta(value, previousValue) : null

  return (
    <div className="min-w-0 py-4 first:pt-0 sm:py-0">
      <div className="text-xs uppercase tracking-wide text-ink-2">{label}</div>
      <div className="mt-1.5 flex items-end justify-between gap-3">
        <AnimatedNumber
          value={value}
          format={format}
          className="numeral animate-numeral-in text-3xl font-semibold text-ink-1"
        />
        {sparklineValues && sparklineValues.length > 1 && (
          <Sparkline values={sparklineValues} className="mb-1 shrink-0" />
        )}
      </div>
      {delta && (
        <div
          className={`numeral mt-1 text-xs ${delta.positive ? 'text-positive' : 'text-negative'}`}
        >
          <span aria-hidden="true">{delta.positive ? '↑' : '↓'}</span> {delta.label}
          <span className="text-ink-3"> vs prior period</span>
        </div>
      )}
    </div>
  )
}
