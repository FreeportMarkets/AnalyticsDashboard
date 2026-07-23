import { AnimatedNumber } from './AnimatedNumber'

/**
 * Smallest previous-period value that gets a percentage. Below this, a
 * percentage is arithmetically true and practically noise: 3 trades -> 9
 * renders "+200.0%", which reads like a step change and isn't one. Under
 * the floor we show the absolute movement instead, which is the honest
 * figure at that scale.
 */
const PCT_FLOOR = 20

export function formatDelta(
  current: number,
  previous: number
): { label: string; direction: 'up' | 'down' | 'flat' } | null {
  const diff = current - previous
  const direction = diff > 0 ? 'up' : diff < 0 ? 'down' : 'flat'

  if (previous === 0) {
    if (current === 0) return null
    return { label: 'new', direction }
  }

  if (Math.abs(previous) < PCT_FLOOR) {
    const sign = diff > 0 ? '+' : ''
    return { label: `${sign}${diff.toLocaleString('en-US')}`, direction }
  }

  const pct = (diff / previous) * 100
  const sign = pct >= 0 ? '+' : ''
  return { label: `${sign}${pct.toFixed(1)}%`, direction }
}

/**
 * Label, big tabular value, optional delta vs. a previous period. No card
 * border -- tiles sit inside a hairline-divided row; the numbers are the
 * design. No sparkline: a 7-point, unlabeled 24px squiggle communicated
 * nothing the delta didn't already say, and it pushed a gap between the
 * value and its delta. The one real trend chart per page now lives below
 * the KPI row (see TimeSeriesBars) instead of being smeared thin across
 * every tile.
 *
 * The value step is fluid across breakpoints (`text-2xl` -> `text-3xl`)
 * because a five-across KPI row gives each tile roughly a fifth of the
 * content width, and a full USD figure like `$89,794,594` does not fit at
 * 30px in that column below ~1500px. Sizing down is better than wrapping
 * a headline number onto two lines.
 *
 * "vs prior period" is NOT repeated per tile -- five identical captions is
 * noise. The comparison is stated once, by the caller, above the row.
 */
export function StatTile({
  label,
  value,
  previousValue,
  format,
  valueTitle,
}: {
  label: string
  value: number
  previousValue?: number
  format?: (n: number) => string
  /**
   * Hover text for the value. Use when `format` abbreviates (e.g. `$89.8M`)
   * so the exact figure is still one hover away rather than lost.
   */
  valueTitle?: string
}) {
  const delta = previousValue !== undefined ? formatDelta(value, previousValue) : null
  const fmt = format ?? ((n: number) => Math.round(n).toLocaleString('en-US'))

  const deltaTone =
    delta?.direction === 'up'
      ? 'text-positive'
      : delta?.direction === 'down'
        ? 'text-negative'
        : 'text-ink-3'
  const deltaGlyph = delta?.direction === 'up' ? '↑' : delta?.direction === 'down' ? '↓' : '→'

  return (
    <div className="min-w-0 py-4 first:pt-0 sm:py-0">
      <div className="truncate text-xs font-medium uppercase tracking-wide text-ink-2">{label}</div>
      <div title={valueTitle}>
        <AnimatedNumber
          value={value}
          format={format}
          className="numeral animate-numeral-in mt-1 block truncate text-2xl font-semibold tracking-tight text-ink-1 xl:text-3xl"
        />
      </div>
      {delta && (
        <div
          className={`numeral mt-1 truncate text-xs ${deltaTone}`}
          title={previousValue !== undefined ? `Prior period: ${fmt(previousValue)}` : undefined}
        >
          {/* The arrow is decorative -- the sign is already in the label,
              and direction must not be carried by color alone. */}
          <span aria-hidden="true">{deltaGlyph}</span> {delta.label}
        </div>
      )}
    </div>
  )
}
