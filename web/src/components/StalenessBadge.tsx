import { formatAge, isStale } from '@/lib/metrics/staleness'

/**
 * Watermark age per sync source. Renders "Live" when healthy (all sources
 * within STALE_THRESHOLD), with per-source details in a tooltip. When stale
 * (any source over threshold), shows "Data Xm behind" with per-source breakdown
 * visible below. The sync deliberately holds a 10-minute watermark lag to avoid
 * skipping late-arriving events, so 10m is normal; 15m+ means cron is not running.
 */
export function StalenessBadge({
  ages,
}: {
  ages: Array<{ source: string; ageSeconds: number }>
}) {
  if (ages.length === 0) return null

  const anyStale = ages.some(a => isStale(a.ageSeconds))

  if (!anyStale) {
    // Healthy state: compact "Live" indicator with detailed tooltip
    const sourceDetails = ages.map(a => `${a.source} ${formatAge(a.ageSeconds)} behind`).join(', ')
    const tooltipText = `Data synced — ${sourceDetails}. The sync holds a deliberate 10-minute lag so late-arriving events aren't missed.`

    return (
      <div
        className="flex items-center gap-1.5 text-xs text-ink-2"
        title={tooltipText}
      >
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-positive" />
        <span>Live</span>
        <span className="sr-only">Data is live, all sources synced within normal lag</span>
      </div>
    )
  }

  // Stale state: alert about the problem with per-source breakdown
  const maxAge = Math.max(...ages.map(a => a.ageSeconds))
  const maxFormatted = formatAge(maxAge)
  const staleCount = ages.filter(a => isStale(a.ageSeconds)).length
  const staleLabel = staleCount === ages.length ? maxFormatted : `${maxFormatted} (${staleCount}/${ages.length} sources)`

  return (
    <div>
      <div
        className="flex items-center gap-1.5 text-xs text-alert"
        title="One or more data sources are stale. The sync cron may not be running. Check logs."
      >
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-alert" />
        <span>Data {maxFormatted} behind</span>
        <span className="sr-only">Data is stale. {staleLabel} old.</span>
      </div>
      <div className="mt-1.5 flex flex-col gap-1 text-xs text-alert">
        {ages.map(a => (
          <div key={a.source} className="flex items-center gap-1.5">
            <span aria-hidden="true" className={`h-1 w-1 rounded-full ${isStale(a.ageSeconds) ? 'bg-alert' : 'bg-positive'}`} />
            <span>{a.source}</span>
            <span className="numeral text-ink-3">{formatAge(a.ageSeconds)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
