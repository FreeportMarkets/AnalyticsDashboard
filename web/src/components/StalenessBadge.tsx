import { formatAge, isStale } from '@/lib/metrics/staleness'

/**
 * Watermark age per sync source. Neutral when fresh, amber when stale --
 * amber is reserved exclusively for this alert state, never decorative.
 */
export function StalenessBadge({
  ages,
}: {
  ages: Array<{ source: string; ageSeconds: number }>
}) {
  if (ages.length === 0) return null

  return (
    <div className="flex items-center gap-3">
      {ages.map(a => {
        const stale = isStale(a.ageSeconds)
        return (
          <div
            key={a.source}
            className={`flex items-center gap-1.5 text-xs ${stale ? 'text-alert' : 'text-ink-2'}`}
            title={`${a.source} watermark: ${formatAge(a.ageSeconds)} behind`}
          >
            <span
              aria-hidden="true"
              className={`h-1.5 w-1.5 rounded-full ${stale ? 'bg-alert' : 'bg-positive'}`}
            />
            <span>{a.source}</span>
            <span className="numeral text-ink-3">{formatAge(a.ageSeconds)}</span>
          </div>
        )
      })}
    </div>
  )
}
