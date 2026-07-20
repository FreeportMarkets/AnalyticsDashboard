/**
 * Pure staleness helpers -- no imports, no I/O. Deliberately split out of
 * staleness.ts so importing these (e.g. from a component or a unit test)
 * cannot drag in `@/lib/db`, which throws at import time if DATABASE_URL is
 * unset. staleness.ts re-exports these so existing `from '@/lib/metrics/staleness'`
 * imports keep working unchanged.
 *
 * The incremental sync deliberately holds its watermark 10 minutes behind
 * wall-clock so late-arriving client events are not stepped over. 15 minutes
 * therefore means the cron itself has stopped, not that it is working normally.
 */
export const STALE_THRESHOLD_SECONDS = 15 * 60

export function isStale(ageSeconds: number): boolean {
  return ageSeconds > STALE_THRESHOLD_SECONDS
}

export function formatAge(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}
