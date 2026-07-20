import { sql } from '@/lib/db'

/**
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

export async function watermarkAge(): Promise<Array<{ source: string; ageSeconds: number }>> {
  const rows = (await sql(
    `SELECT source, EXTRACT(EPOCH FROM (now() - watermark_ts))::int AS age_seconds
       FROM sync_state
      WHERE source IN ('events','trades')
      ORDER BY source`
  )) as Array<{ source: string; age_seconds: number }>
  return rows.map(r => ({ source: r.source, ageSeconds: r.age_seconds }))
}
