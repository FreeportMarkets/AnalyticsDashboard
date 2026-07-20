import { sql } from '@/lib/db'

// Pure helpers live in staleness-format.ts (no db.ts import, so they can be
// imported without DATABASE_URL being set). Re-exported here so existing
// `from '@/lib/metrics/staleness'` call sites keep working unchanged.
export { STALE_THRESHOLD_SECONDS, isStale, formatAge } from './staleness-format'

export async function watermarkAge(): Promise<Array<{ source: string; ageSeconds: number }>> {
  const rows = (await sql(
    `SELECT source, EXTRACT(EPOCH FROM (now() - watermark_ts))::int AS age_seconds
       FROM sync_state
      WHERE source IN ('events','trades')
      ORDER BY source`
  )) as Array<{ source: string; age_seconds: number }>
  return rows.map(r => ({ source: r.source, ageSeconds: r.age_seconds }))
}
