import { utcDateOf } from '@/lib/time'

/**
 * How far behind wall-clock the watermark is held.
 *
 * The FreeApp client flushes every 30s and requeues up to 500 events on failure
 * (FreeApp/services/AnalyticsService.ts:5-7, :150-153), so an event's `timestamp`
 * can precede its arrival by minutes. Advancing the watermark to `now` would step
 * over those rows permanently.
 */
export const WATERMARK_LAG_MS = 10 * 60 * 1000

export function computeNextWatermark(now: Date): Date {
  return new Date(now.getTime() - WATERMARK_LAG_MS)
}

/** Current and previous UTC date partitions -- events near UTC midnight land in the next one. */
export function datesToScan(now: Date): string[] {
  const prev = new Date(now.getTime() - 24 * 60 * 60 * 1000)
  return [utcDateOf(prev), utcDateOf(now)]
}

type SqlTag = ReturnType<typeof import('@neondatabase/serverless').neon>

export async function readWatermark(sql: SqlTag, source: string, fallback: Date): Promise<Date> {
  const rows = (await sql`
    SELECT watermark_ts FROM sync_state WHERE source = ${source}
  `) as Array<{ watermark_ts: string | Date }>
  if (rows.length === 0) return fallback
  return new Date(rows[0]!.watermark_ts)
}

export async function advanceWatermark(sql: SqlTag, source: string, ts: Date): Promise<void> {
  // GREATEST guards against a concurrent or replayed run moving the watermark backwards.
  await sql`
    INSERT INTO sync_state (source, watermark_ts, updated_at)
    VALUES (${source}, ${ts.toISOString()}, now())
    ON CONFLICT (source) DO UPDATE
      SET watermark_ts = GREATEST(sync_state.watermark_ts, EXCLUDED.watermark_ts),
          updated_at   = now()
  `
}
