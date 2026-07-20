import { mapEvent } from './mapEvent'
import { computeNextWatermark, datesToScan } from './watermark'
import type { EventRow, SyncResult } from './types'

export type EventFetcher = (date: string, afterSk: string) => Promise<unknown[]>

export interface SyncEventsDeps {
  now: Date
  readWatermark: () => Promise<Date>
  advanceWatermark: (source: string, ts: Date) => Promise<void>
  ensurePartitions: (dates: string[]) => Promise<string[]>
  fetchEvents: EventFetcher
  insertEvents: (
    rows: EventRow[],
    onRowFailure?: (row: EventRow, reason: string) => Promise<void>
  ) => Promise<number>
  quarantine: (raw: unknown, reason: string) => Promise<void>
}

export const EVENTS_SOURCE = 'events'

/**
 * One incremental sync tick.
 *
 * Order matters: partitions are ensured before any insert, and the watermark is
 * advanced only after every insert has succeeded. A throw anywhere leaves the
 * watermark untouched, so the next tick re-reads the same window -- safe because
 * inserts are ON CONFLICT DO NOTHING.
 *
 * Partitions are ensured from the DISTINCT DATES OF THE MAPPED ROWS actually
 * about to be inserted, NOT from the scan window (`dates`, below). `date` is a
 * DynamoDB partition key, but the ingest endpoint that writes it is
 * unauthenticated -- a crafted event can carry any `date` mapEvent accepts
 * (year 0001-9999), so a fetch of "today"/"yesterday" is not proof every
 * returned row's `date` field is today or yesterday. Ensuring only the scan
 * window would let such a row slip into `events_default` (Postgres's DEFAULT
 * partition), and once a row for a given month sits there, Postgres refuses
 * to ever create that month's dedicated partition ("updated partition
 * constraint ... would be violated") -- a permanent wedge from one row. Ensuring
 * per row-date instead is correct by construction: a partition always exists
 * before any row that belongs in it is written, so nothing can land in the
 * default partition.
 */
export async function syncEvents(deps: SyncEventsDeps): Promise<SyncResult> {
  const watermark = await deps.readWatermark()
  const dates = datesToScan(deps.now)

  // The sort key is `{timestamp}#{wallet8}#{rand8}`, so it sorts by time and a
  // bare ISO timestamp is a valid lower bound for a key-condition range query.
  const afterSk = watermark.toISOString()

  let scanned = 0
  let inserted = 0
  let quarantined = 0

  for (const date of dates) {
    const items = await deps.fetchEvents(date, afterSk)
    scanned += items.length

    const rows: EventRow[] = []
    for (const item of items) {
      const mapped = mapEvent(item)
      if (mapped.ok) rows.push(mapped.value)
      else {
        await deps.quarantine(item, mapped.reason)
        quarantined += 1
      }
    }
    if (rows.length > 0) {
      const rowDates = [...new Set(rows.map(r => r.date))]
      await deps.ensurePartitions(rowDates)
      inserted += await deps.insertEvents(rows, (row, reason) => deps.quarantine(row, reason))
    }
  }

  const next = computeNextWatermark(deps.now)
  await deps.advanceWatermark(EVENTS_SOURCE, next)

  return { scanned, inserted, quarantined, watermark: next.toISOString() }
}
