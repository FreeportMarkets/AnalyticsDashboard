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
  insertEvents: (rows: EventRow[]) => Promise<number>
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
 */
export async function syncEvents(deps: SyncEventsDeps): Promise<SyncResult> {
  const watermark = await deps.readWatermark()
  const dates = datesToScan(deps.now)
  await deps.ensurePartitions(dates)

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
    if (rows.length > 0) inserted += await deps.insertEvents(rows)
  }

  const next = computeNextWatermark(deps.now)
  await deps.advanceWatermark(EVENTS_SOURCE, next)

  return { scanned, inserted, quarantined, watermark: next.toISOString() }
}
