import { mapEvent } from './mapEvent'
import { datesToScan } from './watermark'
import type { EventFetcher } from './syncEvents'
import type { EventRow, SyncResult } from './types'

/**
 * A watermark value that is never a valid ISO timestamp, so a caller reading
 * the returned SyncResult can't mistake it for a moved cursor. Reconciliation
 * never reads or writes sync_state -- see the doc comment below.
 */
export const NOT_ADVANCED = 'not-advanced (reconciliation never moves the watermark)'

export interface ReconcileEventsDeps {
  now: Date
  ensurePartitions: (dates: string[]) => Promise<string[]>
  fetchEvents: EventFetcher
  insertEvents: (
    rows: EventRow[],
    onRowFailure?: (row: EventRow, reason: string) => Promise<void>
  ) => Promise<number>
  quarantine: (raw: unknown, reason: string) => Promise<void>
  /**
   * Present only so a test can prove it is never invoked. Reconciliation
   * must not move the incremental sync's cursor in either direction -- see
   * the module doc comment.
   */
  advanceWatermark: (source: string, ts: Date) => Promise<void>
}

/**
 * Nightly reconciliation for events: a full re-read of the last two UTC date
 * partitions that deliberately IGNORES the incremental sync's watermark.
 *
 * The 60s sync (syncEvents) holds its watermark WATERMARK_LAG_MS (10 min)
 * behind wall-clock so late-arriving events aren't stepped over. That lag was
 * sized from the client's 30s flush interval -- but
 * FreeApp/services/AnalyticsService.ts:32 keeps its send queue in memory
 * only, so lateness is actually bounded by a *foreground session*, not the
 * flush interval. A phone offline for 40 minutes delivers 40-minute-late
 * events, which land behind the watermark and are never picked up by the 60s
 * sync -- no error, no quarantine row, no log line, just silently
 * undercounted metrics.
 *
 * This function closes that hole by re-reading each partition from its
 * start (`afterSk` = '', the same full-window read `scripts/backfill.ts`
 * does) rather than from the watermark, and it never reads sync_state to
 * decide what to scan or calls `advanceWatermark` afterward. Moving the
 * cursor here -- in either direction -- would defeat the incremental sync:
 * reading it to bound the scan could re-narrow a window the 60s sync already
 * advanced past, and advancing it here could step the 60s sync's cursor
 * forward without it ever having read that data itself.
 *
 * Idempotent by construction, so running this nightly over an
 * already-synced window is a safe no-op: events insert with ON CONFLICT DO
 * NOTHING and quarantine dedups on (source, item_hash).
 *
 * A throw from `insertEvents` (or any dep) propagates unmodified -- there is
 * no try/catch here to swallow it. A non-data Postgres error (connection,
 * auth, missing table) must fail the run loudly, not be absorbed as "nothing
 * to reconcile."
 *
 * Partitions are ensured from the DISTINCT DATES OF THE MAPPED ROWS actually
 * about to be inserted, NOT from the scan window (`dates`, below) -- same
 * reasoning as syncEvents.ts. The ingest endpoint is unauthenticated, so a
 * row fetched while re-reading "today"/"yesterday" can still carry any `date`
 * mapEvent accepts; ensuring only the scan window would let such a row land
 * in `events_default` and permanently block that month's dedicated partition
 * from ever being created.
 */
export async function reconcileEvents(deps: ReconcileEventsDeps): Promise<SyncResult> {
  const dates = datesToScan(deps.now)

  let scanned = 0
  let inserted = 0
  let quarantined = 0

  for (const date of dates) {
    // '' is a valid exclusive lower bound for the key-condition range query
    // (see ddbFetchers.ts) -- it selects every sk in the partition, the same
    // full re-read scripts/backfill.ts performs.
    const items = await deps.fetchEvents(date, '')
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

  return { scanned, inserted, quarantined, watermark: NOT_ADVANCED }
}
