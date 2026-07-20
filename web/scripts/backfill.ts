/**
 * Historical backfill: loads the ~1.5M-row DynamoDB history into the mirrored
 * Postgres tables (`events`, `trades`).
 *
 * Resumable: progress is recorded in `sync_state` after EVERY date completes
 * (not batched at the end), under a `backfill:<source>` key -- e.g.
 * `backfill:events`. A crash or Ctrl-C 25 minutes into a 30-minute run resumes
 * at the next unfinished date, not from scratch.
 *
 * MUST NOT disturb the live watermark: `backfill:events`/`backfill:trades` are
 * entirely disjoint sync_state rows from the `events`/`trades` keys the 60s
 * incremental cron owns (src/app/api/cron/sync/route.ts, EVENTS_SOURCE /
 * TRADES_SOURCE). This script never reads or writes those keys.
 *
 * Idempotent: re-running any range, or the same day twice, must not change the
 * resulting data. Events insert ON CONFLICT DO NOTHING (append-only, per
 * insertEvents.ts); trades insert ON CONFLICT DO UPDATE (both DynamoDB writers
 * use PutCommand, which overwrites in place on a repeated key, so a later read
 * of the same key is the current truth -- see insertTrades.ts).
 *
 * Usage:
 *   npx tsx scripts/backfill.ts events [start YYYY-MM-DD] [end YYYY-MM-DD]
 *   npx tsx scripts/backfill.ts trades [start YYYY-MM-DD] [end YYYY-MM-DD]
 *
 * start/end default to DEFAULT_START (see below) and today (UTC). A crash-safe
 * re-run just omits both and lets the resume cursor pick up where it left off.
 */
import { sql } from '../src/lib/db'
import { fetchEvents, fetchTrades, FULL_READ_FLOOR } from '../src/lib/ddbFetchers'
import { mapEvent } from '../src/lib/sync/mapEvent'
import { mapTrade, type TradeRow } from '../src/lib/sync/mapTrade'
import { insertEvents } from '../src/lib/sync/insertEvents'
import { insertTrades } from '../src/lib/sync/insertTrades'
import { ensurePartitions } from '../src/lib/sync/partitions'
import { quarantineRow } from '../src/lib/sync/quarantine'
import type { EventRow } from '../src/lib/sync/types'

/**
 * Measured against the real `freeport-analytics-events` table on 2026-07-20
 * (not assumed, and not copied from any spec): analytics events begin around
 * 2026-02-15 -- there is nothing before it. A date-partitioned Query against an
 * earlier date returns an empty item list, not an error, so scanning further
 * back than this is harmless but wasted work. The default is padded two weeks
 * earlier than the measured floor purely as a safety margin, not because data
 * is known to exist there.
 */
const DEFAULT_START = '2026-02-01'

type Source = 'events' | 'trades'

interface DateResult {
  scanned: number
  inserted: number
  quarantined: number
}

function backfillSourceKey(source: Source): string {
  return `backfill:${source}`
}

function eachDate(start: string, end: string): string[] {
  const out: string[] = []
  const cur = new Date(`${start}T00:00:00Z`)
  const stop = new Date(`${end}T00:00:00Z`)
  while (cur <= stop) {
    out.push(cur.toISOString().slice(0, 10))
    cur.setUTCDate(cur.getUTCDate() + 1)
  }
  return out
}

/**
 * Record `date` as the last date fully processed for `source`. Reuses
 * sync_state's (source, watermark_ts) shape -- but under the disjoint
 * `backfill:<source>` key, never `events`/`trades` -- so this script's cursor
 * can never be confused with, or clobber, the live incremental sync's
 * watermark. GREATEST guards a concurrent or replayed run from moving the
 * cursor backwards, same as the incremental sync's advanceWatermark
 * (watermark.ts).
 */
async function markDateDone(source: Source, date: string): Promise<void> {
  await sql`
    INSERT INTO sync_state (source, watermark_ts, updated_at)
    VALUES (${backfillSourceKey(source)}, ${`${date}T00:00:00.000Z`}, now())
    ON CONFLICT (source) DO UPDATE
      SET watermark_ts = GREATEST(sync_state.watermark_ts, EXCLUDED.watermark_ts),
          updated_at = now()
  `
}

async function lastDoneDate(source: Source): Promise<string | null> {
  const rows = (await sql`
    SELECT watermark_ts FROM sync_state WHERE source = ${backfillSourceKey(source)}
  `) as Array<{ watermark_ts: string | Date }>
  if (rows.length === 0) return null
  return new Date(rows[0]!.watermark_ts).toISOString().slice(0, 10)
}

/**
 * Backfill one UTC date partition of events. `FULL_READ_FLOOR` (imported
 * from ddbFetchers.ts -- see its doc comment for why `''` is not a valid
 * substitute) is a valid exclusive lower bound for the key-condition range
 * query, so this is a full re-read of the date.
 */
async function backfillEventsDate(date: string): Promise<DateResult> {
  let quarantined = 0
  const items = await fetchEvents(date, FULL_READ_FLOOR)

  const rows: EventRow[] = []
  for (const item of items) {
    const mapped = mapEvent(item)
    if (mapped.ok) rows.push(mapped.value)
    else {
      await quarantineRow(sql, 'events', item, mapped.reason)
      quarantined += 1
    }
  }

  let inserted = 0
  if (rows.length > 0) {
    // Partitions are ensured from the DISTINCT DATES OF THE MAPPED ROWS, not
    // from `date` (the scan window). The ingest endpoint that writes this
    // table is unauthenticated, so a fetched row's `date` field is not
    // guaranteed to match the DynamoDB partition it came from. Ensuring only
    // the scan date would let such a row land in `events_default`, and once a
    // row for a given month sits there, Postgres refuses to ever create that
    // month's dedicated partition ("updated partition constraint ... would be
    // violated") -- a permanent wedge from a single row. See syncEvents.ts's
    // doc comment for the full rationale; this mirrors it exactly.
    const rowDates = [...new Set(rows.map(r => r.date))]
    await ensurePartitions(sql as never, rowDates)

    inserted = await insertEvents(sql, rows, async (row, reason) => {
      // Quarantine the RAW item, not the normalized row -- `mapEvent`
      // deliberately nulls type-confused optional fields, and the raw item is
      // what's needed to actually debug a bisection-time failure. See
      // EventRow.raw's doc comment and syncEvents.ts's identical handling.
      await quarantineRow(sql, 'events', row.raw, reason)
      quarantined += 1
    })
  }

  return { scanned: items.length, inserted, quarantined }
}

/** Backfill one trade_date partition of trades. Full re-read, same reasoning as backfillEventsDate. */
async function backfillTradesDate(date: string): Promise<DateResult> {
  let quarantined = 0
  const items = await fetchTrades(date, FULL_READ_FLOOR)

  const rows: TradeRow[] = []
  for (const item of items) {
    const mapped = mapTrade(item)
    if (mapped.ok) rows.push(mapped.value)
    else {
      await quarantineRow(sql, 'trades', item, mapped.reason)
      quarantined += 1
    }
  }

  let inserted = 0
  if (rows.length > 0) {
    // trades has no partitions to ensure (see db/migrations/0001_init.sql --
    // unlike events it is not PARTITION BY RANGE).
    inserted = await insertTrades(sql, rows, async (row, reason) => {
      // Matches reconcileTrades.ts/syncTrades.ts: quarantines the mapped
      // TradeRow (which itself carries the original item under `.raw`), not
      // the bare raw item.
      await quarantineRow(sql, 'trades', row, reason)
      quarantined += 1
    })
  }

  return { scanned: items.length, inserted, quarantined }
}

async function main() {
  const [sourceArg, startArg, endArg] = process.argv.slice(2)
  if (sourceArg !== 'events' && sourceArg !== 'trades') {
    throw new Error('usage: backfill.ts <events|trades> [start YYYY-MM-DD] [end YYYY-MM-DD]')
  }
  const source: Source = sourceArg
  const start = startArg ?? DEFAULT_START
  const end = endArg ?? new Date().toISOString().slice(0, 10)

  const resume = await lastDoneDate(source)
  const dates = eachDate(start, end).filter(d => !resume || d > resume)

  if (resume) {
    console.log(`resuming ${source} backfill after ${resume}; ${dates.length} date(s) remain`)
  } else {
    console.log(`starting ${source} backfill over ${dates.length} date(s) [${start}..${end}]`)
  }

  let totalScanned = 0
  let totalInserted = 0
  let totalQuarantined = 0

  for (const date of dates) {
    const result = source === 'events' ? await backfillEventsDate(date) : await backfillTradesDate(date)

    totalScanned += result.scanned
    totalInserted += result.inserted
    totalQuarantined += result.quarantined

    console.log(
      `${date}  scanned=${result.scanned}  inserted=${result.inserted}  quarantined=${result.quarantined}`
    )

    // Recorded after EVERY date, not batched at the end -- this is what makes
    // a crash mid-run resume at the next unfinished date instead of restarting.
    await markDateDone(source, date)
  }

  console.log(
    `backfill complete: source=${source} dates=${dates.length} ` +
      `scanned=${totalScanned} inserted=${totalInserted} quarantined=${totalQuarantined}`
  )
}

main().catch(e => {
  console.error(e)
  process.exit(1)
})
