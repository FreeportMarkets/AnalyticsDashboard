import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { fetchEvents, fetchTrades } from '@/lib/ddbFetchers'
import { EVENTS_SOURCE } from '@/lib/sync/syncEvents'
import { TRADES_SOURCE } from '@/lib/sync/syncTrades'
import { WATERMARK_LAG_MS } from '@/lib/sync/watermark'
import { ensurePartitions } from '@/lib/sync/partitions'
import { insertEvents } from '@/lib/sync/insertEvents'
import { insertTrades } from '@/lib/sync/insertTrades'
import { quarantineRow } from '@/lib/sync/quarantine'
import { reconcileEvents } from '@/lib/sync/reconcileEvents'
import { reconcileTrades } from '@/lib/sync/reconcileTrades'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

// Reconciliation must never move the incremental sync's cursor -- see
// reconcileEvents.ts's doc comment. This no-op is wired into both dep
// objects below in place of the real sync_state-writing `advanceWatermark`
// used by /api/cron/sync, so even a bug that called it would do nothing.
async function noopAdvanceWatermark(): Promise<void> {}

export async function GET(request: Request) {
  const secret = process.env.CRON_SECRET
  const auth = request.headers.get('authorization')
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 })
  }

  const now = new Date()

  const events = await reconcileEvents({
    now,
    ensurePartitions: dates => ensurePartitions(sql as never, dates),
    fetchEvents,
    insertEvents: (rows, onRowFailure) => insertEvents(sql, rows, onRowFailure),
    quarantine: (raw, reason) => quarantineRow(sql, EVENTS_SOURCE, raw, reason),
    advanceWatermark: noopAdvanceWatermark,
  })

  const trades = await reconcileTrades({
    now,
    fetchTrades,
    insertTrades: (rows, onRowFailure) => insertTrades(sql, rows, onRowFailure),
    quarantine: (raw, reason) => quarantineRow(sql, TRADES_SOURCE, raw, reason),
    advanceWatermark: noopAdvanceWatermark,
  })

  // The empirical answer to "is WATERMARK_LAG_MS long enough?" -- rows
  // inserted here are exactly the ones the 60s incremental sync missed. If
  // this is persistently nonzero, WATERMARK_LAG_MS should be raised from
  // this data, not from reasoning about flush intervals. Logged prominently
  // so it's visible without querying quarantine or sync_state by hand.
  console.log(
    `[reconcile] events.inserted=${events.inserted} trades.inserted=${trades.inserted} ` +
      `(rows the 60s sync's ${WATERMARK_LAG_MS}ms watermark lag missed; ` +
      `events.scanned=${events.scanned} trades.scanned=${trades.scanned})`
  )

  return NextResponse.json({ events, trades })
}
