import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { fetchEvents, fetchTrades } from '@/lib/ddbFetchers'
import { syncEvents, EVENTS_SOURCE } from '@/lib/sync/syncEvents'
import { syncTrades, TRADES_SOURCE } from '@/lib/sync/syncTrades'
import { readWatermark, advanceWatermark } from '@/lib/sync/watermark'
import { ensurePartitions } from '@/lib/sync/partitions'
import { insertEvents } from '@/lib/sync/insertEvents'
import { insertTrades } from '@/lib/sync/insertTrades'
import { quarantineRow } from '@/lib/sync/quarantine'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

// Only used the very first time a source runs, before sync_state has a row.
const COLD_START = new Date('2026-07-01T00:00:00.000Z')

export async function GET(request: Request) {
  const secret = process.env.CRON_SECRET
  const auth = request.headers.get('authorization')
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 })
  }

  const now = new Date()

  const events = await syncEvents({
    now,
    readWatermark: () => readWatermark(sql, EVENTS_SOURCE, COLD_START),
    advanceWatermark: (source, ts) => advanceWatermark(sql, source, ts),
    ensurePartitions: dates => ensurePartitions(sql as never, dates),
    fetchEvents,
    insertEvents: (rows, onRowFailure) => insertEvents(sql, rows, onRowFailure),
    quarantine: (raw, reason) => quarantineRow(sql, EVENTS_SOURCE, raw, reason),
  })

  const trades = await syncTrades({
    now,
    readWatermark: () => readWatermark(sql, TRADES_SOURCE, COLD_START),
    advanceWatermark: (source, ts) => advanceWatermark(sql, source, ts),
    fetchTrades,
    insertTrades: (rows, onRowFailure) => insertTrades(sql, rows, onRowFailure),
    quarantine: (raw, reason) => quarantineRow(sql, TRADES_SOURCE, raw, reason),
  })

  return NextResponse.json({ events, trades })
}
