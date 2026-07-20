import { describe, it, expect, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { neon } from '@neondatabase/serverless'
import { reconcileEvents } from '@/lib/sync/reconcileEvents'
import { reconcileTrades } from '@/lib/sync/reconcileTrades'
import { insertEvents } from '@/lib/sync/insertEvents'
import { ensurePartitions } from '@/lib/sync/partitions'
import { quarantineRow } from '@/lib/sync/quarantine'
import type { EventRow } from '@/lib/sync/types'

function eventItem(overrides: Record<string, unknown> = {}) {
  return {
    date: '2026-07-20',
    sk: '2026-07-20T12:00:00.000Z#0xabc#aaaa1111',
    event: 'session_start',
    wallet_address: '0xabc',
    session_id: 's1',
    timestamp: '2026-07-20T12:00:00.000Z',
    ...overrides,
  }
}

const tradeItem = {
  wallet_address: '0xabc',
  timestamp: '2026-07-20T12:00:00.000Z',
  trade_date: '2026-07-20',
  type: 'perps',
  amount_usd: 610,
  status: 'success',
}

function eventDeps(items: unknown[]) {
  const inserted: unknown[][] = []
  const quarantined: Array<{ raw: unknown; reason: string }> = []
  return {
    inserted,
    quarantined,
    args: {
      now: new Date('2026-07-20T14:00:00.000Z'),
      ensurePartitions: vi.fn(async () => []),
      fetchEvents: vi.fn(async () => items),
      insertEvents: vi.fn(async (rows: unknown[]) => {
        inserted.push(rows)
        return rows.length
      }),
      quarantine: vi.fn(async (raw: unknown, reason: string) => {
        quarantined.push({ raw, reason })
      }),
      advanceWatermark: vi.fn(async () => {}),
    },
  }
}

function tradeDeps(items: unknown[]) {
  return {
    args: {
      now: new Date('2026-07-20T14:00:00.000Z'),
      fetchTrades: vi.fn(async () => items),
      insertTrades: vi.fn(async (rows: unknown[]) => rows.length),
      quarantine: vi.fn(async () => {}),
      advanceWatermark: vi.fn(async () => {}),
    },
  }
}

describe('reconcileEvents', () => {
  it(
    '(a) inserts a row whose timestamp predates the current watermark -- ' +
      'this is the whole point of reconciliation',
    async () => {
      // A watermark of 13:50 (now - WATERMARK_LAG_MS) would have already
      // excluded this row from the 60s sync's next fetch window had that
      // sync's afterSk bound been used here. reconcileEvents is passed no
      // watermark at all -- fetchEvents below is called with '' regardless
      // of how "late" this row's timestamp is, and the row is still mapped
      // and inserted.
      const lateItem = eventItem({ timestamp: '2026-07-20T13:00:00.000Z' })
      const d = eventDeps([lateItem])
      const r = await reconcileEvents(d.args as never)

      expect(r.inserted).toBe(2) // fetched for both scanned dates (prev + current)
      expect(d.args.fetchEvents).toHaveBeenCalledWith('2026-07-19', '')
      expect(d.args.fetchEvents).toHaveBeenCalledWith('2026-07-20', '')
    }
  )

  it('(b) never calls advanceWatermark', async () => {
    const d = eventDeps([eventItem()])
    await reconcileEvents(d.args as never)
    expect(d.args.advanceWatermark).not.toHaveBeenCalled()
  })

  it('(c) a non-data error from insertEvents propagates rather than being swallowed', async () => {
    const d = eventDeps([eventItem()])
    d.args.insertEvents = vi.fn(async () => {
      throw new Error('neon down')
    })
    await expect(reconcileEvents(d.args as never)).rejects.toThrow('neon down')
    expect(d.args.advanceWatermark).not.toHaveBeenCalled()
  })

  it('quarantines unmappable rows without failing the batch', async () => {
    const d = eventDeps([eventItem(), { garbage: true }])
    const r = await reconcileEvents(d.args as never)
    expect(r.quarantined).toBe(2)
    expect(r.inserted).toBe(2)
  })

  it('ensures partitions before inserting', async () => {
    const d = eventDeps([eventItem()])
    await reconcileEvents(d.args as never)
    expect(d.args.ensurePartitions).toHaveBeenCalledWith(['2026-07-19', '2026-07-20'])
  })

  it('reports a non-timestamp watermark value so callers cannot mistake it for a moved cursor', async () => {
    const d = eventDeps([])
    const r = await reconcileEvents(d.args as never)
    expect(() => {
      const t = new Date(r.watermark).getTime()
      if (!Number.isNaN(t)) throw new Error('watermark parsed as a real timestamp')
    }).not.toThrow()
  })
})

describe('reconcileTrades', () => {
  it('(a) inserts a trade regardless of its timestamp relative to any watermark', async () => {
    const d = tradeDeps([tradeItem])
    const r = await reconcileTrades(d.args as never)
    expect(r.inserted).toBe(2)
    expect(d.args.fetchTrades).toHaveBeenCalledWith('2026-07-19', '')
    expect(d.args.fetchTrades).toHaveBeenCalledWith('2026-07-20', '')
  })

  it('(b) never calls advanceWatermark', async () => {
    const d = tradeDeps([tradeItem])
    await reconcileTrades(d.args as never)
    expect(d.args.advanceWatermark).not.toHaveBeenCalled()
  })

  it('(c) a non-data error from insertTrades propagates rather than being swallowed', async () => {
    const d = tradeDeps([tradeItem])
    d.args.insertTrades = vi.fn(async () => {
      throw new Error('neon down')
    })
    await expect(reconcileTrades(d.args as never)).rejects.toThrow('neon down')
    expect(d.args.advanceWatermark).not.toHaveBeenCalled()
  })

  it('quarantines unmappable rows without failing the batch', async () => {
    const d = tradeDeps([tradeItem, { nope: 1 }])
    const r = await reconcileTrades(d.args as never)
    expect(r.quarantined).toBe(2)
    expect(r.inserted).toBe(2)
  })
})

/**
 * Integration test against the REAL Neon test database, following the exact
 * pattern in test/integration.test.ts: `TEST_DATABASE_URL` is read directly
 * out of `.env.local` (not `process.env`) so the suite skips gracefully via
 * `describe.skipIf` rather than failing when credentials are absent, and
 * cleanup happens via a per-run marker rather than a TRUNCATE.
 */
function readEnvLocal(key: string): string | undefined {
  let content: string
  try {
    content = readFileSync(path.join(import.meta.dirname, '..', '.env.local'), 'utf8')
  } catch {
    return undefined
  }
  for (const line of content.split('\n')) {
    const match = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/)
    if (match && match[1] === key) {
      return match[2]!.trim().replace(/^"(.*)"$/, '$1')
    }
  }
  return undefined
}

const TEST_DATABASE_URL = readEnvLocal('TEST_DATABASE_URL')
const MARKER = `reconcile_itest_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`

describe.skipIf(!TEST_DATABASE_URL)('reconcileEvents (real Neon test database)', () => {
  const sql = neon<boolean, boolean>(TEST_DATABASE_URL!)

  it(
    're-reading an already-synced window is a no-op: row count is unchanged and inserted is 0',
    async () => {
      // A date far outside any other test file's fixture range (integration.test.ts
      // fixes on '2026-07-20'). Vitest runs test files concurrently, and this test
      // is the only one that calls ensurePartitions against the real database --
      // a concurrent raw insert() (skipping ensurePartitions, as integration.test.ts's
      // do) into the SAME date would land in the events_default catch-all and make
      // Postgres reject creating a dedicated partition that overlaps it ("updated
      // partition constraint ... would be violated"). A dedicated date sidesteps
      // that race entirely rather than relying on test ordering.
      const now = new Date('2031-03-15T14:00:00.000Z')
      const sk = `${MARKER}#already-synced`
      const row: EventRow = {
        date: '2031-03-15',
        sk,
        ts: new Date('2031-03-15T12:00:00.000Z'),
        event: 'session_start',
        screen: null,
        component: null,
        wallet_address: `${MARKER}-wallet`,
        session_id: null,
        platform: null,
        app_version: null,
        metadata: null,
      }

      try {
        // The real 60s sync always ensures partitions before inserting
        // (syncEvents.ts), so the partition for this row's month already
        // exists by the time it lands. Mirror that here -- otherwise this
        // row would land in the events_default catch-all partition, and
        // reconcileEvents's own ensurePartitions call below would then fail
        // ("updated partition constraint ... would be violated") trying to
        // carve out a dedicated partition that overlaps an existing
        // default-partition row.
        await ensurePartitions(sql as never, [row.date])

        // Simulate the 60s sync already having synced this row.
        const first = await insertEvents(sql, [row])
        expect(first).toBe(1)

        // Reconciliation re-reads the same DynamoDB item (same date/sk/timestamp)
        // that produced `row`, exactly as fetchEvents would return it.
        const raw = {
          date: row.date,
          sk: row.sk,
          event: row.event,
          wallet_address: row.wallet_address,
          timestamp: row.ts.toISOString(),
        }

        const result = await reconcileEvents({
          now,
          ensurePartitions: dates => ensurePartitions(sql as never, dates),
          fetchEvents: async date => (date === row.date ? [raw] : []),
          insertEvents: (rows, onRowFailure) => insertEvents(sql, rows, onRowFailure),
          quarantine: (r, reason) => quarantineRow(sql, 'reconcile-test-events', r, reason),
          advanceWatermark: async () => {
            throw new Error('advanceWatermark must never be called by reconciliation')
          },
        })

        // ON CONFLICT DO NOTHING: the row already exists, so reconciliation
        // inserts zero new rows.
        expect(result.inserted).toBe(0)

        const rows = (await sql`SELECT sk FROM events WHERE sk = ${sk}`) as unknown[]
        expect(rows.length).toBe(1)
      } finally {
        await sql`DELETE FROM events WHERE sk = ${sk}`
        await sql`DELETE FROM quarantine WHERE source = 'reconcile-test-events'`
      }
    }
  )
})
