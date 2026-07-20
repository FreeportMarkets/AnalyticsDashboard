import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { neon } from '@neondatabase/serverless'
import { insertEvents } from '@/lib/sync/insertEvents'
import { insertTrades } from '@/lib/sync/insertTrades'
import { quarantineRow } from '@/lib/sync/quarantine'
import { isDataError } from '@/lib/sync/pgError'
import { ensurePartitions } from '@/lib/sync/partitions'
import type { EventRow } from '@/lib/sync/types'
import type { TradeRow } from '@/lib/sync/mapTrade'

/**
 * These tests run against the REAL Neon test database (not a mock). A green
 * unit suite has twice concealed a broken fix in this project -- insertEvents,
 * insertTrades and quarantineRow have zero coverage that exercises actual SQL,
 * and the bisection fix in particular hinges on how Postgres classifies a real
 * error, which no mock can stand in for.
 *
 * `TEST_DATABASE_URL` is read directly out of `.env.local` (not
 * `process.env`) so this suite works the same way whether it's run via `npx
 * vitest run` locally or in CI, without requiring the runner to export the
 * variable. If it's absent, the whole suite is skipped -- gracefully, not a
 * failure -- so the suite stays green without credentials.
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

// Unique per test-file run so concurrent/rerun invocations never collide on
// the same primary key. Every row written by this suite carries it, and every
// test deletes its own rows in a `finally` so reruns are idempotent -- never
// a TRUNCATE.
const MARKER = `itest_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`

function eventRow(overrides: Partial<EventRow> = {}): EventRow {
  return {
    date: '2026-07-20',
    sk: `${MARKER}#default`,
    ts: new Date('2026-07-20T12:00:00.000Z'),
    event: 'session_start',
    screen: null,
    component: null,
    wallet_address: `${MARKER}-wallet`,
    session_id: null,
    platform: null,
    app_version: null,
    metadata: null,
    ...overrides,
  }
}

function tradeRow(overrides: Partial<TradeRow> = {}): TradeRow {
  return {
    wallet_address: `${MARKER}-wallet`,
    timestamp: '2026-07-20T12:00:00.000Z',
    ts: new Date('2026-07-20T12:00:00.000Z'),
    trade_date: '2026-07-20',
    id: null,
    type: 'perps',
    amount_usd: 100,
    status: 'success',
    source: 'perps',
    client: 'integration-test',
    from_token: null,
    from_mint: null,
    to_token: null,
    to_mint: null,
    amount_from_token: null,
    amount_to_token: null,
    tx_signature: null,
    request_id: null,
    tweet_handle: null,
    tweet_ticker: null,
    tweet_timestamp: null,
    asset: null,
    display_symbol: null,
    side: null,
    size: null,
    price: null,
    leverage: null,
    order_type: null,
    is_close: null,
    is_hip3: null,
    category: null,
    trace_id: null,
    raw: { marker: MARKER },
    ...overrides,
  }
}

describe.skipIf(!TEST_DATABASE_URL)('integration (real Neon test database)', () => {
  // `insertEvents`/`insertTrades` type their `sql` param as `ReturnType<typeof
  // neon>`. Because `neon`'s type params have defaults, TS resolves that
  // `ReturnType<>` using the params' *constraints* (`boolean`), not their
  // defaults (`false`) -- so the alias is `NeonQueryFunction<boolean,
  // boolean>`, not `<false, false>`. A bare `neon(url)` call infers the
  // narrower `<false, false>` and is NOT assignable to the wider alias (the
  // `transaction` method's contravariant parameter makes them structurally
  // incompatible). Match the alias explicitly so this compiles against the
  // real functions under test, not a mock.
  const sql = neon<boolean, boolean>(TEST_DATABASE_URL!)

  describe('insertEvents', () => {
    it('inserts a valid row and returns 1', async () => {
      const row = eventRow({ sk: `${MARKER}#insert-1` })
      try {
        const n = await insertEvents(sql, [row])
        expect(n).toBe(1)
        const rows = (await sql`SELECT sk FROM events WHERE sk = ${row.sk}`) as unknown[]
        expect(rows.length).toBe(1)
      } finally {
        await sql`DELETE FROM events WHERE sk = ${row.sk}`
      }
    })

    it('re-inserting the identical row returns 0 and does not duplicate', async () => {
      const row = eventRow({ sk: `${MARKER}#dup-1` })
      try {
        const first = await insertEvents(sql, [row])
        expect(first).toBe(1)
        const second = await insertEvents(sql, [row])
        expect(second).toBe(0)
        const rows = (await sql`SELECT sk FROM events WHERE sk = ${row.sk}`) as unknown[]
        expect(rows.length).toBe(1)
      } finally {
        await sql`DELETE FROM events WHERE sk = ${row.sk}`
      }
    })

    it(
      'REGRESSION: a chunk with one poisoned row (date 0000-01-01) inserts the ' +
        'valid rows and quarantines only the bad one -- without the bisection ' +
        'fix this whole UNNEST statement rolls back and returns 0',
      async () => {
        const good1 = eventRow({ sk: `${MARKER}#wedge-good-1` })
        const good2 = eventRow({ sk: `${MARKER}#wedge-good-2` })
        const bad = eventRow({ sk: `${MARKER}#wedge-bad-1`, date: '0000-01-01' })
        const failures: Array<{ row: EventRow; reason: string }> = []
        try {
          const n = await insertEvents(sql, [good1, good2, bad], async (row, reason) => {
            failures.push({ row, reason })
          })

          expect(n).toBe(2)
          expect(failures.length).toBe(1)
          expect(failures[0]!.row.sk).toBe(bad.sk)

          const rows = await sql`
            SELECT sk FROM events
            WHERE sk = ${good1.sk} OR sk = ${good2.sk} OR sk = ${bad.sk}
          `
          const sks = (rows as Array<{ sk: string }>).map(r => r.sk).sort()
          expect(sks).toEqual([good1.sk, good2.sk].sort())
        } finally {
          await sql`
            DELETE FROM events WHERE sk = ${good1.sk} OR sk = ${good2.sk} OR sk = ${bad.sk}
          `
        }
      }
    )
  })

  describe('insertTrades', () => {
    it('upserts: a second insert with a changed amount_usd UPDATES the existing row', async () => {
      const row = tradeRow({ timestamp: `${MARKER}-ts-1` })
      try {
        const first = await insertTrades(sql, [row])
        expect(first).toBe(1)

        const changed = { ...row, amount_usd: 999.5 }
        const second = await insertTrades(sql, [changed])
        expect(second).toBe(1)

        const rows = (await sql`
          SELECT amount_usd FROM trades
          WHERE wallet_address = ${row.wallet_address} AND timestamp = ${row.timestamp}
        `) as Array<{ amount_usd: string }>
        expect(rows.length).toBe(1)
        expect(Number(rows[0]!.amount_usd)).toBe(999.5)
      } finally {
        await sql`
          DELETE FROM trades WHERE wallet_address = ${row.wallet_address} AND timestamp = ${row.timestamp}
        `
      }
    })
  })

  describe('quarantineRow', () => {
    it('writes a row, and calling it twice with identical input leaves exactly one row', async () => {
      const source = `${MARKER}-quarantine-source`
      const raw = { marker: MARKER, note: 'duplicate quarantine attempt' }
      const reason = 'test reason for dedup'
      try {
        await quarantineRow(sql, source, raw, reason)
        await quarantineRow(sql, source, raw, reason)
        const rows = (await sql`SELECT id FROM quarantine WHERE source = ${source}`) as unknown[]
        expect(rows.length).toBe(1)
      } finally {
        await sql`DELETE FROM quarantine WHERE source = ${source}`
      }
    })
  })

  describe('partition creation from row dates', () => {
    it(
      'REGRESSION (Task 11): a row dated in a month with no existing partition gets ' +
        'its own monthly partition created before insert, and never lands in ' +
        'events_default -- proving ensurePartitions must be called with the mapped ' +
        "ROW's date, not the scan window, or Postgres permanently refuses to ever " +
        "create that month's partition once a row for it sits in the DEFAULT partition",
      async () => {
        // Year 2088 is a year no other test file uses (integration.test.ts's own
        // fixtures above use 2026; reconcile.test.ts uses 2031). Vitest runs test
        // files concurrently, and creating a partition for a month races with any
        // concurrent raw insert() into that SAME month -- Task 11 hit this for
        // real: a concurrent insert lands in events_default first, and Postgres
        // then rejects "CREATE TABLE ... PARTITION OF events" for that month with
        // "updated partition constraint ... would be violated". A dedicated,
        // otherwise-untouched year sidesteps that race entirely.
        const date = '2088-05-15'
        const partitionName = 'events_2088_05'
        const sk = `${MARKER}#partition-regression`
        const row = eventRow({ sk, date, ts: new Date(`${date}T12:00:00.000Z`) })

        try {
          // Mirrors the real 60s sync's contract post-fix: partitions ensured
          // from the row's own date before it is inserted.
          await ensurePartitions(sql as never, [date])
          const n = await insertEvents(sql, [row])
          expect(n).toBe(1)

          // Prove which partition the row actually landed in -- not merely that
          // it exists. `tableoid::regclass` resolves the physical child table an
          // inherited/partitioned row is stored in.
          const located = (await sql`
            SELECT tableoid::regclass::text AS partition FROM events WHERE sk = ${sk}
          `) as Array<{ partition: string }>
          expect(located.length).toBe(1)
          expect(located[0]!.partition).toBe(partitionName)

          // Cross-check via pg_inherits/pg_class that the partition is a real
          // child of `events`, not a same-named coincidence.
          const inherited = (await sql`
            SELECT c.relname
            FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'events' AND c.relname = ${partitionName}
          `) as unknown[]
          expect(inherited.length).toBe(1)

          // The whole point of the fix: this row must NOT be sitting in the
          // DEFAULT partition.
          const defaultRows = (await sql`
            SELECT sk FROM events_default WHERE sk = ${sk}
          `) as unknown[]
          expect(defaultRows.length).toBe(0)
        } finally {
          await sql`DELETE FROM events WHERE sk = ${sk}`
          // DDL cannot be parameterized; `partitionName` is the hardcoded
          // literal above, not user input. Dropping the partition (rather than
          // leaving it around) keeps reruns idempotent without ever truncating
          // `events` itself.
          await sql(`DROP TABLE IF EXISTS ${partitionName}`)
        }
      }
    )
  })

  describe('non-data errors', () => {
    it('a query against a table that does not exist throws, and is not classified as a data error', async () => {
      let caught: unknown
      try {
        await sql(`SELECT * FROM ${MARKER}_table_does_not_exist`)
      } catch (err) {
        caught = err
      }
      expect(caught).toBeDefined()
      // This is the exact discriminator insertChunk uses to decide whether to
      // bisect-and-quarantine or rethrow. An "undefined table" error is
      // SQLSTATE 42P01 (class 42, syntax_error_or_access_rule_violation) --
      // NOT class 22/23 -- so it must be treated as fail-safe and rethrown,
      // never silently quarantined as if it were a bad row.
      expect(isDataError(caught)).toBe(false)
    })
  })
})
