import { describe, it, expect, afterAll } from 'vitest'
import { readFileSync } from 'node:fs'
import { neon } from '@neondatabase/serverless'
import { ensurePartitions } from '@/lib/sync/partitions'

const env = Object.fromEntries(
  (() => { try { return readFileSync('.env.local', 'utf8').split('\n') } catch { return [] } })()
    .filter(l => l.includes('='))
    .map(l => { const i = l.indexOf('='); return [l.slice(0, i), l.slice(i + 1).replace(/^"|"$/g, '')] })
) as Record<string, string>

const url = env.TEST_DATABASE_URL
const MARK = 'tzproof_'
// Year 2077 is unused by other test files (2026, 2027, 2031, 2088 are taken).
const sql = url ? neon<boolean, boolean>(url) : null

// `src/lib/metrics/queries.ts` imports `sql` from `@/lib/db`, which reads
// `process.env.DATABASE_URL` at module load time (not `TEST_DATABASE_URL`).
// Point it at the test database BEFORE that module is ever evaluated, and
// import it dynamically -- a static `import` would hoist above this
// assignment and `@/lib/db` would throw "DATABASE_URL is not set" during
// collection, even when this suite is about to be skipped.
if (url) process.env.DATABASE_URL = url
const { dailyEventCounts } = url
  ? await import('@/lib/metrics/queries')
  : { dailyEventCounts: undefined as unknown as typeof import('@/lib/metrics/queries')['dailyEventCounts'] }

describe.skipIf(!url)('metric timezone bucketing (real DB)', () => {
  afterAll(async () => {
    if (!sql) return
    await sql('DELETE FROM events WHERE wallet_address LIKE $1', [`${MARK}%`])
  })

  it('buckets by New York day, NOT by the UTC date partition key', async () => {
    if (!sql) return

    // Ensure the 2077-03 partition exists before inserting -- a raw insert
    // into a month with no partition silently falls into events_default,
    // which is itself an alert condition per db/migrations/0001_init.sql.
    await ensurePartitions(sql as never, ['2077-03-16'])

    // 2077-03-16T02:30:00Z is 21:30 on 2077-03-15 in New York.
    // The DynamoDB-style `date` column says 2077-03-16. A correct metric says 03-15.
    await sql(
      `INSERT INTO events (date, sk, ts, event, wallet_address, session_id)
       VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING`,
      ['2077-03-16', `${MARK}a`, '2077-03-16T02:30:00Z', 'session_start', `${MARK}w1`, 's1']
    )
    // A second event squarely inside NY 03-16 as a control.
    await sql(
      `INSERT INTO events (date, sk, ts, event, wallet_address, session_id)
       VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING`,
      ['2077-03-16', `${MARK}b`, '2077-03-16T18:00:00Z', 'session_start', `${MARK}w2`, 's2']
    )

    const rows = await dailyEventCounts('2077-03-15', '2077-03-16')
    const byDay = Object.fromEntries(rows.map(r => [r.day, r.count]))

    // THE ASSERTION. If a query ever groups by `date`, both rows land on
    // 2077-03-16 and this fails.
    expect(byDay['2077-03-15']).toBe(1)
    expect(byDay['2077-03-16']).toBe(1)
  })
})
