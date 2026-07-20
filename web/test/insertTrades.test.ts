import { describe, it, expect } from 'vitest'
import { insertTrades } from '@/lib/sync/insertTrades'
import type { TradeRow } from '@/lib/sync/mapTrade'

/**
 * Unit-level coverage for insertTrades's chunk-internal dedup, using a mock
 * `sql` tag rather than the real database (the real-DB regression test lives
 * in test/integration.test.ts, per the project's pattern of pairing a fast
 * mocked test with a real-DB proof for anything whose correctness depends on
 * actual Postgres behavior).
 *
 * `parseRowCount` counts rows in the generated `VALUES (...),(...)` clause
 * generically (no hardcoded column count) so this mock stays correct if the
 * COLUMNS list in insertTrades.ts ever changes.
 */
function parseRowCount(text: string): number {
  const match = text.match(/VALUES ([\s\S]*?) ON CONFLICT/)
  if (!match) throw new Error(`unexpected query shape: ${text}`)
  return (match[1]!.match(/\),\(/g)?.length ?? 0) + 1
}

function mockSql() {
  const calls: Array<{ text: string; params: unknown[] }> = []
  const sql = (async (text: string, params: unknown[]) => {
    calls.push({ text, params })
    return { rowCount: parseRowCount(text), rows: [] }
  }) as unknown as Parameters<typeof insertTrades>[0]
  return { sql, calls }
}

function row(overrides: Partial<TradeRow> = {}): TradeRow {
  return {
    wallet_address: '0xabc',
    timestamp: '2026-07-20T12:00:00.000Z',
    ts: new Date('2026-07-20T12:00:00.000Z'),
    trade_date: '2026-07-20',
    id: null,
    type: 'perps',
    amount_usd: 100,
    status: 'success',
    source: 'perps',
    client: 'unit-test',
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
    raw: {},
    ...overrides,
  }
}

describe('insertTrades dedup', () => {
  it(
    'two rows sharing (wallet_address, timestamp) insert as one row carrying ' +
      "the LAST row's values, and does not throw",
    async () => {
      const { sql, calls } = mockSql()
      const first = row({ amount_usd: 111 })
      const second = row({ amount_usd: 222 })

      const n = await insertTrades(sql, [first, second])

      expect(n).toBe(1)
      expect(calls.length).toBe(1)
      // amount_usd is the 7th named column: wallet_address, timestamp, ts,
      // trade_date, id, type, amount_usd -- index 6.
      expect(calls[0]!.params[6]).toBe(222)
    }
  )

  it('a duplicate pair split across the 500-row chunk boundary still dedupes to one row', async () => {
    const { sql, calls } = mockSql()
    const rows: TradeRow[] = []
    for (let i = 0; i < 499; i++) {
      rows.push(row({ timestamp: `2026-07-20T00:00:${String(i % 60).padStart(2, '0')}.000Z-${i}` }))
    }
    // Duplicate pair straddling the boundary: index 499 lands in chunk 1
    // (rows.slice(0, 500)), index 500 would land in chunk 2
    // (rows.slice(500, 1000)) if not deduped globally first.
    rows.push(row({ timestamp: 'dup-ts', amount_usd: 1 })) // index 499
    rows.push(row({ timestamp: 'dup-ts', amount_usd: 2 })) // index 500

    const n = await insertTrades(sql, rows)

    // 501 unique keys (499 distinct + 1 deduped pair) after the global
    // pre-chunk dedup -- so only one chunk-worth of work, not two chunks
    // each independently believing their single occurrence is unique.
    expect(n).toBe(500)
    expect(calls.length).toBe(1)
  })
})
