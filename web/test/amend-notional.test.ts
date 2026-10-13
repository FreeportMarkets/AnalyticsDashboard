import { afterAll, describe, expect, it, vi } from 'vitest'
import { PGlite } from '@electric-sql/pglite'
import { mapTrade } from '@/lib/sync/mapTrade'
vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
import { VOLUME_USD_EXPR } from '@/lib/metrics/trades'

const db = new PGlite()
afterAll(async () => { await db.close() })
async function volume(overrides: Record<string, unknown>) {
  const mapped = mapTrade({ wallet_address: '0xabc', timestamp: '2026-10-13T00:00:00Z',
    type: 'perps', category: 'hyperliquid', size: 0.003, price: 63999,
    amount_basis: 'notional', notional_usd: 191.997, ...overrides })
  if (!mapped.ok) throw Error(mapped.reason)
  const row = mapped.value
  const result = await db.query<{ volume: string | null }>(`SELECT ${VOLUME_USD_EXPR} AS volume FROM
    (SELECT $1::text AS type, $2::text AS category, $3::boolean AS is_close,
     $4::numeric AS size, $5::numeric AS price, $6::numeric AS amount_usd,
     $7::numeric AS leverage, $8::jsonb AS raw) AS trades`,
    [row.type, row.category, row.is_close, row.size, row.price, row.amount_usd, row.leverage, JSON.stringify(row.raw)])
  const measured = result.rows[0]
  if (!measured) throw Error('Missing volume row')
  return { row, volume: measured.volume }
}
describe('amend receipts retain unknown margin through mapping and actual volume SQL', () => {
  it.each([false, true])('counts confirmed notional with close=%s without inventing margin', async (is_close) => {
    const result = await volume({ is_close, notional_usd: 150 })
    expect(result.row.amount_usd).toBeNull()
    expect(result.row.leverage).toBeNull()
    expect(Number(result.volume)).toBe(150)
  })
  it('keeps legacy margin/leverage volume unchanged', async () => {
    const result = await volume({ amount_basis: undefined, notional_usd: undefined,
      amount_usd: 20, leverage: 10, is_close: false })
    expect(Number(result.volume)).toBe(200)
  })
  it('does not invent zero or use legacy margin for malformed explicit notional', async () => {
    const result = await volume({ notional_usd: 'bad', amount_usd: 100, leverage: 10 })
    expect(result.volume).toBeNull()
  })
})
