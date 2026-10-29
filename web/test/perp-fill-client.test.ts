import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { PGlite } from '@electric-sql/pglite'
vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
vi.mock('@/lib/hlLedgerTradesApi', async importOriginal => ({
  ...await importOriginal<typeof import('@/lib/hlLedgerTradesApi')>(), fetchHlLedgerTrades: vi.fn(),
}))
import { sql } from '@/lib/db'
import { fetchHlLedgerTrades, type HlLedgerTrade } from '@/lib/hlLedgerTradesApi'
import { FILL_LOG_WINDOW_SECONDS, perpFillClients, recentTrades } from '@/lib/metrics/trades'

// Synthetic wallets only.
const WALLET_A = '0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
const WALLET_B = '0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
const T0 = '2026-10-05T17:52:52.644Z'
const at = (ms: number) => new Date(Date.parse(T0) + ms).toISOString()

const db = new PGlite()
beforeAll(async () => {
  // The real `trades` DDL from the init migration, so the query runs against the actual schema.
  const init = readFileSync(path.resolve(__dirname, '../db/migrations/0001_init.sql'), 'utf8')
  const ddl = init.match(/CREATE TABLE IF NOT EXISTS trades \([\s\S]*?\n\);/)
  if (!ddl) throw Error('trades DDL not found')
  await db.exec(ddl[0])
})
afterAll(async () => { await db.close() })
beforeEach(async () => {
  await db.exec('TRUNCATE trades')
  vi.mocked(sql).mockReset()
  vi.mocked(sql).mockImplementation((async (query: string, params: unknown[]) =>
    (await db.query(query, params)).rows) as unknown as typeof sql)
  vi.mocked(fetchHlLedgerTrades).mockReset()
})

async function log(wallet: string, ts: string, client: string | null, asset = 'HOOD', extra: Record<string, unknown> = {}) {
  await db.query(`INSERT INTO trades (wallet_address, timestamp, ts, type, asset, category, client, raw)
    VALUES ($1, $2::text, $2::text::timestamptz, $3, $4, $5, $6, '{}'::jsonb)`,
  [wallet, ts, extra.type ?? 'perps', asset, extra.category ?? 'hyperliquid', client])
}
const fill = (id: string, patch: Partial<HlLedgerTrade> = {}): HlLedgerTrade => ({
  fillId: id, orderId: id, walletAddress: WALLET_A, market: 'xyz:HOOD', side: 'B', direction: 'Open Long',
  size: '1', price: '100', notionalUsd: '100', filledAt: T0, ...patch,
})

describe('perp fill client attribution', () => {
  it('takes the client of the trade-log row written just after the fill', async () => {
    await log(WALLET_A, at(400), 'web')
    expect(await perpFillClients([fill('1')])).toEqual(['web'])
  })

  it('gives every fill of one order that order client, from one query', async () => {
    await log(WALLET_A, at(390), 'mobile')
    const fills = [fill('1', { orderId: '77' }), fill('2', { orderId: '77' }), fill('3', { orderId: '77' })]
    expect(await perpFillClients(fills)).toEqual(['mobile', 'mobile', 'mobile'])
    expect(vi.mocked(sql)).toHaveBeenCalledTimes(1)
  })

  it('propagates an order client to a sibling fill outside the window', async () => {
    await log(WALLET_A, at(400), 'web')
    const late = fill('2', { orderId: '77', filledAt: at(60_000) })
    expect(await perpFillClients([fill('1', { orderId: '77' }), late])).toEqual(['web', 'web'])
    expect(await perpFillClients([late])).toEqual(['untagged'])
  })

  it('labels fills with no trade-log row untagged', async () => {
    await log(WALLET_B, at(400), 'web') // other wallet
    await log(WALLET_A, at(401), 'web', 'BTC') // other asset
    await log(WALLET_A, at(-400), 'web') // before the fill: an earlier order
    await log(WALLET_A, at(300), 'web', 'HOOD', { type: 'swap' })
    await log(WALLET_A, at(301), 'web', 'HOOD', { category: 'lighter' })
    expect(await perpFillClients([fill('1')])).toEqual(['untagged'])
  })

  it('includes the window edge and excludes rows after it', async () => {
    const edge = FILL_LOG_WINDOW_SECONDS * 1000
    await log(WALLET_A, at(edge), 'web')
    await log(WALLET_B, at(edge + 1), 'mobile')
    expect(await perpFillClients([fill('1'), fill('2', { walletAddress: WALLET_B })])).toEqual(['web', 'untagged'])
    await log(WALLET_B, at(0), 'mobile', 'BTC')
    expect(await perpFillClients([fill('3', { walletAddress: WALLET_B, market: 'BTC' })])).toEqual(['mobile'])
  })

  it('accepts the hl and hyperliquid venue spellings case-insensitively', async () => {
    await log(WALLET_A, at(400), 'web', 'HOOD', { category: 'hl' })
    await log(WALLET_B, at(400), 'mobile', 'HOOD', { category: 'HyperLiquid' })
    expect(await perpFillClients([fill('1'), fill('2', { walletAddress: WALLET_B })])).toEqual(['web', 'mobile'])
  })

  it('leaves both orders untagged when both fill before either row lands with different clients', async () => {
    await log(WALLET_A, at(400), 'mobile')
    await log(WALLET_A, at(500), 'web')
    const fills = [fill('1'), fill('2', { filledAt: at(100) })]
    expect(await perpFillClients(fills)).toEqual(['untagged', 'untagged'])
  })

  it('assigns the shared client when both fill before either row and the rows agree', async () => {
    await log(WALLET_A, at(400), 'mobile')
    await log(WALLET_A, at(500), 'mobile')
    expect(await perpFillClients([fill('1'), fill('2', { filledAt: at(100) })])).toEqual(['mobile', 'mobile'])
  })

  it('does not let an unlogged fill take the client of a logged later order', async () => {
    await log(WALLET_A, at(1_400), 'web') // only the later order logged
    const fills = [fill('1'), fill('2', { filledAt: at(1_000) })]
    expect(await perpFillClients(fills)).toEqual(['untagged', 'untagged'])
  })

  it('leaves a fill untagged when an extra row from an order outside the page is in its window', async () => {
    await log(WALLET_A, at(300), 'mobile') // e.g. an older order not on this page
    await log(WALLET_A, at(400), 'web')
    expect(await perpFillClients([fill('1')])).toEqual(['untagged'])
  })

  it('picks the nearest row when two orders follow closely', async () => {
    await log(WALLET_A, at(400), 'mobile')
    await log(WALLET_A, at(1_400), 'web')
    expect(await perpFillClients([fill('1'), fill('2', { filledAt: at(1_000) })])).toEqual(['mobile', 'web'])
  })

  it('matches wallets case-insensitively in both directions', async () => {
    await log(WALLET_A.toUpperCase().replace('0X', '0x'), at(400), 'web')
    await log(WALLET_B, at(400), 'mobile')
    const fills = [fill('1'), fill('2', { walletAddress: WALLET_B.toUpperCase().replace('0X', '0x') })]
    expect(await perpFillClients(fills)).toEqual(['web', 'mobile'])
  })

  it('normalizes logged labels like the swap rows', async () => {
    await log(WALLET_A, at(400), 'Mobile')
    await log(WALLET_B, at(400), '')
    expect(await perpFillClients([fill('1'), fill('2', { walletAddress: WALLET_B })])).toEqual(['mobile', 'untagged'])
  })

  it('matches native markets by coin and skips the query when there are no fills', async () => {
    await log(WALLET_A, at(400), 'web', 'BTC')
    expect(await perpFillClients([fill('1', { market: 'BTC' })])).toEqual(['web'])
    vi.mocked(sql).mockClear()
    expect(await perpFillClients([])).toEqual([])
    expect(vi.mocked(sql)).not.toHaveBeenCalled()
  })

  it('keeps fills and swaps when only the client lookup fails', async () => {
    vi.mocked(fetchHlLedgerTrades).mockResolvedValue([fill('1')])
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.mocked(sql).mockImplementation((async (query: string) => {
      if (query.includes('WITH ORDINALITY')) throw Error('timeout')
      return [{ ts: new Date(at(5_000)), timestamp: 'spot-id', type: 'swap', asset: 'KALSHI', client: 'mobile',
        wallet_address: 'solana-wallet', volume_usd: 25, is_close: null }]
    }) as unknown as typeof sql)
    const rows = await recentTrades('2026-10-05', '2026-10-05', 5)
    expect(rows.map(r => [r.type, r.client])).toEqual([['swap', 'mobile'], ['perps', 'unknown']])
    expect(errors).toHaveBeenCalledWith('recentTrades: perp fill client lookup failed', 'Error')
    errors.mockRestore()
  })

  it('shows the attributed client on recent trade rows', async () => {
    await log(WALLET_A, at(400), 'mobile')
    vi.mocked(fetchHlLedgerTrades).mockResolvedValue([fill('1'), fill('2', { walletAddress: WALLET_B })])
    const rows = await recentTrades('2026-10-05', '2026-10-05', 5)
    expect(rows.map(r => [r.walletAddress, r.client])).toEqual([[WALLET_B, 'untagged'], [WALLET_A, 'mobile']])
  })
})
