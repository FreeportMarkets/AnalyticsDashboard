import { beforeEach, describe, expect, it, vi } from 'vitest'
vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
vi.mock('@/lib/hlLedgerTradesApi', async importOriginal => ({
  ...await importOriginal<typeof import('@/lib/hlLedgerTradesApi')>(), fetchHlLedgerTrades: vi.fn(),
}))
import { sql } from '@/lib/db'
import { fetchHlLedgerTrades, parseHlLedgerTrades, type HlLedgerTrade } from '@/lib/hlLedgerTradesApi'
import { recentTrades } from '@/lib/metrics/trades'

const fill = (id: string, wallet = '0x1111111111111111111111111111111111111111'): HlLedgerTrade => ({
  fillId: id, orderId: '10', walletAddress: wallet, market: 'xyz:HOOD', side: 'B', direction: 'Open Long',
  size: '0.397', price: '113.31', notionalUsd: '44.98407', filledAt: '2026-10-05T17:52:52.644Z',
})
const report = (trades: HlLedgerTrade[]) => ({ schemaVersion: 1, timezone: 'America/New_York',
  completeness: 'provisional', range: { from: '2026-10-05', to: '2026-10-05' }, trades })
beforeEach(() => { vi.mocked(sql).mockReset(); vi.mocked(sql).mockResolvedValue([]); vi.mocked(fetchHlLedgerTrades).mockReset() })

describe('Recorded fill contract', () => {
  it('retains partial fills sharing an order and timestamp, rejects duplicate identities', () => {
    expect(parseHlLedgerTrades(report([fill('1'), fill('2')]), '2026-10-05', '2026-10-05', 2)).toHaveLength(2)
    expect(() => parseHlLedgerTrades(report([fill('1'), fill('1')]), '2026-10-05', '2026-10-05', 2)).toThrow()
  })
  it('rejects malformed amounts, wrong range, over-limit and invalid/outside timestamps', () => {
    for (const patch of [{ notionalUsd: 'NaN' }, { filledAt: '2020-01-01T00:00:00.000Z' },
      { filledAt: '2026-10-05' }, { filledAt: '2026-10-06T04:00:00.000Z' }]) {
      expect(() => parseHlLedgerTrades(report([{ ...fill('1'), ...patch }]), '2026-10-05', '2026-10-05', 1)).toThrow()
    }
    expect(() => parseHlLedgerTrades(report([fill('1')]), '2026-10-04', '2026-10-05', 1)).toThrow()
    expect(() => parseHlLedgerTrades(report([fill('1'), fill('2')]), '2026-10-05', '2026-10-05', 1)).toThrow()
  })
})

describe('Recent trades source composition', () => {
  it('reads spot only from history and merges exact fills without adding legacy perps', async () => {
    vi.mocked(fetchHlLedgerTrades).mockResolvedValue([fill('1'), { ...fill('2'), direction: 'Close Long', side: 'A' }])
    vi.mocked(sql).mockResolvedValue([{ ts: new Date('2026-10-05T18:00:00Z'), timestamp: 'spot-id', type: 'swap',
      asset: 'KALSHI', wallet_address: 'solana-wallet', volume_usd: 25, is_close: null }])
    const rows = await recentTrades('2026-10-05', '2026-10-05', 3)
    expect(rows.map(r => r.type)).toEqual(['swap', 'perps', 'perps'])
    expect(rows[1]!.volumeUsd).toBe(44.98407)
    expect(rows[1]!.leverage).toBeNull()
    expect(rows[1]!.client).toBe('unknown')
    expect(String(vi.mocked(sql).mock.calls[0]![0])).toContain("AND type = 'swap'")
    expect(new Set(rows.map(r => r.timestamp)).size).toBe(3)
  })
  it('preserves equal-time top-N wallet/id ordering as more rows are requested', async () => {
    const source = [fill('7', '0xcccccccccccccccccccccccccccccccccccccccc'),
      fill('8', '0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'), fill('9')]
    vi.mocked(fetchHlLedgerTrades).mockImplementation(async (_from, _to, limit) => source.slice(0, limit))
    const first = await recentTrades('2026-10-05', '2026-10-05', 2)
    const expanded = await recentTrades('2026-10-05', '2026-10-05', 3)
    expect(expanded.slice(0, 2)).toEqual(first)
  })
  it('preserves reversal direction without claiming it is an opening or a long position', async () => {
    vi.mocked(fetchHlLedgerTrades).mockResolvedValue([{ ...fill('1'), direction: 'Long > Short', side: 'A' }])
    const [row] = await recentTrades('2026-10-05', '2026-10-05')
    expect(row).toMatchObject({ action: 'Long > Short', side: null, isClose: null })
  })
  it('surfaces ledger read failure instead of substituting incomplete legacy history', async () => {
    vi.mocked(fetchHlLedgerTrades).mockRejectedValue(new Error('unavailable'))
    await expect(recentTrades('2026-10-05', '2026-10-05')).rejects.toThrow('unavailable')
  })
})
