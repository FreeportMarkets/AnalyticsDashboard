import { describe, it, expect, vi } from 'vitest'
import { syncTrades } from '@/lib/sync/syncTrades'

const row = {
  wallet_address: '0xabc', timestamp: '2026-07-20T12:00:00.000Z',
  trade_date: '2026-07-20', type: 'perps', amount_usd: 610, status: 'success',
}

describe('syncTrades', () => {
  it('inserts mapped trades across both scanned dates', async () => {
    const args = {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      fetchTrades: vi.fn(async () => [row]),
      insertTrades: vi.fn(async (rows: unknown[]) => rows.length),
      quarantine: vi.fn(async () => {}),
    }
    const r = await syncTrades(args as never)
    expect(r.inserted).toBe(2)
    expect(args.advanceWatermark).toHaveBeenCalledWith('trades', expect.any(Date))
  })

  it('quarantines bad rows without failing the batch', async () => {
    const args = {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      fetchTrades: vi.fn(async () => [row, { nope: 1 }]),
      insertTrades: vi.fn(async (rows: unknown[]) => rows.length),
      quarantine: vi.fn(async () => {}),
    }
    const r = await syncTrades(args as never)
    expect(r.quarantined).toBe(2)
    expect(r.inserted).toBe(2)
  })

  it(
    'REGRESSION: counts rows quarantined via insertTrades\' onRowFailure bisection ' +
      'path (not just map-time failures) -- without this, scanned !== inserted + ' +
      'quarantined whenever bisection quarantines anything',
    async () => {
      const args = {
        now: new Date('2026-07-20T14:00:00.000Z'),
        readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
        advanceWatermark: vi.fn(async () => {}),
        fetchTrades: vi.fn(async () => [row]),
        // Simulate insertTrades bisecting: the row is rejected and reported
        // via onRowFailure -- the pre-fix mocks in this file never invoked it.
        insertTrades: vi.fn(
          async (rows: unknown[], onRowFailure?: (row: never, reason: string) => Promise<void>) => {
            if (onRowFailure) await onRowFailure(rows[0] as never, 'simulated data error')
            return rows.length - 1
          }
        ),
        quarantine: vi.fn(async () => {}),
      }
      const r = await syncTrades(args as never)
      // 2 fetch windows x 1 bisection-quarantined row each, 0 inserted.
      expect(r.inserted).toBe(0)
      expect(r.quarantined).toBe(2)
      expect(r.scanned).toBe(r.inserted + r.quarantined)
    }
  )

  it('does not advance the watermark when an insert throws', async () => {
    const args = {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      fetchTrades: vi.fn(async () => [row]),
      insertTrades: vi.fn(async () => { throw new Error('neon down') }),
      quarantine: vi.fn(async () => {}),
    }
    await expect(syncTrades(args as never)).rejects.toThrow('neon down')
    expect(args.advanceWatermark).not.toHaveBeenCalled()
  })
})
