import { describe, it, expect } from 'vitest'
import { computeNextWatermark, WATERMARK_LAG_MS, datesToScan } from '@/lib/sync/watermark'

describe('watermark', () => {
  it('lags 10 minutes behind now', () => {
    expect(WATERMARK_LAG_MS).toBe(10 * 60 * 1000)
    const now = new Date('2026-07-20T14:00:00.000Z')
    expect(computeNextWatermark(now).toISOString()).toBe('2026-07-20T13:50:00.000Z')
  })

  it('never advances to now, so late arrivals are still captured', () => {
    const now = new Date('2026-07-20T14:00:00.000Z')
    expect(computeNextWatermark(now).getTime()).toBeLessThan(now.getTime())
  })

  it('scans the current and previous UTC date partitions', () => {
    // Events near UTC midnight land in the next partition. app.py:631-646 fetches
    // an extra day for the same reason.
    expect(datesToScan(new Date('2026-07-20T00:05:00.000Z')))
      .toEqual(['2026-07-19', '2026-07-20'])
  })

  it('rolls across a month boundary', () => {
    expect(datesToScan(new Date('2026-08-01T00:05:00.000Z')))
      .toEqual(['2026-07-31', '2026-08-01'])
  })
})
