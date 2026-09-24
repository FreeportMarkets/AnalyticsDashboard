import { describe, expect, it } from 'vitest'
import { mergeDailyVolumeRows } from '@/lib/metrics/mergeDailyVolumeRows'

const tradeDays = [{ day: '2026-09-22', swapVolumeUsd: 20, perpsVolumeUsd: 80, tradeCount: 3 }]
const fillDays = [
  { day: '2026-09-21', notionalUsd: 150, fillCount: 2 },
  { day: '2026-09-22', notionalUsd: 90, fillCount: 1 },
]

describe('daily ledger/trade-log merge', () => {
  it('shows fill-only days and keeps trade-log swaps and counts', () => {
    expect(mergeDailyVolumeRows(tradeDays, fillDays, true)).toEqual([
      { day: '2026-09-21', swapVolumeUsd: 0, perpsVolumeUsd: 150, tradeCount: 0 },
      { day: '2026-09-22', swapVolumeUsd: 20, perpsVolumeUsd: 90, tradeCount: 3 },
    ])
  })

  it('retains the estimate path if fill coverage is unavailable', () => {
    expect(mergeDailyVolumeRows(tradeDays, fillDays, false)).toEqual(tradeDays)
  })
})
