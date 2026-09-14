import { describe, expect, it } from 'vitest'
import { resolveVolume } from '../src/lib/metrics/volumeResolve'

const kpis = (volume: number, prevVolume: number, swap: number, prevSwap: number) => ({
  volumeUsd: { current: volume, previous: prevVolume },
  swapVolumeUsd: { current: swap, previous: prevSwap },
})

describe('resolveVolume', () => {
  it('falls back to the (swap+perps) DB reconstruction when HL has no data', () => {
    const result = resolveVolume(
      { current: 0, previous: 0, hasData: false },
      kpis(3_170_000, 2_900_000, 500_000, 450_000)
    )
    expect(result).toEqual({ current: 3_170_000, previous: 2_900_000, source: 'est' })
  })

  it('adds exact swap volume on top of authoritative HL perps volume once backfilled', () => {
    // Regression for the bug where the Overview headline went perps-only
    // (dropping every swap dollar) the moment hasData flipped true, while
    // the Trades page kept adding swap back in -- see volume-parity
    // integration test and resolveVolume's doc comment.
    const result = resolveVolume(
      { current: 2_000_000, previous: 1_800_000, hasData: true },
      kpis(0, 0, 500_000, 450_000)
    )
    expect(result).toEqual({ current: 2_500_000, previous: 2_250_000, source: 'hl' })
  })

  it('still adds swap volume when HL perps volume is genuinely zero for the range', () => {
    const result = resolveVolume(
      { current: 0, previous: 0, hasData: true },
      kpis(0, 0, 500_000, 450_000)
    )
    expect(result).toEqual({ current: 500_000, previous: 450_000, source: 'hl' })
  })
})
