import { describe, expect, it } from 'vitest'
import { aggregateWalletFills, isFreeportPerpFill, nyDayOf } from '../src/lib/volume/aggregate'
import type { HlFill } from '../src/lib/hl/volume'

// 2026-07-15 13:00:00 UTC = 2026-07-15 09:00 NY. A fixed anchor; every fill
// time below is an offset from this so the NY-day bucketing is deterministic.
const BASE = Date.UTC(2026, 6, 15, 13, 0, 0)

const fill = (over: Partial<HlFill> = {}): HlFill => ({
  sz: '1', px: '1000', dir: 'Open Long', tid: Math.random(), time: BASE,
  builderFee: '0.65', ...over,
})

describe('isFreeportPerpFill', () => {
  it('accepts a perp fill with a builder fee', () => {
    expect(isFreeportPerpFill(fill({ dir: 'Open Long', builderFee: '0.65' }))).toBe(true)
  })
  it('rejects a perp fill with no builder fee (not ours)', () => {
    expect(isFreeportPerpFill(fill({ dir: 'Open Long', builderFee: '0' }))).toBe(false)
    expect(isFreeportPerpFill(fill({ dir: 'Open Long', builderFee: undefined }))).toBe(false)
  })
  it('rejects spot / settlement dirs even with a builder fee', () => {
    for (const dir of ['Buy', 'Sell', 'Spot Dust Conversion', 'Settlement']) {
      expect(isFreeportPerpFill(fill({ dir, builderFee: '0.65' }))).toBe(false)
    }
  })

  // Advanced order types: these use perp directions beyond the basic four and
  // MUST be counted when they carry our builder fee. This is the "handle all
  // order placements" guarantee.
  it('accepts position-flip fills (Long > Short / Short > Long) with a builder fee', () => {
    expect(isFreeportPerpFill(fill({ dir: 'Long > Short', builderFee: '1.2' }))).toBe(true)
    expect(isFreeportPerpFill(fill({ dir: 'Short > Long', builderFee: '1.2' }))).toBe(true)
  })

  it('accepts a TWAP sub-fill (normal dir + twapId) with a builder fee', () => {
    expect(isFreeportPerpFill(fill({ dir: 'Open Long', builderFee: '0.4', twapId: 12345 }))).toBe(true)
  })

  it('accepts an unknown future perp dir as long as it carried our fee', () => {
    // Forward-compat: a new HL perp dir must not be silently dropped.
    expect(isFreeportPerpFill(fill({ dir: 'Some New Perp Action', builderFee: '0.9' }))).toBe(true)
  })

  it('excludes a liquidation fill that carried no builder fee (not our trade)', () => {
    expect(isFreeportPerpFill(fill({ dir: 'Liquidated Isolated Long', builderFee: '0' }))).toBe(false)
  })
})

describe('nyDayOf', () => {
  it('buckets by NY calendar day, not UTC', () => {
    // 2026-07-16 03:00 UTC = 2026-07-15 23:00 NY -> still the 15th.
    expect(nyDayOf(Date.UTC(2026, 6, 16, 3, 0, 0))).toBe('2026-07-15')
    // 2026-07-16 05:00 UTC = 2026-07-16 01:00 NY -> the 16th.
    expect(nyDayOf(Date.UTC(2026, 6, 16, 5, 0, 0))).toBe('2026-07-16')
  })
})

describe('aggregateWalletFills', () => {
  it('sums notional and fee per day for attributed perp fills', () => {
    const r = aggregateWalletFills([
      fill({ tid: 1, sz: '2', px: '1000', builderFee: '1.3' }), // 2000
      fill({ tid: 2, sz: '1', px: '500', builderFee: '0.325' }), // 500
    ])
    expect(r.buckets).toHaveLength(1)
    expect(r.buckets[0]!.notionalUsd).toBe(2500)
    expect(r.buckets[0]!.builderFeeUsd).toBeCloseTo(1.625, 6)
    expect(r.buckets[0]!.fillCount).toBe(2)
    expect(r.attributedFills).toBe(2)
  })

  it('excludes non-builder-fee fills from volume but still advances the watermark', () => {
    const t2 = BASE + 60_000
    const r = aggregateWalletFills([
      fill({ tid: 1, sz: '1', px: '1000', builderFee: '0.65', time: BASE }),
      fill({ tid: 2, sz: '9', px: '1000', builderFee: '0', time: t2 }), // not ours
    ])
    expect(r.buckets).toHaveLength(1)
    expect(r.buckets[0]!.notionalUsd).toBe(1000) // the $9k non-builder fill excluded
    expect(r.maxFillMs).toBe(t2) // ...but the watermark still moved past it
    expect(r.attributedFills).toBe(1)
  })

  it('splits across NY days', () => {
    const r = aggregateWalletFills([
      fill({ tid: 1, sz: '1', px: '1000', time: Date.UTC(2026, 6, 16, 3, 0, 0) }), // NY 15th
      fill({ tid: 2, sz: '1', px: '2000', time: Date.UTC(2026, 6, 16, 5, 0, 0) }), // NY 16th
    ])
    expect(r.buckets.map(b => b.day)).toEqual(['2026-07-15', '2026-07-16'])
    expect(r.buckets[0]!.notionalUsd).toBe(1000)
    expect(r.buckets[1]!.notionalUsd).toBe(2000)
  })

  it('ignores fills at or before the watermark (incremental)', () => {
    const r = aggregateWalletFills([
      fill({ tid: 1, sz: '1', px: '1000', time: BASE }),            // <= since, skip
      fill({ tid: 2, sz: '1', px: '3000', time: BASE + 1_000 }),    // after, keep
    ], BASE)
    expect(r.buckets).toHaveLength(1)
    expect(r.buckets[0]!.notionalUsd).toBe(3000)
    expect(r.maxFillMs).toBe(BASE + 1_000)
  })

  it('dedups by tid across the input', () => {
    const r = aggregateWalletFills([
      fill({ tid: 7, sz: '1', px: '1000' }),
      fill({ tid: 7, sz: '1', px: '1000' }), // duplicate
    ])
    expect(r.buckets[0]!.notionalUsd).toBe(1000)
    expect(r.buckets[0]!.fillCount).toBe(1)
  })

  it('returns the watermark unchanged and no buckets for an empty input', () => {
    const r = aggregateWalletFills([], 12345)
    expect(r.buckets).toEqual([])
    expect(r.maxFillMs).toBe(12345)
    expect(r.attributedFills).toBe(0)
  })

  it('uses absolute size so shorts add positive notional', () => {
    const r = aggregateWalletFills([fill({ tid: 1, dir: 'Close Short', sz: '-3', px: '1000' })])
    expect(r.buckets[0]!.notionalUsd).toBe(3000)
  })
})
