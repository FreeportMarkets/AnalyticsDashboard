import { describe, it, expect } from 'vitest'
import { nyRangeToUtc } from '@/lib/metrics/nyRange'

describe('nyRangeToUtc', () => {
  it('maps an EDT day to its UTC instants', () => {
    // 2026-07-20 00:00 EDT = 04:00Z; exclusive end is 2026-07-21 00:00 EDT = 04:00Z
    const r = nyRangeToUtc('2026-07-20', '2026-07-20')
    expect(r.fromUtc.toISOString()).toBe('2026-07-20T04:00:00.000Z')
    expect(r.toUtc.toISOString()).toBe('2026-07-21T04:00:00.000Z')
  })

  it('maps an EST day to its UTC instants', () => {
    const r = nyRangeToUtc('2026-01-15', '2026-01-15')
    expect(r.fromUtc.toISOString()).toBe('2026-01-15T05:00:00.000Z')
    expect(r.toUtc.toISOString()).toBe('2026-01-16T05:00:00.000Z')
  })

  it('spans a multi-day range', () => {
    const r = nyRangeToUtc('2026-07-01', '2026-07-07')
    expect(r.fromUtc.toISOString()).toBe('2026-07-01T04:00:00.000Z')
    expect(r.toUtc.toISOString()).toBe('2026-07-08T04:00:00.000Z')
  })

  it('rejects a malformed or non-calendar date', () => {
    expect(() => nyRangeToUtc('2026-02-30', '2026-03-01')).toThrow(/invalid date/i)
    expect(() => nyRangeToUtc('nope', '2026-03-01')).toThrow(/invalid date/i)
  })

  it('rejects an inverted range', () => {
    expect(() => nyRangeToUtc('2026-07-10', '2026-07-01')).toThrow(/range/i)
  })
})
