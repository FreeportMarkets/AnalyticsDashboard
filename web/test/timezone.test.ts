import { describe, it, expect } from 'vitest'
import { NY_TZ, nyDateExpr, utcDateOf, nyDateOf } from '@/lib/time'

describe('timezone handling', () => {
  it('uses America/New_York', () => {
    expect(NY_TZ).toBe('America/New_York')
  })

  it('builds a NY-bucketing SQL expression', () => {
    expect(nyDateExpr('ts')).toBe("(ts AT TIME ZONE 'America/New_York')::date")
  })

  // THE guard. An event at 01:30 UTC on Jul 20 is 21:30 EDT on Jul 19.
  // The DynamoDB partition key says 2026-07-20; every metric must say 2026-07-19.
  // If this ever passes as equal, the highest-severity risk in the spec is live.
  it('proves the UTC partition key disagrees with the NY bucket', () => {
    const ts = new Date('2026-07-20T01:30:00.000Z')
    expect(utcDateOf(ts)).toBe('2026-07-20')
    expect(nyDateOf(ts)).toBe('2026-07-19')
    expect(utcDateOf(ts)).not.toBe(nyDateOf(ts))
  })

  it('agrees mid-afternoon, when both fall on the same calendar day', () => {
    const ts = new Date('2026-07-20T18:00:00.000Z')
    expect(utcDateOf(ts)).toBe(nyDateOf(ts))
  })

  it('handles the EST/EDT boundary', () => {
    // 2026-11-01 06:30Z is 02:30 EDT (UTC-4), still Nov 1 in NY.
    expect(nyDateOf(new Date('2026-11-01T06:30:00.000Z'))).toBe('2026-11-01')
    // 2026-01-15 02:30Z is 21:30 EST (UTC-5) on Jan 14.
    expect(nyDateOf(new Date('2026-01-15T02:30:00.000Z'))).toBe('2026-01-14')
  })
})
