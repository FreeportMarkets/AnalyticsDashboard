import { describe, it, expect } from 'vitest'
import { NY_TZ, nyDateExpr, utcDateOf, nyDateOf, isValidCalendarDate } from '@/lib/time'

describe('timezone handling', () => {
  it('uses America/New_York', () => {
    expect(NY_TZ).toBe('America/New_York')
  })

  it('builds a NY-bucketing SQL expression', () => {
    expect(nyDateExpr('ts')).toBe("(ts AT TIME ZONE 'America/New_York')::date")
  })

  it('builds a NY-bucketing SQL expression for a qualified column', () => {
    expect(nyDateExpr('e.ts')).toBe("(e.ts AT TIME ZONE 'America/New_York')::date")
  })

  it('rejects a tsColumn that is not a plain SQL identifier', () => {
    expect(() => nyDateExpr("ts'; DROP TABLE events; --")).toThrow()
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

describe('isValidCalendarDate', () => {
  it.each(['2026-07-19', '2024-02-29', '0001-01-01', '9999-12-31'])(
    'accepts a real calendar date: %s',
    (value) => {
      expect(isValidCalendarDate(value)).toBe(true)
    },
  )

  it.each([
    '2026-02-30',
    '2026-02-29',
    '2026-04-31',
    '2026-13-01',
    '2026-00-10',
    '26-01-01',
    'not-a-date',
    '',
  ])('rejects an invalid or malformed date: %s', (value) => {
    expect(isValidCalendarDate(value)).toBe(false)
  })

  // JS `Date` uses proleptic year numbering and round-trips year zero
  // cleanly; Postgres's `date` type has no year zero and errors on it.
  // Verified live against postgres:16-alpine: `SELECT '0000-06-15'::date`
  // -> "ERROR: date/time field value out of range". Left unrejected, a row
  // with this date reaches the UNNEST-based bulk insert and throws, wedging
  // the sync watermark permanently (the ingest endpoint is unauthenticated,
  // so this is trivially craftable).
  it.each(['0000-01-01', '0000-06-15'])('rejects a year-zero date: %s', (value) => {
    expect(isValidCalendarDate(value)).toBe(false)
  })
})
