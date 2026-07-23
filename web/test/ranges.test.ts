import { describe, expect, it } from 'vitest'
import { RANGES, addDays, isRangeKey, rangeSpanDays, rangeStart } from '../src/lib/ranges'

/**
 * These figures are compared by eye against the Streamlit dashboard, so the
 * preset arithmetic is a contract, not an implementation detail. Streamlit's
 * default (app.py) is:
 *
 *     value=(today - timedelta(days=7), today)
 *
 * -- an INCLUSIVE range of `today - lookback` through `today`. If these tests
 * fail, either fix the regression or change app.py in the same commit.
 */
describe('range presets (Streamlit parity)', () => {
  it('starts lookbackDays before the end date, not lookbackDays - 1', () => {
    expect(rangeStart('2026-07-23', '7d')).toBe('2026-07-16')
    expect(rangeStart('2026-07-23', '30d')).toBe('2026-06-23')
    expect(rangeStart('2026-07-23', '90d')).toBe('2026-04-24')
  })

  it('matches what Streamlit computes for the same lookback', () => {
    const end = '2026-07-23'
    for (const key of Object.keys(RANGES) as Array<keyof typeof RANGES>) {
      // Streamlit: start = today - timedelta(days=N); end = today
      const streamlitStart = addDays(end, -RANGES[key].lookbackDays)
      expect(rangeStart(end, key)).toBe(streamlitStart)
    }
  })

  it('covers lookbackDays + 1 calendar dates, inclusive', () => {
    const end = '2026-07-23'
    for (const key of Object.keys(RANGES) as Array<keyof typeof RANGES>) {
      const start = rangeStart(end, key)
      let count = 0
      for (let d = start; d <= end; d = addDays(d, 1)) count++
      expect(count).toBe(rangeSpanDays(key))
      expect(count).toBe(RANGES[key].lookbackDays + 1)
    }
  })

  it('crosses a DST boundary without losing or gaining a date', () => {
    // US DST ends 2026-11-01. A naive +24h loop drops or repeats a day here.
    const start = rangeStart('2026-11-03', '7d')
    expect(start).toBe('2026-10-27')
    let count = 0
    for (let d = start; d <= '2026-11-03'; d = addDays(d, 1)) count++
    expect(count).toBe(8)
  })

  it('crosses month and year boundaries', () => {
    expect(rangeStart('2026-01-03', '7d')).toBe('2025-12-27')
    expect(rangeStart('2026-03-05', '90d')).toBe('2025-12-05')
  })

  it('accepts only the three known keys', () => {
    expect(isRangeKey('7d')).toBe(true)
    expect(isRangeKey('30d')).toBe(true)
    expect(isRangeKey('90d')).toBe(true)
    expect(isRangeKey('1d')).toBe(false)
    expect(isRangeKey('')).toBe(false)
    expect(isRangeKey(undefined)).toBe(false)
  })
})
