import { describe, expect, it } from 'vitest'
import { journeyPreset } from '@/lib/journeyRanges'

describe('Journey independent cohort calendars', () => {
  it.each([
    ['2026-09-22T01:00:00Z', '2026-09-22', '2026-09-21'],
    ['2026-03-08T04:30:00Z', '2026-03-08', '2026-03-07'],
    ['2026-03-09T04:30:00Z', '2026-03-09', '2026-03-09'],
    ['2026-11-01T04:30:00Z', '2026-11-01', '2026-11-01'],
    ['2026-11-02T04:30:00Z', '2026-11-02', '2026-11-01'],
  ])('preserves UTC intro and NY account dates at %s', (now, utc, ny) => {
    for (const days of [14, 30, 90]) {
      const range = journeyPreset(days, new Date(now))
      expect(range.to).toBe(utc)
      expect(range.accountTo).toBe(ny)
      expect((Date.parse(range.to) - Date.parse(range.from)) / 86400000 + 1).toBe(days)
      expect((Date.parse(range.accountTo) - Date.parse(range.accountFrom)) / 86400000 + 1).toBe(days)
    }
  })
})
