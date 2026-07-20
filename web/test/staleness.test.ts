import { describe, it, expect } from 'vitest'
import { formatAge, isStale, STALE_THRESHOLD_SECONDS } from '@/lib/metrics/staleness'

describe('staleness', () => {
  it('treats 10 minutes as the threshold', () => {
    // The sync holds its watermark 10 minutes behind wall-clock by design,
    // so anything materially beyond that means the cron is not running.
    expect(STALE_THRESHOLD_SECONDS).toBe(15 * 60)
  })

  it('formats ages readably', () => {
    expect(formatAge(45)).toBe('45s')
    expect(formatAge(90)).toBe('1m 30s')
    expect(formatAge(3700)).toBe('1h 1m')
  })

  it('flags staleness past the threshold', () => {
    expect(isStale(600)).toBe(false)
    expect(isStale(901)).toBe(true)
  })
})
