import { describe, it, expect } from 'vitest'
import { partitionNameFor, partitionBoundsFor } from '@/lib/sync/partitions'

describe('partition naming', () => {
  it('names a partition per month', () => {
    expect(partitionNameFor('2026-07-19')).toBe('events_2026_07')
    expect(partitionNameFor('2026-01-01')).toBe('events_2026_01')
  })

  it('computes half-open month bounds', () => {
    expect(partitionBoundsFor('2026-07-19')).toEqual({ from: '2026-07-01', to: '2026-08-01' })
  })

  it('rolls the year over in December', () => {
    expect(partitionBoundsFor('2026-12-31')).toEqual({ from: '2026-12-01', to: '2027-01-01' })
  })

  it('rejects a malformed date rather than generating a bad DDL identifier', () => {
    expect(() => partitionNameFor('nope')).toThrow(/invalid date/i)
    expect(() => partitionNameFor("2026-07-01'; DROP TABLE events; --")).toThrow(/invalid date/i)
  })
})
