import { describe, it, expect } from 'vitest'
import { partitionNameFor, partitionBoundsFor, ensurePartitions } from '@/lib/sync/partitions'

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

  it('rejects a calendar-rollover date like 2026-02-30', () => {
    expect(() => partitionNameFor('2026-02-30')).toThrow(/invalid date/i)
  })

  it('pads a sub-1000 year in both bounds so they cannot be misread by a two-digit-year heuristic', () => {
    expect(partitionBoundsFor('0099-07-15')).toEqual({ from: '0099-07-01', to: '0099-08-01' })
  })
})

describe('ensurePartitions', () => {
  function makeFakeSql() {
    const calls: string[] = []
    const fakeSql = (async (text: string) => {
      calls.push(text)
      return []
    }) as never
    return { calls, fakeSql }
  }

  it('dedups two dates in the same month into exactly one CREATE TABLE', async () => {
    const { calls, fakeSql } = makeFakeSql()
    await ensurePartitions(fakeSql, ['2026-07-01', '2026-07-19'])
    const createTables = calls.filter((c) => c.includes('CREATE TABLE'))
    expect(createTables).toHaveLength(1)
  })

  it('emits one CREATE TABLE per distinct month', async () => {
    const { calls, fakeSql } = makeFakeSql()
    await ensurePartitions(fakeSql, ['2026-07-19', '2026-08-01'])
    const createTables = calls.filter((c) => c.includes('CREATE TABLE'))
    expect(createTables).toHaveLength(2)
  })

  it('names the partition and bounds correctly for a July 2026 input', async () => {
    const { calls, fakeSql } = makeFakeSql()
    await ensurePartitions(fakeSql, ['2026-07-19'])
    const createTable = calls.find((c) => c.includes('CREATE TABLE'))
    expect(createTable).toContain('events_2026_07')
    expect(createTable).toContain("FROM ('2026-07-01') TO ('2026-08-01')")
  })

  it('emits three CREATE INDEX statements per partition', async () => {
    const { calls, fakeSql } = makeFakeSql()
    await ensurePartitions(fakeSql, ['2026-07-19'])
    const createIndexes = calls.filter((c) => c.includes('CREATE INDEX'))
    expect(createIndexes).toHaveLength(3)
  })

  it('returns the ensured partition names', async () => {
    const { fakeSql } = makeFakeSql()
    const result = await ensurePartitions(fakeSql, ['2026-07-19', '2026-08-01'])
    expect(result.sort()).toEqual(['events_2026_07', 'events_2026_08'])
  })
})
