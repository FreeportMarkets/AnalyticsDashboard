import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
vi.mock('@/lib/hlLedgerApi', () => ({ fetchHlLedgerMetrics: vi.fn() }))

import { sql } from '@/lib/db'
import { fetchHlLedgerMetrics } from '@/lib/hlLedgerApi'
import { hlVolumeDaily, hlVolumeKpi, hlVolumeTotal } from '@/lib/metrics/hlVolumeRead'

const day = { day: '2026-09-23', fillCount: 2, notionalUsd: '150.25',
  confirmedFeeUsd: '0.0975', unresolvedBuilderFeeUsd: '0.03',
  unresolvedFillCount: 1, latestFillAt: null, latestIngestAt: null }
const report = (from: string, to: string, days = [day]) => ({
  schemaVersion: 1 as const, timezone: 'America/New_York' as const,
  completeness: 'provisional' as const, range: { from, to }, days,
})

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubEnv('HL_VOLUME_SOURCE', 'backend')
})
afterEach(() => vi.unstubAllEnvs())

describe('gated backend fill-ledger volume', () => {
  it('uses recorded fills and recipient-proven fees without counting unresolved fees as revenue', async () => {
    vi.mocked(fetchHlLedgerMetrics).mockResolvedValue(report('2026-09-23', '2026-09-23'))
    expect(await hlVolumeTotal('2026-09-23', '2026-09-23')).toEqual({
      notionalUsd: 150.25, builderFeeUsd: 0.0975, unresolvedBuilderFeeUsd: 0.03,
      fillCount: 2, source: 'backend',
    })
    expect(await hlVolumeDaily('2026-09-23', '2026-09-23')).toEqual([
      { day: '2026-09-23', notionalUsd: 150.25, fillCount: 2 },
    ])
    expect(sql).not.toHaveBeenCalled()
  })

  it('does not treat an empty backend report as proven zero volume', async () => {
    vi.mocked(fetchHlLedgerMetrics).mockImplementation(async (from, to) => report(from, to, []))
    expect(await hlVolumeKpi('2026-09-23', '2026-09-23', '2026-09-22', '2026-09-22'))
      .toMatchObject({ current: 0, previous: 0, hasData: false })
  })

  it('falls back when the enabled backend read fails', async () => {
    vi.mocked(fetchHlLedgerMetrics).mockRejectedValue(new Error('backend unavailable'))
    expect(await hlVolumeKpi('2026-09-23', '2026-09-23', '2026-09-22', '2026-09-22'))
      .toMatchObject({ current: 0, previous: 0, hasData: false })
  })

  it('keeps the Neon daily read when the flag is off', async () => {
    vi.stubEnv('HL_VOLUME_SOURCE', '')
    vi.mocked(sql).mockResolvedValueOnce([{ day: '2026-09-23', notional: 100, fills: 1 }] as never)
    expect(await hlVolumeDaily('2026-09-23', '2026-09-23')).toEqual([
      { day: '2026-09-23', notionalUsd: 100, fillCount: 1 },
    ])
    expect(fetchHlLedgerMetrics).not.toHaveBeenCalled()
  })
})
