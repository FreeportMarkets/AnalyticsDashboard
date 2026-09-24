import { describe, expect, it } from 'vitest'
import { isHlLedgerMetrics } from '../src/lib/hlLedgerApi'

const day = {
  day: '2026-09-23', fillCount: 157, notionalUsd: '349352.422649',
  confirmedFeeUsd: '50.155444', unresolvedBuilderFeeUsd: '174.308859',
  unresolvedFillCount: 62, latestFillAt: '2026-09-23T23:02:06.038Z',
  latestIngestAt: '2026-09-23T23:03:57.388Z',
}

describe('HL ledger response boundary', () => {
  const report = { schemaVersion: 1, timezone: 'America/New_York', completeness: 'provisional',
    range: { from: '2026-09-23', to: '2026-09-23' }, days: [day] }

  it('accepts exact decimal volume and keeps unresolved fees distinct', () => {
    expect(isHlLedgerMetrics(report, '2026-09-23', '2026-09-23')).toBe(true)
    expect(isHlLedgerMetrics({ ...report, days: [{ ...day, confirmedFeeUsd: '224.464303',
      unresolvedBuilderFeeUsd: '0' }] }, '2026-09-23', '2026-09-23')).toBe(true)
  })

  it('rejects wrong day, counts, money shape and contract version', () => {
    expect(isHlLedgerMetrics(report, '2026-09-22', '2026-09-23')).toBe(false)
    expect(isHlLedgerMetrics({ ...report, schemaVersion: 2 }, '2026-09-23', '2026-09-23')).toBe(false)
    expect(isHlLedgerMetrics({ ...report, days: [{ ...day, unresolvedFillCount: 158 }] }, '2026-09-23', '2026-09-23')).toBe(false)
    expect(isHlLedgerMetrics({ ...report, days: [{ ...day, notionalUsd: 'NaN' }] }, '2026-09-23', '2026-09-23')).toBe(false)
  })
})
