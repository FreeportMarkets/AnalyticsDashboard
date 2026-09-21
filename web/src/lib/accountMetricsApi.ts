import { COHORT_DAYS, type AccountMetrics } from './accountMetrics'

export type AccountMetricsFailure = 'not_configured' | 'unauthorized' | 'bad_range' | 'not_deployed' | 'unavailable' | 'invalid_response'
export type AccountMetricsResult = { ok: true; data: AccountMetrics } | { ok: false; reason: AccountMetricsFailure }
const record = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value)
const integer = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
const date = (value: unknown): value is string => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(`${value}T00:00:00Z`)) && new Date(`${value}T00:00:00Z`).toISOString().slice(0, 10) === value
const timestamp = (value: unknown) => value === null || typeof value === 'string' && Number.isFinite(Date.parse(value))
const money = (value: unknown) => value === null || typeof value === 'string' && /^\d+(\.\d+)?$/.test(value) && Number.isFinite(Number(value))

/** Reject contract drift rather than rendering plausible numbers with wrong denominators. */
export function isAccountMetrics(value: unknown, from: string, to: string): value is AccountMetrics {
  if (!record(value) || value.schemaVersion !== 1 || value.timezone !== 'America/New_York' || value.cohortBasis !== 'account_created'
    || !timestamp(value.generatedAt) || value.generatedAt === null || !record(value.range) || value.range.from !== from || value.range.to !== to
    || !record(value.coverage) || !['partial', 'unavailable', 'stale'].includes(String(value.coverage.status))
    || !Array.isArray(value.coverage.warnings) || !value.coverage.warnings.every(w => typeof w === 'string')
    || !['accountsAsOf', 'activityAsOf', 'activitySourceFrom', 'activitySourceThrough'].every(key => timestamp(value.coverage && (value.coverage as Record<string, unknown>)[key]))
    || !Array.isArray(value.rows) || value.expectedLtv !== null || value.acquisitionCost !== null) return false
  // Status can be unavailable because only activity is missing. Validate each
  // source separately so valid financial observations remain usable.
  if (value.coverage.accountsAsOf === null && value.rows.length > 0) return false
  const hasActivityCoverage = ['activityAsOf', 'activitySourceFrom', 'activitySourceThrough']
    .every(key => value.coverage && (value.coverage as Record<string, unknown>)[key] !== null)
  const dates = new Set<string>()
  return value.rows.every(row => {
    if (!record(row) || !date(row.cohortDate) || row.cohortDate < from || row.cohortDate > to || dates.has(row.cohortDate)
      || !integer(row.accounts) || !integer(row.mobileLinkedAccounts) || row.mobileLinkedAccounts > row.accounts
      || !['observedFundedAccounts', 'observedFirstTradeAccounts'].every(key => row[key] === null || integer(row[key]) && (row[key] as number) <= (row.accounts as number))
      || !Array.isArray(row.horizons) || row.horizons.length !== COHORT_DAYS.length) return false
    dates.add(row.cohortDate)
    const days = new Set<number>()
    return row.horizons.every(cell => {
      if (!record(cell) || !integer(cell.days) || !COHORT_DAYS.includes(cell.days as 1 | 7 | 14 | 30) || days.has(cell.days)
        || typeof cell.mature !== 'boolean' || !integer(cell.eligibleAccounts) || cell.eligibleAccounts > (row.accounts as number)
        || !integer(cell.eligibleMobileAccounts) || cell.eligibleMobileAccounts > (row.mobileLinkedAccounts as number)
        || !money(cell.observedFeeRevenueUsd) || !money(cell.observedFeeRevenuePerAccountUsd)) return false
      days.add(cell.days)
      if (!hasActivityCoverage && (cell.eligibleMobileAccounts !== 0 || cell.appReturningAccounts !== null || cell.appReturnRate !== null)) return false
      if (!cell.mature && (cell.eligibleAccounts !== 0 || cell.eligibleMobileAccounts !== 0 || cell.observedFeeRevenueUsd !== null || cell.observedFeeRevenuePerAccountUsd !== null)) return false
      return [['fundedAccounts', 'fundingRate'], ['firstTradeAccounts', 'firstTradeRate'], ['appReturningAccounts', 'appReturnRate'], ['tradingReturningAccounts', 'tradingReturnRate']].every(([countKey, rateKey]) => {
        const count = cell[countKey!], rate = cell[rateKey!]
        const denominator = countKey === 'appReturningAccounts' ? cell.eligibleMobileAccounts as number : cell.eligibleAccounts as number
        if (!cell.mature && (count !== null || rate !== null)) return false
        if (count === null) return rate === null
        if (!integer(count) || count > denominator) return false
        return denominator === 0 ? rate === null : typeof rate === 'number' && Number.isFinite(rate) && Math.abs(rate - count / denominator) < 0.000001
      })
    })
  })
}

export async function fetchAccountMetrics({ from, to }: { from: string; to: string }): Promise<AccountMetricsResult> {
  const secret = process.env.ANALYTICS_FUNNEL_READ_SECRET
  if (!secret) return { ok: false, reason: 'not_configured' }
  const base = process.env.TRADING_API_BASE_URL ?? 'https://trading-api.freeportmarkets.com'
  try {
    const response = await fetch(`${base}/v1/analytics/accounts?${new URLSearchParams({ from, to })}`, {
      headers: { 'x-funnel-secret': secret }, signal: AbortSignal.timeout(10_000), next: { revalidate: 60 },
    })
    if (!response.ok) return { ok: false, reason: response.status === 401 ? 'unauthorized' : response.status === 400 ? 'bad_range' : response.status === 404 || response.status === 501 ? 'not_deployed' : 'unavailable' }
    const data: unknown = await response.json()
    return isAccountMetrics(data, from, to) ? { ok: true, data } : { ok: false, reason: 'invalid_response' }
  } catch {
    return { ok: false, reason: 'unavailable' }
  }
}
