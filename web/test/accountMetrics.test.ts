import { afterEach, describe, expect, it, vi } from 'vitest'
import { createElement } from 'react'
import { createRequire } from 'node:module'
import { accountFixture, horizon, immature } from './fixtures/accountMetrics'
import { summarizeHorizon, unavailableCellLabel } from '@/lib/accountMetrics'
import { fetchAccountMetrics, isAccountMetrics } from '@/lib/accountMetricsApi'
import { AccountCohortReport } from '@/components/AccountCohortReport'
import { AccountCohortsSection } from '@/components/AccountCohortsSection'
const { renderToStaticMarkup } = createRequire(import.meta.url)('react-dom/server')
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals() })

describe('account measurement contract', () => {
  it('accepts v1 with explicit immature nulls and rejects invented future zeros', () => {
    const data = accountFixture()
    expect(isAccountMetrics(data, data.range.from, data.range.to)).toBe(true)
    data.rows[0]!.horizons[1]!.appReturningAccounts = 0
    expect(isAccountMetrics(data, data.range.from, data.range.to)).toBe(false)
  })
  it('rejects wrong version, requested range, cohort basis and impossible numerators', () => {
    const data = accountFixture()
    for (const changed of [{ ...data, schemaVersion: 2 }, { ...data, cohortBasis: 'first_seen' }, { ...data, range: { from: '2026-09-14', to: data.range.to } }]) expect(isAccountMetrics(changed, data.range.from, data.range.to)).toBe(false)
    data.rows[0]!.horizons[0]!.fundedAccounts = 11
    expect(isAccountMetrics(data, data.range.from, data.range.to)).toBe(false)
  })
  it('weights mature observed denominators and uses mobile denominators for app return', () => {
    const rows = [
      { observedFundedAccounts: null, observedFirstTradeAccounts: null, cohortDate: '2026-09-01', accounts: 1, mobileLinkedAccounts: 1, horizons: [horizon({ eligibleAccounts: 1, eligibleMobileAccounts: 1, appReturningAccounts: 1 })] },
      { observedFundedAccounts: null, observedFirstTradeAccounts: null, cohortDate: '2026-09-02', accounts: 99, mobileLinkedAccounts: 9, horizons: [horizon({ eligibleAccounts: 99, eligibleMobileAccounts: 9, appReturningAccounts: 1 })] },
      { observedFundedAccounts: null, observedFirstTradeAccounts: null, cohortDate: '2026-09-21', accounts: 1000, mobileLinkedAccounts: 1000, horizons: [immature(1)] },
    ]
    expect(summarizeHorizon(rows, 1, 'appReturn')).toEqual({ numerator: 2, denominator: 10, includedCohorts: 2, rate: 0.2 })
    expect(summarizeHorizon(rows, 7, 'appReturn').rate).toBeNull()
  })
  it('renders immature and missing-source cells without zero-retention claims or LTV forecasts', () => {
    const data = accountFixture()
    const html = renderToStaticMarkup(createElement(AccountCohortReport, { data }))
    expect(html).toContain('Still observing')
    expect(html).toContain('1 / 5')
    expect(html).toContain('Expected LTV — unavailable')
    expect(html).toContain('CAC and ROAS — unavailable')
    expect(html).not.toContain('>0.0%<')
    data.coverage.accountsAsOf = null
    data.rows = []
    const missing = renderToStaticMarkup(createElement(AccountCohortReport, { data }))
    expect(missing).toContain('Account counts are unavailable')
    expect(missing).not.toContain('0</span> non-guest accounts')
  })
  it('shows current progress without mixing it into fixed-age denominators', () => {
    const data = accountFixture()
    data.rows[0]!.observedFundedAccounts = 4
    const html = renderToStaticMarkup(createElement(AccountCohortReport, { data, initialMetric: 'funding' }))
    expect(html).toContain('Observed so far')
    expect(html).toContain('not D1/D7 conversion rates')
    expect(html).toContain('20.0%')
    expect(html).toContain('Still observing')
    data.rows[0]!.observedFundedAccounts = null
    expect(isAccountMetrics(data, data.range.from, data.range.to)).toBe(true)
    expect(renderToStaticMarkup(createElement(AccountCohortReport, { data, initialMetric: 'funding' }))).toContain('Source unavailable')
  })
  it('labels a measured day without eligible early-linked installs correctly', () => {
    expect(unavailableCellLabel(accountFixture().rows[0]!, horizon({ eligibleMobileAccounts: 0, appReturningAccounts: 0 }), 'appReturn')).toBe('No eligible linked accounts')
  })
  it('distinguishes missing mobile links from missing source coverage', () => {
    const row = accountFixture().rows[0]!
    expect(unavailableCellLabel({ ...row, mobileLinkedAccounts: 0 }, horizon(), 'appReturn')).toBe('No linked accounts')
    expect(unavailableCellLabel(row, horizon(), 'appReturn')).toBe('Source unavailable')
  })
  it('maps a deployment gap to unavailable copy without falling back to Neon', async () => {
    vi.stubEnv('ANALYTICS_FUNNEL_READ_SECRET', 'test-only')
    const request = vi.fn<typeof fetch>(async () => new Response('{}', { status: 404 }))
    vi.stubGlobal('fetch', request)
    const html = renderToStaticMarkup(await AccountCohortsSection({ from: '2026-09-15', to: '2026-09-21' }))
    expect(html).toContain('not deployed yet')
    expect(html).toContain('legacy event mirror is not a substitute')
    expect(request).toHaveBeenCalledOnce()
    expect(request.mock.calls[0]?.[0]).toContain('/v1/analytics/accounts?from=2026-09-15&to=2026-09-21')
  })
  it('sends authentication only server-side and preserves valid response nulls', async () => {
    const data = accountFixture()
    vi.stubEnv('ANALYTICS_FUNNEL_READ_SECRET', 'test-only')
    const request = vi.fn<typeof fetch>(async () => Response.json(data))
    vi.stubGlobal('fetch', request)
    const result = await fetchAccountMetrics(data.range)
    expect(result).toEqual({ ok: true, data })
    expect(request.mock.calls[0]?.[1]).toMatchObject({ headers: { 'x-funnel-secret': 'test-only' } })
  })
})
