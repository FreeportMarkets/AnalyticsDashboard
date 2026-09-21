import { fetchAccountMetrics } from '@/lib/accountMetricsApi'
import type { AccountMetric } from '@/lib/accountMetrics'
import { AccountCohortReport } from './AccountCohortReport'
import { SectionHeading } from './SectionHeading'

export function AccountCohortsLoading() {
  return <section aria-label="Account cohorts" className="mt-8 space-y-3 border-t border-hairline pt-8" aria-busy="true">
    <SectionHeading>Account cohorts</SectionHeading><p role="status" className="text-sm text-ink-2">Loading account cohorts and source coverage…</p>
    <div className="h-32 rounded-md bg-surface" aria-hidden="true" />
  </section>
}

export async function AccountCohortsSection({ from, to, initialMetric }: { from: string; to: string; initialMetric?: AccountMetric }) {
  const result = await fetchAccountMetrics({ from, to })
  if (result.ok) return <AccountCohortReport data={result.data} initialMetric={initialMetric} />
  const reason = result.reason === 'not_deployed' ? 'The versioned account measurement endpoint is not deployed yet.'
    : result.reason === 'not_configured' || result.reason === 'unauthorized' ? 'The dashboard cannot authenticate to the account measurement source.'
    : result.reason === 'bad_range' ? 'The account source rejected this date range.'
    : result.reason === 'invalid_response' ? 'The source returned an unsupported or inconsistent metric contract.'
    : 'The account measurement source is currently unavailable.'
  return <section aria-label="Account cohorts" className="mt-8 space-y-3 border-t border-hairline pt-8">
    <SectionHeading>Account cohorts</SectionHeading>
    <p role="status" className="text-sm text-alert">{reason}</p>
    <p className="max-w-3xl text-sm leading-relaxed text-ink-2">Account retention, verified card funding, first-trade conversion and observed cohort fees are unavailable. The legacy event mirror is not a substitute for current mobile data. Expected LTV and acquisition cost are also unavailable.</p>
  </section>
}
