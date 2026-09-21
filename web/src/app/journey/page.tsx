import { Suspense } from 'react'
import { AccountPopulationControls } from '@/components/AccountPopulationControls'
import { ACCOUNT_POPULATIONS, type AccountPopulation } from '@/lib/accountMetrics'
import { JourneyDaily } from '@/components/JourneyDaily'
import { AccountCohortsLoading } from '@/components/AccountCohortsSection'
import { auth } from '@/auth'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { JourneyMilestones, JourneyTiming } from '@/components/JourneyMilestones'
import { journeyPreset } from '@/lib/journeyRanges'
import {
  fetchFunnel,
  formatCohortRange,
  daysBehind,
  type FunnelCoverage,
} from '@/lib/funnelApi'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

/** Legacy device telemetry is diagnostic; account metrics use the separate model. */

function first(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v
}

async function IntroDiagnostics({ params }: { params: Record<string, string | string[] | undefined> }) {
  const windowKey = first(params.window) === '24h' ? '24h' : '7d'
  const days = Number(first(params.days)) || 30
  const order = first(params.order) === 'journey' ? 'journey' : 'reach'
  // Forwarded RAW, deliberately. Silently dropping a malformed date here would
  // render a confident-looking funnel for the default range instead of the
  // dedicated bad_range state — the same "plausible number for the wrong dates"
  // failure the API rejects 400 to avoid. The API is the single validator.
  const from = first(params.from)
  const to = first(params.to)
  const accountFrom = first(params.accountFrom)
  const accountTo = first(params.accountTo)
  const preserveAccountRange = (qs: URLSearchParams) => {
    if (accountFrom) qs.set('accountFrom', accountFrom)
    if (accountTo) qs.set('accountTo', accountTo)
    if (first(params.population)) qs.set('population', first(params.population)!)
    return qs
  }

  const result = await fetchFunnel({ window: windowKey, days, from, to })

  const coverage = result.ok ? result.data.coverage : undefined
  const rangeLabel = formatCohortRange(coverage)
  const behind = daysBehind(coverage)
  const cohortSize = result.ok ? (result.data.steps[0]?.cohort_size ?? 0) : 0
  // The unmatured tail. Rendered as its own block and NEVER added to the
  // headline: those users have not had their window yet, so their later steps
  // are undercounted and a blended total would belong to no real population.
  const provisional = result.ok ? result.data.provisional : null
  const provisionalCoverage: FunnelCoverage | undefined = provisional?.coverage
  const provisionalSize = provisional?.steps[0]?.cohort_size ?? 0
  // Date coverage is a separate limitation from sample size.
  const thinData = Boolean(coverage && coverage.cohort_days > 0 && coverage.cohort_days < 5)
  const headingMeta = [
    `${cohortSize.toLocaleString('en-US')} devices`,
    rangeLabel,
    `followed ${windowKey} each`,
  ]
    .filter(Boolean)
    .join(' · ')

  // Keep each API's calendar explicit, including between UTC and NY midnight.
  const presetNow = new Date()
  const ranges: { label: string; href: string; active: boolean }[] = [
    { label: '14d', ...journeyPreset(14, presetNow), days },
    { label: '30d', ...journeyPreset(30, presetNow), days },
    { label: '90d', ...journeyPreset(90, presetNow), days },
  ].map((r) => {
    const qs = new URLSearchParams({ window: windowKey, days: String(r.days), order })
    if (r.from) qs.set('from', r.from)
    if (r.to) qs.set('to', r.to)
    qs.set('accountFrom', r.accountFrom)
    qs.set('accountTo', r.accountTo)
    if (first(params.population)) qs.set('population', first(params.population)!)
    return {
      label: r.label,
      href: `/journey?${qs.toString()}`,
      active: (from === r.from && to === r.to) || (!from && !to && Number(r.label.slice(0, -1)) === days),
    }
  })

  const pill = (active: boolean) =>
    `inline-flex min-h-11 items-center rounded-md px-3 py-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${active ? 'bg-raised text-ink-1' : 'text-ink-2 hover:bg-surface hover:text-accent'}`

  const orderHref = (next: string) => {
    const qs = preserveAccountRange(new URLSearchParams({ window: windowKey, days: String(days), order: next }))
    if (from) qs.set('from', from)
    if (to) qs.set('to', to)
    return `/journey?${qs.toString()}`
  }

  return (
    <>
      <section aria-label="Intro cohort filters" className="flex flex-wrap items-start justify-between gap-4 border-t border-hairline pt-8">
        <div><h2 className="text-lg font-semibold text-ink-1">Device intro diagnostics</h2><p className="mt-1 max-w-3xl text-sm text-ink-2">Legacy device-level event reach. Restored sessions, replay markers and instrumentation changes can affect who is included. These are not new-account conversion, verified funding, or retention metrics.</p></div>
          <div className="flex flex-col gap-1.5">
            <div className="flex flex-wrap items-center gap-1 text-sm">
              <span className="pr-2 text-xs text-ink-2">Observation window</span>
              {(['24h', '7d'] as const).map((w) => {
                const qs = preserveAccountRange(new URLSearchParams({ window: w, days: String(days), order }))
                if (from) qs.set('from', from)
                if (to) qs.set('to', to)
                return (
                  <a key={w} href={`/journey?${qs.toString()}`} className={pill(w === windowKey)} aria-current={w === windowKey ? 'true' : undefined}>
                    {w}
                  </a>
                )
              })}
            </div>
            <div className="flex flex-wrap items-center gap-1 text-xs">
              <span className="pr-1 text-ink-2">Intro start dates (UTC)</span>
              {ranges.map((r) => (
                <a key={r.label} href={r.href} className={pill(r.active)} aria-current={r.active ? 'true' : undefined}>
                  {r.label}
                </a>
              ))}
            </div>
          </div>
      </section>

      {!result.ok && (
        <div className="rounded-lg border border-hairline p-6 text-sm text-ink-2">
          {result.reason === 'not_configured' && (
            <>
              <span className="text-ink-1">Not configured.</span> Set{' '}
              <code className="text-ink-1">ANALYTICS_FUNNEL_READ_SECRET</code> in this project&apos;s
              environment to the value stored in Secrets Manager.
            </>
          )}
          {result.reason === 'unauthorized' && (
            <>
              <span className="text-ink-1">Rejected by the API.</span> The configured secret does not
              match the one on the trading backend.
            </>
          )}
          {result.reason === 'bad_range' && (
            <>
              <span className="text-ink-1">That date range cannot be read.</span> Dates must be{' '}
              <code className="text-ink-1">YYYY-MM-DD</code>, must exist on the calendar, and{' '}
              <code className="text-ink-1">from</code> must not come after{' '}
              <code className="text-ink-1">to</code>. Pick a preset above to reset it.
            </>
          )}
          {result.reason === 'unreachable' && (
            <>
              <span className="text-ink-1">Backend unreachable.</span> The rollup itself is
              unaffected — it writes on its own schedule regardless of this page.
            </>
          )}
        </div>
      )}

      {result.ok && result.data.steps.length === 0 && (
        <div className="rounded-lg border border-hairline p-6 text-sm text-ink-2">
          <span className="text-ink-1">No completed observation windows in this range.</span>{' '}
          Try a wider intro date range or the 24-hour window. Recent starters appear below while
          their observation windows are still open.
        </div>
      )}

      {result.ok && result.data.steps.length > 0 && (
        <section className="space-y-5" aria-label="Completed observation windows">
          <SectionHeading meta={headingMeta}>Recorded intro milestone reach</SectionHeading>
          <p className="text-sm leading-relaxed text-ink-2">
            Each included device has a completed {windowKey === '24h' ? '24-hour' : '7-day'} observation window.
            {coverage?.cohort_days ? ` Totals cover ${coverage.cohort_days} intro start dates (UTC).` : ' Actual intro date coverage was not supplied.'}
            {behind !== null && ` The newest included start date is ${behind === 0 ? 'today' : `${behind} days ago`}.`}
          </p>
          {thinData && (
            <p className="rounded-lg border border-hairline bg-surface/30 p-4 text-sm text-ink-2">
              Limited date coverage: only {coverage?.cohort_days} intro start dates have completed
              this observation window. Check a wider range before treating these results as a trend.
            </p>
          )}
          <div className="space-y-4 pt-3">
            <SectionHeading right={
              <nav aria-label="Milestone order" className="flex flex-wrap gap-1 text-sm">
                <a href={orderHref('reach')} className={pill(order === 'reach')} aria-current={order === 'reach' ? 'true' : undefined}>Most reached</a>
                <a href={orderHref('journey')} className={pill(order === 'journey')} aria-current={order === 'journey' ? 'true' : undefined}>Journey order</a>
              </nav>
            }>Milestone reach</SectionHeading>
            <p className="max-w-4xl text-sm leading-relaxed text-ink-2">
              {order === 'reach' ? 'Sorted by device count, highest first. ' : 'Grouped in the reported journey order. '}
              Each milestone is counted independently within the same intro cohort. People can take
              different paths, so the difference between rows is not a drop-off rate.
            </p>
            <JourneyMilestones steps={result.data.steps} order={order} />
          </div>
          <JourneyTiming steps={result.data.steps} />
          <details className="rounded-xl border border-hairline p-5 text-sm text-ink-2">
            <summary className="cursor-pointer rounded font-medium text-ink-1 focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-4">How to read these numbers</summary>
            <div className="mt-4 max-w-3xl space-y-3 leading-relaxed">
              <p>A true step-by-step funnel counts only devices that completed every prior step in order. Its counts can only stay the same or decrease. This view receives separate milestone totals, which cannot reveal the overlap between steps or the route each device took.</p>
              <p>For example, a transfer can add funds without a payment checkout. Referral points can let someone trade without depositing. More traders than depositors does not tell us exactly how many skipped a deposit.</p>
              <p>Counts represent devices whose first non-replay intro start falls in the included UTC dates. One person on two devices can count twice. Missing or incorrect replay markers can include returning users. Builds without this tracking are absent; instrumentation changes limit comparisons. Deposit and trade milestones are recorded client telemetry, not the verified account ledger.</p>
              <p>The 24-hour and 7-day views can include different intro dates because each cohort must finish its observation window. Compare the date coverage before comparing conversion rates. Recent starters below remain separate from completed windows.</p>
            </div>
          </details>
        </section>
      )}

      {/* The newest days, still filling. Deliberately its own block with its own
          total: these cohorts have not had their full window, so their later
          steps are undercounted and the two totals must never be added. */}
      {result.ok && provisional && provisional.steps.length > 0 && (
        <section className="space-y-4 rounded-lg border border-dashed border-hairline p-5">
          <SectionHeading
            meta={[
              `${provisionalSize.toLocaleString('en-US')} devices`,
              formatCohortRange(provisionalCoverage),
              'still filling',
            ]
              .filter(Boolean)
              .join(' · ')}
          >
            Recent starters · still observing
          </SectionHeading>

          <p className="text-sm leading-relaxed text-ink-2">
            These devices still have time left in their {windowKey === '24h' ? '24-hour' : '7-day'} observation
            window. Their current activity is shown separately; their recorded reach is incomplete and must not be scored as a final outcome.
          </p>
          <details>
            <summary className="cursor-pointer rounded py-2 text-sm font-medium text-ink-1 focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-4">View activity so far</summary>
            <div className="mt-3"><JourneyMilestones steps={provisional.steps} order={order} /></div>
          </details>
        </section>
      )}
    </>
  )
}


export default async function JourneyPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const [session, params] = await Promise.all([auth(), searchParams])
  if (!session?.user) return null
  const population = first(params.population) ?? 'all'
  const validPopulation = Object.hasOwn(ACCOUNT_POPULATIONS, population)
  return <main className="mx-auto w-full max-w-[1600px] space-y-8 px-4 py-8 sm:px-8">
    <PageHeader title="Journey" subtitle="Account creation cohorts and independent device intro diagnostics." />
    <AccountPopulationControls params={params} population={population} />
    <Suspense key={`accounts:${first(params.accountFrom)}:${first(params.accountTo)}:${population}`} fallback={<AccountCohortsLoading />}>
      {validPopulation ? <JourneyDaily from={first(params.accountFrom)} to={first(params.accountTo)} population={population as AccountPopulation} /> : <p role="status" className="text-sm text-alert">Unknown account population. Choose a population above to load its report.</p>}
    </Suspense>
    <Suspense key={`intro:${JSON.stringify(params)}`} fallback={<p role="status" className="text-sm text-ink-2">Loading device intro diagnostics…</p>}>
      <IntroDiagnostics params={params} />
    </Suspense>
  </main>
}
