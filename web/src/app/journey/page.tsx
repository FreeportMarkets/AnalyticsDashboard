import { Suspense } from 'react'
import { JourneyDaily } from '@/components/JourneyDaily'
import { auth } from '@/auth'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { JourneyMilestones, JourneyOutcome, JourneyTiming } from '@/components/JourneyMilestones'
import {
  fetchFunnel,
  formatCohortRange,
  daysBehind,
  type FunnelCoverage,
} from '@/lib/funnelApi'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

/**
 * Daily account growth plus the intro-device → first-trade cohort.
 *
 * Starts at 100% of a day's NEW USERS — devices whose first non-replay
 * `intro_started` landed that day — and shows how far they got, with the time
 * each leg took. The cohort is deliberately NOT `app_first_open`: that fires on
 * every build (including ones predating the funnel instrumentation) and on
 * existing users reinstalling, which polluted the top of the funnel. Only
 * instrumented builds emit `intro_started`, and only a genuinely-new user emits
 * it non-replay, so this cohort excludes both by construction. See
 * docs/analytics-funnel.md in freeport-trading-backend.
 *
 * Distinct from /funnels, which composes arbitrary sequences from the synced
 * `events` table (keyed on `wallet_address`, so it cannot see anything before an
 * account exists). Only the intro cohort is keyed on device identity;
 * the daily chart counts accounts and uses New York calendar days.
 */

function first(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v
}

/**
 * Cohort dates are UTC days.
 *
 * `cohort_date` is derived from `server_ts` in UTC by the rollup, and the rest
 * of this page already reads them that way (`formatCohortRange` parses at UTC
 * noon on purpose). Deriving presets in local time instead would shift the
 * bounds by a day for anyone west of Greenwich and, between 00:00 UTC and local
 * midnight, would ask for a `to` that excludes the newest cohort — the exact
 * blind spot the provisional block exists to close.
 */
const today = () => new Date().toISOString().slice(0, 10)
const daysAgo = (n: number) =>
  new Date(Date.now() - n * 86_400_000).toISOString().slice(0, 10)

/**
 * A preset labelled "14d" must select 14 cohort dates, not 15.
 *
 * The API's range is inclusive at BOTH ends, so `from = today - 14` through
 * `to = today` is 15 calendar days. Offsetting by n-1 makes the label true.
 */
const presetFrom = (n: number) => daysAgo(n - 1)

export default async function JourneyPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const [session, params] = [await auth(), await searchParams]
  if (!session) return null

  const windowKey = first(params.window) === '24h' ? '24h' : '7d'
  const days = Number(first(params.days)) || 30
  const order = first(params.order) === 'journey' ? 'journey' : 'reach'
  // Forwarded RAW, deliberately. Silently dropping a malformed date here would
  // render a confident-looking funnel for the default range instead of the
  // dedicated bad_range state — the same "plausible number for the wrong dates"
  // failure the API rejects 400 to avoid. The API is the single validator.
  const from = first(params.from)
  const to = first(params.to)

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

  // Date presets are inclusive UTC cohort dates, not activity dates.
  const ranges: { label: string; href: string; active: boolean }[] = [
    { label: '14d', from: presetFrom(14), to: today(), days },
    { label: '30d', from: presetFrom(30), to: today(), days },
    { label: '90d', from: presetFrom(90), to: today(), days },
  ].map((r) => {
    const qs = new URLSearchParams({ window: windowKey, days: String(r.days), order })
    if (r.from) qs.set('from', r.from)
    if (r.to) qs.set('to', r.to)
    return {
      label: r.label,
      href: `/journey?${qs.toString()}`,
      active: (from === r.from && to === r.to) || (!from && !to && Number(r.label.slice(0, -1)) === days),
    }
  })

  const pill = (active: boolean) =>
    `inline-flex min-h-11 items-center rounded-md px-3 py-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${active ? 'bg-raised text-ink-1' : 'text-ink-2 hover:bg-surface hover:text-accent'}`

  const orderHref = (next: string) => {
    const qs = new URLSearchParams({ window: windowKey, days: String(days), order: next })
    if (from) qs.set('from', from)
    if (to) qs.set('to', to)
    return `/journey?${qs.toString()}`
  }

  return (
    // Same shell as every other page: without it the heading sits flush against
    // the viewport edge and gets clipped at the top.
    <main className="mx-auto w-full max-w-[1600px] space-y-8 px-4 py-8 sm:px-8">
      <PageHeader
        title="Journey"
        subtitle={
          <>
            Track daily signups and first actions, then explore conversion after the intro.
          </>
        }
      />

      <Suspense fallback={<section aria-label="Daily growth" className="rounded-xl border border-hairline p-5 text-sm text-ink-2"><p role="status">Loading daily signups and first actions…</p></section>}>
        <JourneyDaily />
      </Suspense>

      <section aria-label="Intro cohort filters" className="flex flex-wrap items-start justify-between gap-4 border-t border-hairline pt-8">
        <div><h2 className="text-lg font-semibold text-ink-1">Intro cohort analysis</h2><p className="mt-1 text-sm text-ink-2">Follow each device for a full observation window after it starts the intro.</p></div>
          <div className="flex flex-col gap-1.5">
            <div className="flex flex-wrap items-center gap-1 text-sm">
              <span className="pr-2 text-xs text-ink-2">Time to convert</span>
              {(['24h', '7d'] as const).map((w) => {
                const qs = new URLSearchParams({ window: w, days: String(days), order })
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
          <SectionHeading meta={headingMeta}>Intro to first trade</SectionHeading>
          <p className="text-sm leading-relaxed text-ink-2">
            Everyone here has had the full {windowKey === '24h' ? '24 hours' : '7 days'} to convert.
            {coverage?.cohort_days ? ` Totals cover ${coverage.cohort_days} intro start dates (UTC).` : ' Actual intro date coverage was not supplied.'}
            {behind !== null && ` The newest included start date is ${behind === 0 ? 'today' : `${behind} days ago`}.`}
          </p>
          <JourneyOutcome steps={result.data.steps} windowLabel={windowKey === '24h' ? '24 hours' : '7 days'} />
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
              <p>Counts represent devices whose first non-replay intro start falls in the included UTC dates. One person on two devices can count twice. Existing users and builds without this tracking are excluded.</p>
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
            window. Their current activity is shown separately; it is too early to score their final conversion.
          </p>
          <details>
            <summary className="cursor-pointer rounded py-2 text-sm font-medium text-ink-1 focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-4">View activity so far</summary>
            <div className="mt-3"><JourneyMilestones steps={provisional.steps} order={order} /></div>
          </details>
        </section>
      )}
    </main>
  )
}
