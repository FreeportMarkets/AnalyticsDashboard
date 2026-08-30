import { auth } from '@/auth'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import {
  fetchFunnel,
  formatDuration,
  formatCohortRange,
  daysBehind,
  STEP_LABELS,
  type FunnelStep,
  type FunnelCoverage,
} from '@/lib/funnelApi'

export const dynamic = 'force-dynamic'

/**
 * The new-user → first-trade timeline.
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
 * account exists). This page is keyed on device identity.
 */

function first(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v
}

const pct = (f: number) => `${(f * 100).toFixed(1)}%`

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

function Bar({ fraction, muted = false }: { fraction: number; muted?: boolean }) {
  return (
    <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-surface">
      <div
        className={`h-full rounded-full ${muted ? 'bg-accent-bar/40' : 'bg-accent-bar'}`}
        style={{ width: `${Math.max(fraction * 100, fraction > 0 ? 1.5 : 0)}%` }}
      />
    </div>
  )
}

/**
 * One funnel. Shared by the matured block and the provisional one so the two
 * can never drift into showing the same numbers differently — the only visual
 * difference is `muted`, which is what marks a block as still filling.
 */
function FunnelTable({
  steps,
  muted = false,
}: {
  steps: FunnelStep[]
  muted?: boolean
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-hairline text-left text-xs uppercase tracking-wide text-ink-2">
            <th className="pb-2 pr-4 font-medium">Step</th>
            <th className="pb-2 pr-4 font-medium">Reached</th>
            <th className="w-[28%] pb-2 pr-4 font-medium">Share</th>
            <th className="pb-2 pr-4 text-right font-medium tabular-nums">Drop</th>
            <th className="pb-2 pr-4 text-right font-medium tabular-nums">Median leg</th>
            <th className="pb-2 text-right font-medium tabular-nums">p90 leg</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((step: FunnelStep, i: number) => {
            const prev = i > 0 ? steps[i - 1] : null
            // Drop measured against the previous step, which is what the
            // eye compares. It can be NEGATIVE: a device that skipped the
            // step above still counts here, and that gain is the skip
            // rate rather than an error.
            const drop = prev ? prev.conversion - step.conversion : 0
            return (
              <tr key={step.step_key} className="border-b border-hairline/50">
                <td className="py-3 pr-4 text-ink-1">
                  {STEP_LABELS[step.step_key] ?? step.step_key}
                  <div className="text-xs text-ink-2">{step.step_key}</div>
                </td>
                <td className="py-3 pr-4 tabular-nums text-ink-1">
                  {step.users_reached.toLocaleString('en-US')}
                </td>
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-3">
                    <Bar fraction={step.conversion} muted={muted} />
                    <span className="w-14 shrink-0 text-right tabular-nums text-ink-1">
                      {pct(step.conversion)}
                    </span>
                  </div>
                </td>
                <td className="py-3 pr-4 text-right tabular-nums text-ink-2">
                  {i === 0 ? '—' : drop > 0 ? `−${pct(drop)}` : drop < 0 ? `+${pct(-drop)}` : '0%'}
                </td>
                <td className="py-3 pr-4 text-right tabular-nums text-ink-1">
                  {formatDuration(step.p50_ms)}
                </td>
                <td className="py-3 text-right tabular-nums text-ink-2">
                  {formatDuration(step.p90_ms)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default async function JourneyPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const [session, params] = [await auth(), await searchParams]
  if (!session) return null

  const windowKey = first(params.window) === '24h' ? '24h' : '7d'
  const days = Number(first(params.days)) || 30
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
  // Percentages off a handful of people are noise, and the 7d window sits here
  // for its first week of life. Say so rather than letting someone quote 33%.
  const thinData = Boolean(coverage && coverage.cohort_days > 0 && coverage.cohort_days < 5)
  const headingMeta = [
    `${cohortSize.toLocaleString('en-US')} new users`,
    rangeLabel,
    `followed ${windowKey} each`,
  ]
    .filter(Boolean)
    .join(' · ')

  // Presets write an explicit cohort-date range. "All" drops it and falls back
  // to the trailing-`days` behaviour the page has always had.
  const ranges: { label: string; href: string; active: boolean }[] = [
    // "All" drops the range entirely and asks for the widest horizon the API
    // serves (MAX_DAYS = 90), so the label is not a promise it cannot keep.
    { label: 'All', from: undefined, to: undefined, days: 90 },
    { label: '14d', from: presetFrom(14), to: today(), days },
    { label: '30d', from: presetFrom(30), to: today(), days },
    { label: '90d', from: presetFrom(90), to: today(), days },
  ].map((r) => {
    const qs = new URLSearchParams({ window: windowKey, days: String(r.days) })
    if (r.from) qs.set('from', r.from)
    if (r.to) qs.set('to', r.to)
    return {
      label: r.label,
      href: `/journey?${qs.toString()}`,
      active: (from ?? undefined) === r.from && (to ?? undefined) === r.to,
    }
  })

  const pill = (active: boolean) =>
    active
      ? 'rounded-md bg-white/10 px-3 py-1.5 text-ink-1'
      : 'rounded-md px-3 py-1.5 text-ink-2 hover:text-accent'

  return (
    // Same shell as every other page: without it the heading sits flush against
    // the viewport edge and gets clipped at the top.
    <main className="mx-auto w-full max-w-[1600px] space-y-8 px-8 py-8">
      <PageHeader
        title="Journey"
        subtitle={
          <>
            How far new users get after they start the intro. Each one is followed for{' '}
            <strong className="text-ink-1">{windowKey} from their own intro start</strong> &mdash; this
            is <strong className="text-ink-1">not</strong> the last {windowKey} of activity, and the
            total below is every new user across all the days listed, added together. Counting is per
            device, so one person on two phones counts twice. Existing users and un-instrumented
            builds never enter.
          </>
        }
        right={
          <div className="flex flex-col items-end gap-1.5">
            <div className="flex gap-1 text-sm">
              {(['24h', '7d'] as const).map((w) => {
                const qs = new URLSearchParams({ window: w, days: String(days) })
                if (from) qs.set('from', from)
                if (to) qs.set('to', to)
                return (
                  <a key={w} href={`/journey?${qs.toString()}`} className={pill(w === windowKey)}>
                    {w}
                  </a>
                )
              })}
            </div>
            <div className="flex items-center gap-1 text-xs">
              <span className="pr-1 text-ink-2">Cohort dates</span>
              {ranges.map((r) => (
                <a key={r.label} href={r.href} className={pill(r.active)}>
                  {r.label}
                </a>
              ))}
            </div>
          </div>
        }
      />

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
          <span className="text-ink-1">No matured cohorts yet.</span> A cohort appears in the{' '}
          <code className="text-ink-1">24h</code> window two days after its date, and in{' '}
          <code className="text-ink-1">7d</code> eight days after — so this stays empty for the first
          days after the rollup is switched on. If it is still empty later, check that{' '}
          <code className="text-ink-1">intro_started</code> is arriving at ingest (only builds with
          the funnel instrumentation emit it).
        </div>
      )}

      {result.ok && result.data.steps.length > 0 && (
        <section className="space-y-4">
          <SectionHeading meta={headingMeta}>New user &rarr; first trade</SectionHeading>

          {/* A total summed over N days, printed without saying so, reads as
              "today". That single omission is what made this page unusable. */}
          <p className="-mt-2 text-xs leading-relaxed text-ink-2">
            <strong className="text-ink-1">
              {cohortSize.toLocaleString('en-US')} devices started the intro
            </strong>{' '}
            {rangeLabel ? (
              <>
                between <strong className="text-ink-1">{rangeLabel}</strong>
                {coverage?.cohort_days ? ` (${coverage.cohort_days} days, added together)` : null}
              </>
            ) : (
              'in the published range'
            )}
            {behind !== null && (
              <>
                {'. '}Newest day here is{' '}
                <strong className="text-ink-1">
                  {behind === 0 ? 'today' : `${behind} day${behind === 1 ? '' : 's'} old`}
                </strong>
                {behind > 0 && ` — a day only appears once everyone in it has had their full ${windowKey}.`}
              </>
            )}
          </p>

          {thinData && (
            <div className="rounded-lg border border-hairline bg-white/[0.02] p-4 text-xs leading-relaxed text-ink-2">
              <span className="text-ink-1">Not enough data to read yet.</span> This window only has{' '}
              {coverage?.cohort_days} day{coverage?.cohort_days === 1 ? '' : 's'} of cohorts in it,
              because a day has to wait a full {windowKey} before it can be included. The{' '}
              <code className="text-ink-1">7d</code> view fills in more slowly than{' '}
              <code className="text-ink-1">24h</code> for exactly this reason: it contains fewer
              matured days, and therefore a smaller total. The difference between the two headline
              numbers is <em>days included</em>, not a different population &mdash; percentages off
              this few devices are noise either way.
            </div>
          )}

          <FunnelTable steps={result.data.steps} />

          <p className="text-xs leading-relaxed text-ink-2">
            The headline number is a <strong className="text-ink-1">running total across every day
            listed above</strong>, not a count for today. Switching between {' '}
            <code className="text-ink-1">24h</code> and <code className="text-ink-1">7d</code> changes
            how long each person is followed, which changes how old a day must be to qualify — so the
            two views legitimately contain different numbers of days and different totals. Cohort =
            new users who started the intro (non-replay), so existing users and un-instrumented builds
            never enter here. Each row counts devices that reached that step within {windowKey} of
            starting the intro, whether or not they passed through the step above — people do skip deposit and trade on referral points. A step can therefore read
            higher than the one above it, and that gain is the skip rate, not a bug. Leg times are
            medians with p90 beside them: a wide gap means a subset is stuck rather than the whole
            step being slow.
          </p>
        </section>
      )}

      {/* The newest days, still filling. Deliberately its own block with its own
          total: these cohorts have not had their full window, so their later
          steps are undercounted and the two totals must never be added. */}
      {result.ok && provisional && provisional.steps.length > 0 && (
        <section className="space-y-4 rounded-lg border border-dashed border-hairline p-5">
          <SectionHeading
            meta={[
              `${provisionalSize.toLocaleString('en-US')} new users`,
              formatCohortRange(provisionalCoverage),
              'still filling',
            ]
              .filter(Boolean)
              .join(' · ')}
          >
            Too recent to be final
          </SectionHeading>

          <p className="-mt-2 text-xs leading-relaxed text-ink-2">
            These days have <strong className="text-ink-1">not had their full {windowKey}</strong>{' '}
            yet, so everyone in them still has time left. Counts here can only rise, and the later
            steps are undercounted for that reason alone.{' '}
            <strong className="text-ink-1">Do not add this total to the one above</strong>, and do not
            compare these percentages with the matured ones — the two are measured over different
            amounts of elapsed time. This is here so a campaign switched on this week is visible at
            all, not so it can be scored yet.
          </p>

          <FunnelTable steps={provisional.steps} muted />
        </section>
      )}
    </main>
  )
}
