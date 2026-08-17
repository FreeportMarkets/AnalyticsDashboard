import { auth } from '@/auth'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { fetchFunnel, formatDuration, STEP_LABELS, type FunnelStep } from '@/lib/funnelApi'

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

function Bar({ fraction }: { fraction: number }) {
  return (
    <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-surface">
      <div
        className="h-full rounded-full bg-accent-bar"
        style={{ width: `${Math.max(fraction * 100, fraction > 0 ? 1.5 : 0)}%` }}
      />
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

  const result = await fetchFunnel({ window: windowKey, days })

  return (
    // Same shell as every other page: without it the heading sits flush against
    // the viewport edge and gets clipped at the top.
    <main className="mx-auto w-full max-w-[1600px] space-y-8 px-8 py-8">
      <PageHeader
        title="Journey"
        subtitle={
          <>
            New user &rarr; first trade. Cohort = devices whose first (non-replay) intro start landed
            in the last {days} days and have had a full {windowKey} to convert. Existing users and
            builds without the funnel instrumentation are excluded by construction.
          </>
        }
        right={
          <div className="flex gap-1 text-sm">
            {(['24h', '7d'] as const).map((w) => (
              <a
                key={w}
                href={`/journey?window=${w}&days=${days}`}
                className={
                  w === windowKey
                    ? 'rounded-md bg-white/10 px-3 py-1.5 text-ink-1'
                    : 'rounded-md px-3 py-1.5 text-ink-2 hover:text-accent'
                }
              >
                {w}
              </a>
            ))}
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
          <SectionHeading meta={`${result.data.steps[0]?.cohort_size.toLocaleString('en-US')} new users`}>
            New user &rarr; first trade
          </SectionHeading>

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
                {result.data.steps.map((step: FunnelStep, i: number) => {
                  const prev = i > 0 ? result.data.steps[i - 1] : null
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
                          <Bar fraction={step.conversion} />
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

          <p className="text-xs leading-relaxed text-ink-2">
            Cohort = new users who started the intro (non-replay), so existing users and
            un-instrumented builds never enter here. Each row counts devices that reached that step
            within {windowKey} of starting the intro, whether or not they passed through the step
            above — people do skip deposit and trade on referral points. A step can therefore read
            higher than the one above it, and that gain is the skip rate, not a bug. Leg times are
            medians with p90 beside them: a wide gap means a subset is stuck rather than the whole
            step being slow.
          </p>
        </section>
      )}
    </main>
  )
}
