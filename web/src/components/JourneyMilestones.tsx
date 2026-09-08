import { formatDuration, STEP_LABELS, type FunnelStep } from '@/lib/funnelApi'
import { milestoneGroup, orderMilestones, tradeOutcome, type JourneyOrder } from '@/lib/journey'

const count = (n: number) => n.toLocaleString('en-US')
const pct = (n: number) => `${(n * 100).toFixed(1)}%`

function ReachBar({ fraction }: { fraction: number }) {
  return (
    <div aria-hidden="true" className="h-2 w-full overflow-hidden rounded-full bg-surface">
      <div className="h-full rounded-full bg-accent-bar" style={{ width: `${Math.min(100, Math.max(0, fraction * 100))}%` }} />
    </div>
  )
}

export function JourneyOutcome({ steps, windowLabel }: { steps: FunnelStep[]; windowLabel: string }) {
  const outcome = tradeOutcome(steps)
  return (
    <div className="space-y-5">
      <dl className="grid gap-4 md:grid-cols-3">
        {[
          { label: 'Started the intro', value: count(outcome.cohortSize), detail: 'Unique devices in the completed observation window' },
          { label: `Traded within ${windowLabel}`, value: outcome.traded === null ? '—' : count(outcome.traded), detail: outcome.conversion === null ? 'Trade outcome unavailable' : `${pct(outcome.conversion)} of intro starters · at least one filled trade` },
          { label: 'No recorded trade', value: outcome.withoutTrade === null ? '—' : count(outcome.withoutTrade), detail: outcome.conversion === null ? 'Requires a recorded trade outcome' : `${pct(1 - outcome.conversion)} of intro starters · within the same window` },
        ].map(item => (
          <div key={item.label} className="rounded-xl border border-hairline bg-surface/30 p-5">
            <dt className="text-sm text-ink-2">{item.label}</dt>
            <dd className="numeral mt-3 text-3xl font-semibold text-ink-1">{item.value}</dd>
            <dd className="mt-2 text-sm leading-relaxed text-ink-2">{item.detail}</dd>
          </div>
        ))}
      </dl>
      <div className="rounded-xl border border-hairline p-5">
        <h3 className="font-semibold text-ink-1">What this tells you</h3>
        <p className="mt-2 max-w-4xl text-sm leading-relaxed text-ink-2">
          {!outcome.valid ? 'The milestone counts are inconsistent with the cohort size. Check tracking and the data source before interpreting conversion.'
            : outcome.cohortSize === 0 ? 'There are no intro starters in this observation window yet.'
            : outcome.traded === null ? 'The data source has not supplied both intro and trade milestones. Conversion cannot be calculated yet.'
            : outcome.withoutTrade === 0 ? `Every intro starter has a recorded filled trade within ${windowLabel}. This measures the first trade outcome; repeat trading is a separate question.`
            : `${count(outcome.withoutTrade!)} of ${count(outcome.cohortSize)} intro starters had no recorded filled trade within ${windowLabel}. To find the cause, the next useful breakdown is: never signed in, signed in without funding, and funded without trading. These milestone totals cannot separate those groups.`}
        </p>
        {outcome.traded !== null && outcome.withoutTrade !== 0 && (
          <p className="mt-3 max-w-4xl text-sm leading-relaxed text-ink-2">
            Investigate the paths separately: checkout for users buying funds, transfers for users sending funds,
            and trading with referral points. A payment step is not required for every trader.
          </p>
        )}
      </div>
    </div>
  )
}

export function JourneyMilestones({ steps, order }: { steps: FunnelStep[]; order: JourneyOrder }) {
  const sorted = orderMilestones(steps, order)
  return (
    <div className="overflow-hidden rounded-xl border border-hairline">
      <div className="hidden grid-cols-[minmax(0,1fr)_7rem_minmax(8rem,1fr)] gap-5 border-b border-hairline bg-surface px-5 py-3 text-xs font-medium uppercase tracking-wide text-ink-2 sm:grid">
        <span>Milestone</span><span className="text-right">Devices</span><span>Share of intro starters</span>
      </div>
      <ul className="divide-y divide-hairline">
        {sorted.map(step => {
          const share = step.cohort_size > 0 ? step.users_reached / step.cohort_size : null
          return (
            <li key={step.step_key} className="grid gap-3 px-5 py-4 sm:grid-cols-[minmax(0,1fr)_7rem_minmax(8rem,1fr)] sm:items-center sm:gap-5">
              <div>
                <div className="font-medium text-ink-1">{STEP_LABELS[step.step_key] ?? step.step_key}</div>
                <div className="mt-1 text-xs text-ink-2">{milestoneGroup(step.step_key)}</div>
              </div>
              <div className="numeral text-lg text-ink-1 sm:text-right">
                {count(step.users_reached)}<span className="ml-2 font-sans text-xs text-ink-2 sm:hidden">devices</span>
              </div>
              <div className="flex items-center gap-3">
                <ReachBar fraction={share ?? 0} />
                <span className="numeral w-16 shrink-0 text-right text-sm text-ink-1">{share === null ? '—' : pct(share)}</span>
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export function JourneyTiming({ steps }: { steps: FunnelStep[] }) {
  return (
    <details className="rounded-xl border border-hairline p-5">
      <summary className="cursor-pointer rounded text-sm font-medium text-ink-1 focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-4">
        Timing &amp; event definitions
      </summary>
      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-ink-2">
        These are the step timing summaries supplied by the data source. The median is the middle
        recorded duration; p90 is the duration at or below which 90% of recorded samples fall.
        Timing sample counts are unavailable, so these values do not establish how many users were delayed or why.
      </p>
      <div role="region" aria-label="Milestone timing details" tabIndex={0} className="mt-4 overflow-x-auto focus-visible:outline-2 focus-visible:outline-accent">
        <table className="w-full min-w-[560px] text-left text-sm">
          <thead className="border-b border-hairline text-ink-2">
            <tr><th scope="col" className="py-3 font-medium">Milestone / event</th><th scope="col" className="p-3 text-right font-medium">Median duration</th><th scope="col" className="py-3 text-right font-medium">90th percentile</th></tr>
          </thead>
          <tbody>
            {orderMilestones(steps, 'journey').map(step => (
              <tr key={step.step_key} className="border-b border-hairline last:border-0">
                <th scope="row" className="py-3 font-normal text-ink-1">{STEP_LABELS[step.step_key] ?? step.step_key}<code className="mt-1 block text-xs text-ink-2">{step.step_key}</code></th>
                <td className="numeral p-3 text-right">{formatDuration(step.p50_ms)}</td>
                <td className="numeral py-3 text-right">{formatDuration(step.p90_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}
