import { fetchJourneyAccounts } from '@/lib/privyAccounts'
import { firstAccountActions, buildDailyJourney, type FirstAction } from '@/lib/metrics/journeyDaily'
import { nyDateOf } from '@/lib/time'
import { addDays } from '@/lib/ranges'
import { JourneyDailyChart } from './JourneyDailyChart'
import { SectionHeading } from './SectionHeading'

export async function JourneyDaily() {
  try {
    const snapshot = await fetchJourneyAccounts()
    const today = nyDateOf(new Date())
    let actions: FirstAction[] = []
    let sourceTimes: Array<{ source: string; watermark_ts: string }> = []
    let actionsAvailable = false
    try {
      const { sql } = await import('@/lib/db')
      const results = await Promise.all([
        firstAccountActions(sql, snapshot.accounts, today),
        sql(`SELECT source, watermark_ts FROM sync_state WHERE source IN ('events', 'trades')`),
      ])
      actions = results[0]
      sourceTimes = results[1] as Array<{ source: string; watermark_ts: string }>
      actionsAvailable = sourceTimes.length === 2
    } catch {
      // A failed activity source must not turn missing conversions into zeros.
    }
    const daily = buildDailyJourney(snapshot.accounts, actions, addDays(today, -89), today)
    const time = (iso: string) => new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZone: 'America/New_York' })
    return (
      <section aria-label="Daily growth" className="space-y-4">
        <SectionHeading>Daily growth</SectionHeading>
        <JourneyDailyChart data={daily} actionsAvailable={actionsAvailable} />
        {!actionsAvailable && <p className="text-sm text-alert">Deposit and trade history is unavailable or has not finished its first sync. Signup counts remain available.</p>}
        <p className="text-xs leading-relaxed text-ink-2">Signups checked {time(snapshot.fetchedAt)} ET. {sourceTimes.map(s => `${s.source === 'events' ? 'Deposit events' : 'Trade records'} synced through ${time(s.watermark_ts)} ET.`).join(' ')} Today is incomplete; recent activity can arrive late.</p>
        <details className="rounded-xl border border-hairline p-5 text-sm text-ink-2">
          <summary className="cursor-pointer rounded font-medium text-ink-1 focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-4">Daily metric definitions</summary>
          <div className="mt-4 max-w-3xl space-y-3 leading-relaxed">
            <p>Signups count non-guest accounts by their Privy creation timestamp, including accounts without wallets. This is an account creation measure; imported or pre-created accounts can also be included. Daily buckets use America/New_York.</p>
            <p>First trades and deposits mean the earliest successful action in all stored history for each currently linked account, across its wallets. They can overstate lifetime firsts if older history is missing. Unlinked wallets are excluded. Changing the chart range does not reset a user’s first action.</p>
            <p>A successful deposit is a recorded payment success or funds-arrival event. Payment success can precede spendable funds. Trade counts require a successful trade event or a success/filled trade record; submitted, pending and failed orders do not count.</p>
            <p>“Same-day signup conversion” counts accounts created that day whose first recorded deposit or trade happened after signup and before the next New York midnight. These counts are subsets of that day’s signups. Today’s signup count may lag by about five minutes while its source refreshes.</p>
          </div>
        </details>
      </section>
    )
  } catch {
    return <section aria-label="Daily growth" className="space-y-3 rounded-xl border border-hairline p-5"><SectionHeading>Daily growth</SectionHeading><p className="text-sm text-ink-2">Daily signup data is unavailable. The account source must return a complete listing before daily counts can be shown. Check the Privy connection and try again.</p></section>
  }
}
