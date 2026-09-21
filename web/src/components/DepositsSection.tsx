import { depositSummary } from '@/lib/metrics/trades'
import { SectionHeading } from './SectionHeading'
import { StatTile } from './StatTile'
import { DataTable } from './DataTable'

const count = (value: number) => value.toLocaleString('en-US')
const ratio = (value: number) => `${(value * 100).toFixed(1)}%`
const sectionClass = 'mt-8 border-t border-hairline pt-8'

export function DepositsLoading() {
  return <section aria-label="Deposit events" className={sectionClass} aria-busy="true">
    <SectionHeading>Deposit events</SectionHeading>
    <p role="status" className="mt-3 text-sm text-ink-2">Loading deposit events… Trading data is available above.</p>
    <div className="mt-4 grid h-16 grid-cols-3 gap-6" aria-hidden="true">
      {[0, 1, 2].map(key => <div key={key} className="rounded-md bg-surface" />)}
    </div>
  </section>
}

/** Independent streaming boundary: a deposit outage must not hide trading. */
export async function DepositsSection({ start, end }: { start: string; end: string }) {
  let deposits
  try {
    deposits = await depositSummary(start, end)
  } catch {
    return <section aria-label="Deposit events" className={sectionClass}>
      <SectionHeading>Deposit events</SectionHeading>
      <p role="status" className="mt-3 text-sm text-alert">Deposit events are unavailable. Trading data above is unaffected; this section will retry on the next refresh.</p>
    </section>
  }
  return <section aria-label="Deposit events" className={sectionClass}>
    <SectionHeading>Deposit events</SectionHeading>
    <p className="mt-2 max-w-3xl text-sm leading-relaxed text-ink-2">Recorded client events from the trading backend. These are event counts, not unique orders or funded accounts. Payment success can precede delivery, and some deposit event families are not included.</p>
    <div className="mt-4 grid gap-x-6 divide-y divide-hairline sm:grid-cols-3 sm:divide-x sm:divide-y-0">
      <StatTile label="Initiation events" value={deposits.initiated} format={count} />
      <div className="sm:pl-6"><StatTile label="Success events" value={deposits.success} format={count} /></div>
      <div className="sm:pl-6"><StatTile label="Error events" value={deposits.error} format={count} /></div>
    </div>
    <div className="mt-6">
      <SectionHeading as="h3">By provider</SectionHeading>
      <div className="mt-3"><DataTable rowKey={row => row.provider} rows={deposits.byProvider} columns={[
        { key: 'provider', header: 'Provider', render: row => row.provider },
        { key: 'initiated', header: 'Initiation events', align: 'right', render: row => count(row.initiated) },
        { key: 'success', header: 'Success events', align: 'right', render: row => count(row.success) },
        { key: 'error', header: 'Error events', align: 'right', render: row => count(row.error) },
        { key: 'ratio', header: 'Success / initiation events', align: 'right', render: row => row.initiated > 0 ? ratio(row.success / row.initiated) : '—' },
      ]} /></div>
      <p className="mt-2 text-xs text-ink-2">The event ratio is diagnostic and may exceed 100%; it is not a conversion rate.</p>
    </div>
  </section>
}
