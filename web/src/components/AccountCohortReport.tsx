'use client'

import { useState } from 'react'
import { COHORT_DAYS, metricValue, summarizeHorizon, unavailableCellLabel, type AccountMetric, type AccountMetrics, type AccountHorizon, type AccountCohort } from '@/lib/accountMetrics'
import { SectionHeading } from './SectionHeading'

const count = (value: number) => value.toLocaleString('en-US')
const usd = (value: number) => value.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 4 })
const time = (value: string | null) => value ? new Date(value).toLocaleString('en-US', { timeZone: 'America/New_York', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) + ' ET' : 'Unavailable'
export const ACCOUNT_METRICS: Record<AccountMetric, { label: string; definition: string }> = {
  appReturn: { label: 'App return', definition: 'Returned on the exact New York calendar day D1, D7, D14 or D30 after account creation. Denominator: eligible accounts linked to a production mobile install before the return day began. Missing mobile coverage stays unavailable; this is not a churn estimate.' },
  tradingReturn: { label: 'Trading return', definition: 'Recorded a confirmed trade on the exact New York calendar day after account creation. Denominator: eligible created accounts. Historical trade coverage is incomplete; inactivity is not permanent churn.' },
  funding: { label: 'Verified card funding', definition: 'At least one verified positive card onramp within the first 24 hours, 7, 14 or 30 days after account creation. Denominator: eligible created accounts. This is a lower bound: direct crypto transfers and unverified historical orders are not covered.' },
  firstTrade: { label: 'First recorded trade', definition: 'First recorded confirmed fill within the first 24 hours, 7, 14 or 30 days after account creation. Denominator: eligible created accounts. A trade can occur without a card deposit. Historical Spot and venue coverage is incomplete.' },
  revenue: { label: 'Observed fees / account', definition: 'Verified Freeport fee revenue within the first 24 hours, 7, 14 or 30 days, divided by eligible created accounts. Limited to verified Hyperliquid USDC and Relay receipts. Unknown fees are not zero; this is partial gross revenue, not profit or expected LTV.' },
}

function CohortCell({ row, cell, metric }: { row: AccountCohort; cell: AccountHorizon | undefined; metric: AccountMetric }) {
  const point = cell ? metricValue(cell, metric) : null
  if (!cell?.mature || point?.rate === null || !point) return <span className="text-xs text-ink-2">{unavailableCellLabel(row, cell, metric)}</span>
  return <>
    <span className="numeral block font-medium">{metric === 'revenue' ? usd(point.rate) : `${(point.rate * 100).toFixed(1)}%`}</span>
    <span className="numeral mt-1 block text-xs text-ink-2">{metric === 'revenue' ? `${usd(point.value!)} / ${count(point.denominator)} accounts` : `${count(point.value!)} / ${count(point.denominator)}`}</span>
  </>
}

export function AccountCohortReport({ data, initialMetric = 'appReturn' }: { data: AccountMetrics; initialMetric?: AccountMetric }) {
  const [metric, setMetric] = useState(initialMetric)
  const progressKey = metric === 'funding' ? 'observedFundedAccounts' : metric === 'firstTrade' ? 'observedFirstTradeAccounts' : null
  const progressRows = progressKey ? data.rows.filter(row => row[progressKey] !== null) : []
  const progressCount = progressKey ? progressRows.reduce((total, row) => total + (row[progressKey] ?? 0), 0) : null
  const accounts = data.rows.reduce((total, row) => total + row.accounts, 0)
  const mobile = data.rows.reduce((total, row) => total + row.mobileLinkedAccounts, 0)
  return <section aria-label="Account cohorts" className="mt-8 space-y-5 border-t border-hairline pt-8">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><SectionHeading>Account cohorts</SectionHeading><p className="mt-1 text-sm text-ink-2">Created {data.range.from} – {data.range.to} · New York time</p></div>
      <span className="rounded-md border border-hairline px-3 py-1.5 text-sm text-alert">{data.coverage.status === 'stale' ? 'Sources delayed' : data.coverage.status === 'unavailable' ? 'Source coverage unavailable' : 'Partial coverage'}</span>
    </div>
    <p className="max-w-3xl text-sm leading-relaxed text-ink-2">{data.coverage.accountsAsOf ? <><span className="numeral text-ink-1">{count(accounts)}</span> non-guest accounts · <span className="numeral text-ink-1">{count(mobile)}</span> linked to mobile installs. </> : <>Account counts are unavailable until a complete snapshot is recorded. </>}Account creation includes web, imported and walletless accounts. This is not an install or mobile-signup cohort.</p>
    <div className="flex flex-wrap gap-1" aria-label="Cohort metric">
      {(Object.entries(ACCOUNT_METRICS) as Array<[AccountMetric, typeof ACCOUNT_METRICS[AccountMetric]]>).map(([key, option]) => <button key={key} type="button" aria-pressed={metric === key} onClick={() => setMetric(key)} className={`min-h-10 rounded-md px-3 py-2 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${metric === key ? 'bg-raised text-ink-1' : 'text-ink-2 hover:bg-surface'}`}>{option.label}</button>)}
    </div>
    <p className="max-w-4xl text-sm leading-relaxed text-ink-2">{ACCOUNT_METRICS[metric].definition}</p>
    {progressKey && <p className="max-w-4xl text-sm text-ink-2"><strong className="font-medium text-ink-1">Observed so far</strong> includes current progress through the report time, even for immature cohorts. These counts are not D1/D7 conversion rates; every cohort has had a different amount of time.</p>}
    <div className="overflow-x-auto rounded-md border border-hairline">
      <table className="w-full min-w-[780px] border-collapse text-sm">
        <caption className="sr-only">{ACCOUNT_METRICS[metric].label} by account creation date; each metric displays its eligible denominator.</caption>
        <thead className="bg-raised text-left text-xs text-ink-2"><tr>
          <th scope="col" className="px-3 py-3 font-medium">Account creation date</th>
          <th scope="col" className="px-3 py-3 text-right font-medium">Accounts / mobile linked</th>
          {progressKey && <th scope="col" className="px-3 py-3 text-right font-medium">Observed so far</th>}
          {COHORT_DAYS.map(day => <th scope="col" key={day} className="px-3 py-3 text-right font-medium">D{day}</th>)}
        </tr></thead>
        <tbody>
          <tr className="border-b border-hairline bg-surface align-top">
            <th scope="row" className="px-3 py-4 text-left font-medium">Cohort totals<span className="mt-1 block text-xs font-normal text-ink-2">Fixed-age rates weighted by eligible accounts</span></th>
            <td className="numeral px-3 py-4 text-right">{data.coverage.accountsAsOf ? `${count(accounts)} / ${count(mobile)}` : 'Unavailable'}</td>
            {progressKey && <td className="px-3 py-4 text-right"><span className="numeral block font-semibold">{progressRows.length ? count(progressCount!) : 'Unavailable'}</span><span className="mt-1 block text-xs text-ink-2">{progressRows.length ? `${progressRows.length} cohorts · current count` : 'Source unavailable'}</span></td>}
            {COHORT_DAYS.map(day => {
              const summary = summarizeHorizon(data.rows, day, metric)
              return <td key={day} className="px-3 py-4 text-right">{summary.rate === null ? <span className="text-xs text-ink-2">No eligible observations</span> : <><span className="numeral block font-semibold">{metric === 'revenue' ? usd(summary.rate) : `${(summary.rate * 100).toFixed(1)}%`}</span><span className="numeral mt-1 block text-xs text-ink-2">{metric === 'revenue' ? usd(summary.numerator) : count(summary.numerator)} / {count(summary.denominator)}</span><span className="mt-1 block text-xs text-ink-2">{summary.includedCohorts} {summary.includedCohorts === 1 ? 'cohort' : 'cohorts'}</span></>}</td>
            })}
          </tr>
          {[...data.rows].reverse().map(row => <tr key={row.cohortDate} className="border-b border-hairline align-top last:border-0 hover:bg-surface">
            <th scope="row" className="numeral px-3 py-4 text-left font-normal">{row.cohortDate}</th>
            <td className="numeral px-3 py-4 text-right">{count(row.accounts)} / {count(row.mobileLinkedAccounts)}</td>
            {progressKey && <td className="numeral px-3 py-4 text-right">{row[progressKey] === null ? <span className="text-xs text-ink-2">Source unavailable</span> : count(row[progressKey])}</td>}
            {COHORT_DAYS.map(day => <td key={day} className="px-3 py-4 text-right"><CohortCell row={row} cell={row.horizons.find(cell => cell.days === day)} metric={metric} /></td>)}
          </tr>)}
          {data.rows.length === 0 && <tr><td colSpan={progressKey ? 7 : 6} className="px-3 py-6 text-sm text-ink-2">{data.coverage.accountsAsOf ? 'No accounts were created in this selected range.' : 'The account source has not produced a complete snapshot. Counts are unavailable.'}</td></tr>}
        </tbody>
      </table>
    </div>
    <p className="max-w-4xl text-xs leading-relaxed text-ink-2">“Still observing” means the full observation window has not elapsed. Unavailable cells are excluded from the weighted total, not counted as zero. Conversion and fee windows wait until the cohort’s full exact-return day has closed, so the last eligible date is conservative.</p>
    <dl className="flex flex-wrap gap-x-8 gap-y-3 border-t border-hairline pt-4 text-xs text-ink-2">
      <div><dt>Account snapshot</dt><dd className="mt-1 text-ink-1">{time(data.coverage.accountsAsOf)}</dd></div>
      <div><dt>Mobile activity checked</dt><dd className="mt-1 text-ink-1">{time(data.coverage.activityAsOf)}</dd></div>
      <div><dt>Report generated</dt><dd className="mt-1 text-ink-1">{time(data.generatedAt)}</dd></div>
    </dl>
    <details className="rounded-md border border-hairline p-4 text-sm text-ink-2">
      <summary className="cursor-pointer font-medium text-ink-1 focus-visible:outline-2 focus-visible:outline-accent">Source coverage and definitions</summary>
      <p className="mt-3">Mobile activity source: {time(data.coverage.activitySourceFrom)} through {time(data.coverage.activitySourceThrough)}. Account and mobile metrics come from the trading backend’s versioned measurement model.</p>
      <ul className="mt-3 list-disc space-y-2 pl-5">{data.coverage.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>
    </details>
    <dl className="grid gap-5 border-t border-hairline pt-5 sm:grid-cols-2">
      <div><dt className="font-medium">Expected LTV — unavailable</dt><dd className="mt-1 max-w-prose text-sm text-ink-2">A forecast requires longer mature cohorts, explicit assumptions and backtesting. Observed fee revenue is not an LTV estimate.</dd></div>
      <div><dt className="font-medium">Acquisition cost, CAC and ROAS — unavailable</dt><dd className="mt-1 max-w-prose text-sm text-ink-2">Creator costs have not been reconciled into this report. An empty earnings ledger does not establish zero spend. Apple Ads spend is not imported.</dd></div>
    </dl>
  </section>
}
