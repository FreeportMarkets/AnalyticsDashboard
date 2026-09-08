'use client'

import { useState } from 'react'
import type { DailyJourneyPoint } from '@/lib/metrics/journeyDaily'

const fmt = (n: number) => n.toLocaleString('en-US')
const dayLabel = (day: string) => new Date(`${day}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
const control = 'min-h-11 rounded-md px-3 py-2 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'

export function JourneyDailyChart({ data, actionsAvailable }: { data: DailyJourneyPoint[]; actionsAvailable: boolean }) {
  const [range, setRange] = useState(14)
  const [mode, setMode] = useState<'activity' | 'signup'>('activity')
  const [selected, setSelected] = useState(data.length - 1)
  const shown = data.slice(-range)
  const point = data[selected] ?? data[data.length - 1]!
  const today = data[data.length - 1]?.day
  const series = [
    { key: 'signups' as const, label: 'New signups', color: 'var(--color-accent)', dash: undefined },
    { key: mode === 'activity' ? 'firstDeposits' as const : 'signupDayDeposits' as const, label: mode === 'activity' ? 'First deposits' : 'Signed up & deposited', color: 'var(--color-positive)', dash: '7 4' },
    { key: mode === 'activity' ? 'firstTrades' as const : 'signupDayTrades' as const, label: mode === 'activity' ? 'First trades' : 'Signed up & traded', color: 'var(--color-ink-2)', dash: '2 4' },
  ].filter(s => actionsAvailable || s.key === 'signups')
  const max = Math.max(2, ...shown.flatMap(d => series.map(s => d[s.key])))
  const ceiling = Math.ceil(max / 2) * 2
  const x = (i: number) => 3 + i / Math.max(shown.length - 1, 1) * 94
  const y = (value: number) => 95 - value / ceiling * 85
  const selectedIndex = shown.findIndex(d => d.day === point.day)
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-semibold text-ink-1">{point.day === today ? 'Today so far' : dayLabel(point.day)} <span className="text-sm font-normal text-ink-2">· {point.day} · New York time</span></h3>
        <label className="flex items-center gap-2 text-sm text-ink-2">Inspect day
          <select className={`${control} border border-hairline bg-surface text-ink-1`} value={point.day} onChange={event => setSelected(data.findIndex(d => d.day === event.target.value))}>
            {[...shown].reverse().map(d => <option key={d.day} value={d.day}>{dayLabel(d.day)}{d.day === today ? ' · today' : ''}</option>)}
          </select>
        </label>
      </div>
      <dl className="grid gap-4 lg:grid-cols-3" aria-live="polite">
        {[
          { label: 'New signups', value: point.signups, detail: 'Accounts created on this day' },
          { label: 'First successful deposits', value: actionsAvailable ? point.firstDeposits : null, detail: actionsAvailable ? `${fmt(point.signupDayDeposits)} also signed up on this day` : 'Deposit history unavailable' },
          { label: 'First filled trades', value: actionsAvailable ? point.firstTrades : null, detail: actionsAvailable ? `${fmt(point.signupDayTrades)} also signed up on this day` : 'Trade history unavailable' },
        ].map(item => (
          <div key={item.label} className="rounded-xl border border-hairline bg-surface/30 p-5">
            <dt className="text-sm text-ink-2">{item.label}</dt>
            <dd className="numeral mt-3 text-3xl font-semibold">{item.value === null ? '—' : fmt(item.value)}</dd>
            <dd className="mt-2 text-sm text-ink-2">{item.detail}</dd>
          </div>
        ))}
      </dl>
      <p className="text-sm text-ink-2">First deposits and trades mean first in the available history, across each account’s linked wallets. Older missing records can affect these counts.</p>
      <div className="rounded-xl border border-hairline p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div role="group" aria-label="Daily metric definition" className="flex flex-wrap gap-1">
            {([['activity', 'First actions by day'], ['signup', 'Same-day signup conversion']] as const).map(([key, label]) => (
              <button key={key} type="button" aria-pressed={mode === key} className={`${control} ${mode === key ? 'bg-raised text-ink-1' : 'text-ink-2 hover:bg-surface'}`} onClick={() => setMode(key)}>{label}</button>
            ))}
          </div>
          <div role="group" aria-label="Daily chart range" className="flex gap-1">
            {[14, 30, 90].map(n => <button key={n} type="button" aria-pressed={range === n} className={`${control} ${range === n ? 'bg-raised text-ink-1' : 'text-ink-2 hover:bg-surface'}`} onClick={() => { setRange(n); setSelected(data.length - 1) }}>{n}d</button>)}
          </div>
        </div>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-ink-2">
          {mode === 'activity' ? 'Each action is plotted on the day it first appears in recorded history. A first-time trader today may have signed up earlier.' : 'Of the accounts created each day, how many also deposited or traded before midnight that same day. Deposit and trade groups can overlap.'}
        </p>
        <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-sm text-ink-2">
          {series.map(s => <span key={s.key} className="flex items-center gap-2"><svg aria-hidden="true" width="24" height="12"><line x1="0" y1="6" x2="24" y2="6" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} /></svg>{s.label}</span>)}
        </div>
        <div className="mt-5 flex gap-2">
          <div aria-hidden="true" className="numeral flex w-8 shrink-0 flex-col justify-between pb-3 pt-5 text-right text-xs text-ink-2"><span>{fmt(ceiling)}</span><span>{fmt(ceiling / 2)}</span><span>0</span></div>
          <div className="relative h-60 min-w-0 flex-1">
            <svg role="img" aria-label={`${mode === 'activity' ? 'Daily first actions' : 'Same-day signup conversion'}, ${dayLabel(shown[0]!.day)} to ${dayLabel(shown[shown.length - 1]!.day)}. Exact values are in the daily table below.`} viewBox="0 0 100 100" preserveAspectRatio="none" className="h-full w-full overflow-visible">
              {[0, ceiling / 2, ceiling].map(value => <line key={value} x1="0" x2="100" y1={y(value)} y2={y(value)} stroke="var(--color-hairline)" vectorEffect="non-scaling-stroke" />)}
              <rect x={x(shown.length - 1) - 1.5} y="5" width="3" height="90" fill="var(--color-surface)" />
              {series.map(s => <polyline key={s.key} points={shown.map((d, i) => `${x(i)},${y(d[s.key])}`).join(' ')} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} vectorEffect="non-scaling-stroke" />)}
              {selectedIndex >= 0 && <line x1={x(selectedIndex)} x2={x(selectedIndex)} y1="5" y2="95" stroke="var(--color-ink-2)" strokeDasharray="3 4" vectorEffect="non-scaling-stroke" />}
            </svg>
          </div>
        </div>
        <div aria-hidden="true" className="ml-10 flex justify-between text-xs text-ink-2"><span>{dayLabel(shown[0]!.day)}</span><span>{dayLabel(shown[Math.floor(shown.length / 2)]!.day)}</span><span>Today · partial</span></div>
        <p className="mt-4 text-sm text-ink-2" aria-live="polite">{dayLabel(point.day)}: {series.map(s => `${fmt(point[s.key])} ${s.label.toLowerCase()}`).join(' · ')}</p>
        <details className="mt-4 border-t border-hairline pt-3">
          <summary className={`${control} cursor-pointer text-ink-1`}>View daily numbers</summary>
          <div role="region" aria-label="Daily signup and conversion table" tabIndex={0} className="overflow-x-auto focus-visible:outline-2 focus-visible:outline-accent">
            <table className="w-full min-w-[540px] text-sm">
              <thead><tr className="border-b border-hairline text-ink-2"><th scope="col" className="py-3 text-left font-medium">Date</th>{series.map(s => <th key={s.key} scope="col" className="p-3 text-right font-medium">{s.label}</th>)}</tr></thead>
              <tbody>{[...shown].reverse().map(d => <tr key={d.day} className="border-b border-hairline last:border-0"><th scope="row" className="py-3 text-left font-normal">{dayLabel(d.day)}{d.day === today ? ' · partial' : ''}</th>{series.map(s => <td key={s.key} className="numeral p-3 text-right">{fmt(d[s.key])}</td>)}</tr>)}</tbody>
            </table>
          </div>
        </details>
      </div>
    </div>
  )
}
