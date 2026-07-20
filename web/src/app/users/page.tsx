import { auth, signOut } from '@/auth'
import {
  activeUsers,
  sessionStats,
  topUsersByActivity,
  topTraders,
  activityHeatmap,
  newVsReturning,
  retentionCurve,
  cohortRetention,
} from '@/lib/metrics/users'
import { watermarkAge } from '@/lib/metrics/staleness'
import { NY_TZ } from '@/lib/time'
import { sql } from '@/lib/db'
import { fetchWalletIdentities } from '@/lib/privyIdentities'
import { PageHeader } from '@/components/PageHeader'
import { StatTile } from '@/components/StatTile'
import { DataTable } from '@/components/DataTable'
import { StalenessBadge } from '@/components/StalenessBadge'
import { TraderCell } from '@/components/TraderCell'

export const dynamic = 'force-dynamic'

const RANGES = {
  '7d': { label: '7D', days: 7 },
  '30d': { label: '30D', days: 30 },
  '90d': { label: '90D', days: 90 },
} as const

type RangeKey = keyof typeof RANGES

function isRangeKey(v: string | undefined): v is RangeKey {
  return v === '7d' || v === '30d' || v === '90d'
}

function today(): string {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: NY_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date())
}

function addDays(date: string, days: number): string {
  const d = new Date(`${date}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

function formatDateRange(start: string, end: string): string {
  const fmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' })
  const startLabel = fmt.format(new Date(`${start}T12:00:00Z`))
  const endLabel = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    .format(new Date(`${end}T12:00:00Z`))
  return `${startLabel} – ${endLabel}`
}

function shortDay(day: string): string {
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' }).format(new Date(`${day}T12:00:00Z`))
}

function formatTimestamp(iso: string): string {
  return new Intl.DateTimeFormat('en-US', {
    timeZone: NY_TZ, month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  }).format(new Date(iso))
}

const compact = (n: number) => Math.round(n).toLocaleString('en-US')
const usd = (n: number) => `$${Math.round(n).toLocaleString('en-US')}`
const minutes = (n: number) => `${n.toFixed(1)}m`
const pct = (n: number) => `${n.toFixed(0)}%`

const DOW_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

export default async function UsersPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const params = await searchParams
  const rawRange = typeof params.range === 'string' ? params.range : undefined
  const range: RangeKey = isRangeKey(rawRange) ? rawRange : '7d'

  const [session, end] = [await auth(), today()]
  const start = addDays(end, -(RANGES[range].days - 1))

  const [au, ss, topUsers, traders, heatmap, nvr, curve, cohorts, ages] = await Promise.all([
    activeUsers(start, end),
    sessionStats(start, end),
    topUsersByActivity(start, end, 20),
    topTraders(start, end, 20),
    activityHeatmap(start, end),
    newVsReturning(start, end),
    retentionCurve(start, end),
    cohortRetention(start, end),
    watermarkAge(),
  ])
  // Identity lookup scoped to just the wallets on this page -- see
  // src/lib/privyIdentities.ts and the identical comment on /trades.
  const privyMap = await fetchWalletIdentities(sql, [...topUsers.map(u => u.wallet), ...traders.map(t => t.wallet)])

  const maxDau = Math.max(...au.daily.map(d => d.users), 1)
  const maxHeat = Math.max(...heatmap.map(c => c.count), 1)
  const maxNvr = Math.max(...nvr.map(d => d.newUsers + d.returningUsers), 1)
  const heatByKey = new Map(heatmap.map(c => [`${c.dayOfWeek}-${c.hour}`, c.count]))

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <PageHeader
        title="Users & Retention"
        subtitle={
          <div className="flex flex-wrap items-center gap-3">
            <span>{formatDateRange(start, end)} · {NY_TZ}</span>
            <nav aria-label="Date range" className="flex items-center gap-1">
              {(Object.keys(RANGES) as RangeKey[]).map(key => (
                <a
                  key={key}
                  href={key === '7d' ? '/users' : `/users?range=${key}`}
                  aria-current={key === range ? 'true' : undefined}
                  className={`numeral rounded-sm px-2 py-0.5 text-xs outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent ${
                    key === range
                      ? 'bg-surface text-ink-1'
                      : 'text-ink-3 hover:text-ink-1'
                  }`}
                >
                  {RANGES[key].label}
                </a>
              ))}
            </nav>
          </div>
        }
        right={
          <div className="flex items-center gap-4">
            <StalenessBadge ages={ages} />
            <form action={async () => { 'use server'; await signOut({ redirectTo: '/login' }) }}>
              <button className="text-xs text-ink-2 outline-none transition-colors hover:text-ink-1 focus-visible:ring-2 focus-visible:ring-accent">
                {session?.user?.email} · sign out
              </button>
            </form>
          </div>
        }
      />

      <div key={range} className="animate-content-fade">
        <section aria-label="Key metrics" className="mt-8 grid gap-x-6 divide-y divide-hairline/60 sm:grid-cols-5 sm:divide-x sm:divide-y-0">
          <StatTile label="DAU" value={au.dau} format={compact} sparklineValues={au.daily.map(d => d.users)} />
          <div className="sm:pl-6">
            <StatTile label="WAU" value={au.wau} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="MAU" value={au.mau} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Sessions" value={ss.sessionCount} format={compact} sparklineValues={ss.daily.map(d => d.totalMinutes)} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Median session" value={ss.medianDurationMin} format={minutes} />
          </div>
        </section>

        <section aria-label="Session detail" className="mt-6 border-t border-hairline/60 pt-4 text-xs text-ink-2">
          <span className="numeral">{compact(ss.distinctUsers)}</span> users with sessions ·{' '}
          <span className="numeral">{ss.avgSessionsPerUser.toFixed(1)}</span> sessions/user avg ·{' '}
          p90 <span className="numeral">{minutes(ss.p90DurationMin)}</span>
        </section>

        <section aria-label="Daily active users" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">
            Daily active users · {NY_TZ}
          </h2>
          <div className="mt-4 flex h-20 items-end gap-[2px]">
            {au.daily.map(d => (
              <div key={d.day} className="group relative flex-1">
                <div
                  className="rounded-t-[1px] bg-accent-dim transition-colors group-hover:bg-accent"
                  style={{ height: `${Math.max((d.users / maxDau) * 100, d.users > 0 ? 3 : 1)}%` }}
                />
                <span className="pointer-events-none absolute -top-6 left-1/2 z-10 hidden -translate-x-1/2 whitespace-nowrap rounded-sm bg-surface px-1.5 py-0.5 text-[10px] text-ink-1 group-hover:block">
                  {shortDay(d.day)} · {compact(d.users)}
                </span>
              </div>
            ))}
          </div>
          <div className="numeral mt-1.5 flex justify-between text-[10px] text-ink-3">
            <span>{shortDay(au.daily[0]?.day ?? start)}</span>
            <span>{shortDay(au.daily[au.daily.length - 1]?.day ?? end)}</span>
          </div>
        </section>

        <section aria-label="New vs returning users" className="mt-8 border-t border-hairline/60 pt-8">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">New vs returning</h2>
            <div className="flex items-center gap-3 text-[10px] text-ink-3">
              <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden="true" />new</span>
              <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-accent-dim" aria-hidden="true" />returning</span>
            </div>
          </div>
          <div className="mt-4 flex h-20 items-end gap-[2px]">
            {nvr.map(d => {
              const total = d.newUsers + d.returningUsers
              const newPct = total > 0 ? (d.newUsers / maxNvr) * 100 : 0
              const retPct = total > 0 ? (d.returningUsers / maxNvr) * 100 : 0
              return (
                <div key={d.day} className="group relative flex flex-1 flex-col justify-end">
                  <div className="rounded-t-[1px] bg-accent transition-opacity group-hover:opacity-80" style={{ height: `${newPct}%` }} />
                  <div className="bg-accent-dim transition-opacity group-hover:opacity-80" style={{ height: `${retPct}%` }} />
                  <span className="pointer-events-none absolute -top-6 left-1/2 z-10 hidden -translate-x-1/2 whitespace-nowrap rounded-sm bg-surface px-1.5 py-0.5 text-[10px] text-ink-1 group-hover:block">
                    {shortDay(d.day)} · {d.newUsers}n/{d.returningUsers}r
                  </span>
                </div>
              )
            })}
          </div>
          <div className="numeral mt-1.5 flex justify-between text-[10px] text-ink-3">
            <span>{shortDay(nvr[0]?.day ?? start)}</span>
            <span>{shortDay(nvr[nvr.length - 1]?.day ?? end)}</span>
          </div>
        </section>

        <section aria-label="Activity heatmap" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">
            Activity heatmap · {NY_TZ}
          </h2>
          <div className="mt-4 overflow-x-auto">
            <div className="grid min-w-[640px] grid-cols-[2.5rem_repeat(24,1fr)] gap-[2px]">
              <div />
              {Array.from({ length: 24 }, (_, h) => (
                <div key={h} className="numeral text-center text-[9px] text-ink-3">
                  {h % 3 === 0 ? h : ''}
                </div>
              ))}
              {DOW_LABELS.map((label, dow) => (
                <div key={label} className="contents">
                  <div className="flex items-center text-[10px] text-ink-3">{label}</div>
                  {Array.from({ length: 24 }, (_, hour) => {
                    const count = heatByKey.get(`${dow}-${hour}`) ?? 0
                    const intensity = count / maxHeat
                    return (
                      <div
                        key={hour}
                        title={`${label} ${hour}:00 · ${compact(count)} events`}
                        className="aspect-square rounded-[1px] bg-surface"
                        style={{ backgroundColor: 'var(--color-accent)', opacity: count > 0 ? Math.max(intensity, 0.08) : 0 }}
                      />
                    )
                  })}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section
          aria-label="Top users and traders"
          className="mt-8 grid gap-x-10 gap-y-8 divide-y divide-hairline/60 border-t border-hairline/60 pt-8 md:grid-cols-2 md:divide-x md:divide-y-0"
        >
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Top users by activity</h2>
            <div className="mt-3">
              <DataTable
                rowKey={row => row.wallet}
                rows={topUsers}
                columns={[
                  { key: 'wallet', header: 'User', render: r => <TraderCell wallet={r.wallet} privyMap={privyMap} /> },
                  { key: 'events', header: 'Events', align: 'right', render: r => compact(r.events) },
                  { key: 'sessions', header: 'Sessions', align: 'right', render: r => compact(r.sessions) },
                  { key: 'lastSeen', header: 'Last seen', align: 'right', render: r => formatTimestamp(r.lastSeen) },
                ]}
              />
            </div>
          </div>
          <div className="pt-8 md:pl-10 md:pt-0">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Top traders</h2>
            <div className="mt-3">
              <DataTable
                rowKey={row => row.wallet}
                rows={traders}
                columns={[
                  { key: 'wallet', header: 'User', render: r => <TraderCell wallet={r.wallet} privyMap={privyMap} /> },
                  { key: 'trades', header: 'Trades', align: 'right', render: r => compact(r.trades) },
                  { key: 'volumeUsd', header: 'Volume', align: 'right', render: r => usd(r.volumeUsd) },
                ]}
              />
            </div>
          </div>
        </section>

        <section aria-label="Retention curve" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Average retention curve</h2>
          <div className="mt-4 flex h-24 items-end gap-4">
            {curve.map(pt => (
              <div key={pt.offsetDay} className="group relative flex flex-1 flex-col items-center justify-end">
                <span className="numeral mb-1 text-[10px] text-ink-2">{pt.avgPct.toFixed(0)}%</span>
                <div
                  className="w-full max-w-8 rounded-t-[1px] bg-accent-dim transition-colors group-hover:bg-accent"
                  style={{ height: `${Math.max(pt.avgPct, pt.avgPct > 0 ? 3 : 1)}%` }}
                />
                <span className="numeral mt-1.5 text-[10px] text-ink-3">{pt.label}</span>
              </div>
            ))}
          </div>
        </section>

        <section aria-label="Cohort retention" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Cohort retention</h2>
          <div className="mt-4 overflow-x-auto">
            <div className="min-w-[560px]">
              <div className="grid grid-cols-[5.5rem_5rem_repeat(6,1fr)] gap-1 text-[10px] uppercase tracking-wide text-ink-3">
                <div>Cohort</div>
                <div className="text-right">Users</div>
                {curve.map(pt => <div key={pt.offsetDay} className="text-center">{pt.label}</div>)}
              </div>
              <div className="mt-1 space-y-1">
                {cohorts.map(c => (
                  <div key={c.cohortDate} className="grid grid-cols-[5.5rem_5rem_repeat(6,1fr)] items-center gap-1">
                    <div className="numeral text-xs text-ink-2">{shortDay(c.cohortDate)}</div>
                    <div className="numeral text-right text-xs text-ink-1">{compact(c.cohortSize)}</div>
                    {c.cells.map(cell => (
                      <div
                        key={cell.offsetDay}
                        title={`${shortDay(c.cohortDate)} → D${cell.offsetDay}: ${cell.pct.toFixed(0)}% (${cell.retained}/${c.cohortSize})`}
                        className="numeral flex h-6 items-center justify-center rounded-[1px] text-[10px] text-ink-1"
                        style={{ backgroundColor: 'var(--color-accent)', opacity: cell.retained > 0 ? Math.max(cell.pct / 100, 0.1) : 0 }}
                      >
                        {pct(cell.pct)}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}
