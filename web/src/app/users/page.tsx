import { Suspense } from 'react'
import { AccountCohortsSection, AccountCohortsLoading } from '@/components/AccountCohortsSection'
import { auth, signOut } from '@/auth'
import {
  activeUsers,
  sessionStats,
  topUsersByActivity,
  topTraders,
  activityHeatmap,
} from '@/lib/metrics/users'
import { watermarkAge } from '@/lib/metrics/staleness'
import { NY_TZ } from '@/lib/time'
import { RANGES, isRangeKey, rangeStart, todayNy, type RangeKey } from '@/lib/ranges'
import { sql } from '@/lib/db'
import { fetchWalletIdentities } from '@/lib/privyIdentities'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { StatTile } from '@/components/StatTile'
import { DataTable } from '@/components/DataTable'
import { StalenessBadge } from '@/components/StalenessBadge'
import { AutoRefresh } from '@/components/AutoRefresh'
import { TraderCell } from '@/components/TraderCell'
import { TimeSeriesBars } from '@/components/TimeSeriesBars'

export const dynamic = 'force-dynamic'

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

const DOW_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

export default async function UsersPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const params = await searchParams
  const rawRange = typeof params.range === 'string' ? params.range : undefined
  const range: RangeKey = isRangeKey(rawRange) ? rawRange : '7d'

  const [session, end] = [await auth(), todayNy()]
  const start = rangeStart(end, range)


  return (
    <main className="mx-auto w-full max-w-[1600px] px-8 py-8">
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
                      ? 'bg-raised font-medium text-ink-1'
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
            <AutoRefresh intervalMs={60_000} renderedAt={Date.now()} />
            <form action={async () => { 'use server'; await signOut({ redirectTo: '/login' }) }}>
              <button className="text-xs text-ink-2 outline-none transition-colors hover:text-ink-1 focus-visible:ring-2 focus-visible:ring-accent">
                {session?.user?.email} · sign out
              </button>
            </form>
          </div>
        }
      />

      <Suspense key={`accounts:${start}:${end}`} fallback={<AccountCohortsLoading />}>
        <AccountCohortsSection from={start} to={end} initialMetric="appReturn" />
      </Suspense>

      <Suspense key={`web:${start}:${end}`} fallback={<p role="status" className="mt-8 text-sm text-ink-2">Loading web activity and trading diagnostics…</p>}>
        <WebDiagnostics start={start} end={end} />
      </Suspense>
    </main>
  )
}


async function WebDiagnostics({ start, end }: { start: string; end: string }) {
  try {
    const [au, ss, topUsers, traders, heatmap, ages] = await Promise.all([
      activeUsers(start, end, 'web'),
      sessionStats(start, end, 'web'),
      topUsersByActivity(start, end, 20, 'web'),
      topTraders(start, end, 20),
      activityHeatmap(start, end, 'web'),
      watermarkAge(),
    ])
    // Identity lookup scoped to just the wallets on this page -- see
    // src/lib/privyIdentities.ts and the identical comment on /trades.
    const privyMap = await fetchWalletIdentities(sql, [...topUsers.map(u => u.wallet), ...traders.map(t => t.wallet)])

    const maxHeat = Math.max(...heatmap.map(c => c.count), 1)
    const heatByKey = new Map(heatmap.map(c => [`${c.dayOfWeek}-${c.hour}`, c.count]))

    return (
      <div className="animate-content-fade">
        <section className="mt-8 border-t border-hairline pt-8">
          <SectionHeading meta={<StalenessBadge ages={ages} />}>Web activity</SectionHeading>
          <p className="mt-2 max-w-3xl text-sm text-ink-2">Web events from the legacy analytics mirror only. Active users are distinct connected wallets, not account cohorts; anonymous visitors are excluded. This source does not measure current mobile activity or retention.</p>
        </section>
        <section aria-label="Key metrics" className="mt-8 grid gap-x-6 divide-y divide-hairline sm:grid-cols-5 sm:divide-x sm:divide-y-0">
          <StatTile label="Daily active web wallets" value={au.dau} format={compact} />
          <div className="sm:pl-6">
            <StatTile label="Weekly active web wallets" value={au.wau} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Monthly active web wallets" value={au.mau} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Web session events" value={ss.sessionCount} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Median reported duration" value={ss.medianDurationMin} format={minutes} />
          </div>
        </section>

        <section aria-label="Session detail" className="mt-6 border-t border-hairline pt-4 text-xs text-ink-2">
          <span className="numeral">{compact(ss.distinctUsers)}</span> users with sessions ·{' '}
          <span className="numeral">{ss.avgSessionsPerUser.toFixed(1)}</span> sessions/user avg ·{' '}
          p90 <span className="numeral">{minutes(ss.p90DurationMin)}</span>
        </section>

        <section aria-label="Daily active users" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading meta={NY_TZ}>Daily active web wallets</SectionHeading>
          <div className="mt-4">
            <TimeSeriesBars
              data={au.daily.map(d => ({ day: d.day, value: d.users }))}
              formatValue={compact}
              formatDay={shortDay}
            />
          </div>
        </section>

        <section aria-label="Activity heatmap" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading meta={NY_TZ}>Web activity heatmap</SectionHeading>
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
          className="mt-8 grid items-start gap-x-10 gap-y-8 divide-y divide-hairline border-t border-hairline pt-8 md:grid-cols-2 md:divide-x md:divide-y-0"
        >
          <div>
            <SectionHeading>Top web wallets by activity</SectionHeading>
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
            <SectionHeading>Top traders</SectionHeading>
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


      </div>
    )
  } catch {
    return <section className="mt-8 border-t border-hairline pt-8"><SectionHeading>Web activity</SectionHeading><p role="status" className="mt-3 text-sm text-alert">The legacy web/trading mirror is unavailable. Account cohorts above use an independent source.</p></section>
  }
}
