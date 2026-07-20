import { auth, signOut } from '@/auth'
import {
  kpiSummary,
  platformSplit,
  topEvents,
  dailySeries,
  hourlyActivity,
  tradeSummary,
} from '@/lib/metrics/overview'
import { watermarkAge } from '@/lib/metrics/staleness'
import { NY_TZ } from '@/lib/time'
import { PageHeader } from '@/components/PageHeader'
import { StatTile } from '@/components/StatTile'
import { BarList } from '@/components/BarList'
import { DataTable } from '@/components/DataTable'
import { StalenessBadge } from '@/components/StalenessBadge'

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

const usd = (n: number) =>
  `$${Math.round(n).toLocaleString('en-US')}`
const compact = (n: number) => Math.round(n).toLocaleString('en-US')

/**
 * Every volume figure on this page is the DB reconstruction of perps
 * notional (`VOLUME_USD_EXPR` in lib/metrics/trades.ts, imported by
 * overview.ts) -- it multiplies margin by leverage, not Hyperliquid's
 * authoritative per-fill data, and runs ~15% high as a result. Same "est."
 * tag as the Trades page, so the two pages read consistently.
 */
function EstTag() {
  return (
    <span
      className="numeral ml-1.5 rounded-sm bg-alert-dim px-1 py-px align-middle text-[10px] uppercase tracking-wide text-alert"
      title="Perps volume is reconstructed from intended order size (opens: margin × leverage; closes: size × price), not Hyperliquid's per-fill data. Runs ~15% high vs the authoritative HL figure. Swap volume is exact."
    >
      est.
    </span>
  )
}

export default async function OverviewPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const params = await searchParams
  const rawRange = typeof params.range === 'string' ? params.range : undefined
  const range: RangeKey = isRangeKey(rawRange) ? rawRange : '7d'

  const [session, end] = [await auth(), today()]
  const start = addDays(end, -(RANGES[range].days - 1))

  const [kpis, platforms, events, series, hourly, trades, ages] = await Promise.all([
    kpiSummary(start, end),
    platformSplit(start, end),
    topEvents(start, end, 8),
    dailySeries(start, end),
    hourlyActivity(start, end),
    tradeSummary(start, end),
    watermarkAge(),
  ])

  const maxHour = Math.max(...hourly.map(h => h.count), 1)

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <PageHeader
        title="Overview"
        subtitle={
          <div className="flex flex-wrap items-center gap-3">
            <span>{formatDateRange(start, end)} · {NY_TZ}</span>
            <nav aria-label="Date range" className="flex items-center gap-1">
              {(Object.keys(RANGES) as RangeKey[]).map(key => (
                <a
                  key={key}
                  href={key === '7d' ? '/' : `/?range=${key}`}
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
          <StatTile
            label="Events"
            value={kpis.events.current}
            previousValue={kpis.events.previous}
            format={compact}
            sparklineValues={series.map(d => d.events)}
          />
          <div className="sm:pl-6">
            <StatTile
              label="Active users"
              value={kpis.users.current}
              previousValue={kpis.users.previous}
              format={compact}
              sparklineValues={series.map(d => d.users)}
            />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label="Sessions"
              value={kpis.sessions.current}
              previousValue={kpis.sessions.previous}
              format={compact}
            />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label="Trades"
              value={kpis.trades.current}
              previousValue={kpis.trades.previous}
              format={compact}
              sparklineValues={series.map(d => d.trades)}
            />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label="Volume (est.)"
              value={kpis.volumeUsd.current}
              previousValue={kpis.volumeUsd.previous}
              format={usd}
              sparklineValues={series.map(d => d.volumeUsd)}
            />
          </div>
        </section>
        <p className="mt-2 text-xs text-ink-3" title="Perps volume is reconstructed from intended order size (opens: margin × leverage; closes: size × price), not Hyperliquid's per-fill data. Runs ~15% high vs the authoritative HL figure. Swap volume is exact.">
          <span className="text-alert">est.</span> volume includes reconstructed perps volume — see tooltip.
        </p>

        <section
          aria-label="Breakdowns"
          className="mt-10 grid gap-x-10 gap-y-8 divide-y divide-hairline/60 border-t border-hairline/60 pt-8 md:grid-cols-2 md:divide-x md:divide-y-0"
        >
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Platform split</h2>
            <div className="mt-3">
              <BarList
                items={platforms.map(p => ({ label: p.platform, value: p.events, sublabel: `${compact(p.users)}u` }))}
                formatValue={compact}
              />
            </div>
          </div>
          <div className="pt-8 md:pl-10 md:pt-0">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Top events</h2>
            <div className="mt-3">
              <BarList
                items={events.map(e => ({ label: e.event, value: e.count }))}
                formatValue={compact}
              />
            </div>
          </div>
        </section>

        <section
          aria-label="Trade breakdowns"
          className="mt-8 grid gap-x-10 gap-y-8 divide-y divide-hairline/60 border-t border-hairline/60 pt-8 md:grid-cols-2 md:divide-x md:divide-y-0"
        >
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">
              Volume by type <EstTag />
            </h2>
            <div className="mt-3">
              <BarList
                items={trades.byType.map(t => ({ label: t.type, value: t.volumeUsd, sublabel: `${compact(t.count)}x` }))}
                formatValue={usd}
              />
            </div>
          </div>
          <div className="pt-8 md:pl-10 md:pt-0">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">
              Volume by client <EstTag />
            </h2>
            <div className="mt-3">
              <BarList
                items={trades.byClient.map(t => ({ label: t.client, value: t.volumeUsd, sublabel: `${compact(t.count)}x` }))}
                formatValue={usd}
              />
            </div>
          </div>
        </section>

        <section aria-label="Hourly activity" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">
            Hourly activity · {NY_TZ}
          </h2>
          <div className="mt-4 flex h-20 items-end gap-[3px]">
            {hourly.map(h => (
              <div key={h.hour} className="group relative flex-1">
                <div
                  className="rounded-t-[1px] bg-accent-dim transition-colors group-hover:bg-accent"
                  style={{ height: `${Math.max((h.count / maxHour) * 100, h.count > 0 ? 3 : 1)}%` }}
                />
                <span className="pointer-events-none absolute -top-6 left-1/2 hidden -translate-x-1/2 whitespace-nowrap rounded-sm bg-surface px-1.5 py-0.5 text-[10px] text-ink-1 group-hover:block">
                  {h.hour}:00 · {compact(h.count)}
                </span>
              </div>
            ))}
          </div>
          <div className="numeral mt-1.5 flex justify-between text-[10px] text-ink-3">
            <span>0:00</span>
            <span>6:00</span>
            <span>12:00</span>
            <span>18:00</span>
            <span>23:00</span>
          </div>
        </section>

        <section aria-label="Daily detail" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Daily detail</h2>
          <div className="mt-3">
            <DataTable
              rowKey={row => row.day}
              rows={[...series].reverse()}
              columns={[
                { key: 'day', header: 'Day', render: r => r.day },
                { key: 'events', header: 'Events', align: 'right', render: r => compact(r.events) },
                { key: 'users', header: 'Users', align: 'right', render: r => compact(r.users) },
                { key: 'trades', header: 'Trades', align: 'right', render: r => compact(r.trades) },
                { key: 'volume', header: 'Volume (est.)', align: 'right', render: r => usd(r.volumeUsd) },
              ]}
            />
          </div>
        </section>
      </div>
    </main>
  )
}
