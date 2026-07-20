import {
  sendSummary,
  sendsByType,
  tapRateByDay,
  timeToTradeAfterTap,
  mostTappedTickers,
  mostEngagedUsers,
} from '@/lib/metrics/notifications'
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

const compact = (n: number) => Math.round(n).toLocaleString('en-US')
const pct = (n: number) => `${n.toFixed(1)}%`

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(1)}h`
}

function shortWallet(address: string): string {
  return address.length > 12 ? `${address.slice(0, 5)}…${address.slice(-4)}` : address
}

export default async function NotificationsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const params = await searchParams
  const rawRange = typeof params.range === 'string' ? params.range : undefined
  const range: RangeKey = isRangeKey(rawRange) ? rawRange : '7d'

  const end = today()
  const start = addDays(end, -(RANGES[range].days - 1))

  const [summary, byType, byDay, timeToTrade, tickers, users, ages] = await Promise.all([
    sendSummary(start, end),
    sendsByType(start, end),
    tapRateByDay(start, end),
    timeToTradeAfterTap(start, end),
    mostTappedTickers(start, end),
    mostEngagedUsers(start, end),
    watermarkAge(),
  ])

  const maxTapRate = Math.max(...byDay.map(d => d.tapRatePct), 1)
  const conversionPct = timeToTrade.totalTaps > 0
    ? (timeToTrade.tradedWithin24h / timeToTrade.totalTaps) * 100
    : 0

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <PageHeader
        title="Notifications"
        subtitle={
          <div className="flex flex-wrap items-center gap-3">
            <span>{formatDateRange(start, end)} · {NY_TZ}</span>
            <nav aria-label="Date range" className="flex items-center gap-1">
              {(Object.keys(RANGES) as RangeKey[]).map(key => (
                <a
                  key={key}
                  href={key === '7d' ? '/notifications' : `/notifications?range=${key}`}
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
        right={<StalenessBadge ages={ages} />}
      />

      <div key={range} className="animate-content-fade">
        <section aria-label="Key metrics" className="mt-8 grid gap-x-6 divide-y divide-hairline/60 sm:grid-cols-5 sm:divide-x sm:divide-y-0">
          <StatTile
            label="Sent"
            value={summary.sent}
            format={compact}
            sparklineValues={byDay.map(d => d.sent)}
          />
          <div className="sm:pl-6">
            <StatTile
              label="Tap rate"
              value={summary.tapRatePct}
              format={pct}
              sparklineValues={byDay.map(d => d.tapRatePct)}
            />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label="Tapped"
              value={summary.tapped}
              format={compact}
              sparklineValues={byDay.map(d => d.tapped)}
            />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label="Delivery rate"
              value={summary.deliveryRatePct}
              format={pct}
            />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label="Trades after tap (24h)"
              value={conversionPct}
              format={pct}
            />
          </div>
        </section>

        <section
          aria-label="Sends and tap rate"
          className="mt-10 grid gap-x-10 gap-y-8 divide-y divide-hairline/60 border-t border-hairline/60 pt-8 md:grid-cols-2 md:divide-x md:divide-y-0"
        >
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Sends by type</h2>
            <div className="mt-3">
              <BarList
                items={byType.map(t => ({
                  label: t.type,
                  value: t.recipients,
                  sublabel: t.sends !== t.recipients ? `${compact(t.sends)}x` : undefined,
                }))}
                formatValue={compact}
              />
            </div>
          </div>
          <div className="pt-8 md:pl-10 md:pt-0">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">
              Tap rate by day · {NY_TZ}
            </h2>
            <div className="mt-4 flex h-20 items-end gap-[3px]">
              {byDay.map(d => (
                <div key={d.day} className="group relative flex-1">
                  <div
                    className="rounded-t-[1px] bg-accent-dim transition-colors group-hover:bg-accent"
                    style={{ height: `${Math.max((d.tapRatePct / maxTapRate) * 100, d.tapRatePct > 0 ? 3 : 1)}%` }}
                  />
                  <span className="pointer-events-none absolute -top-6 left-1/2 hidden -translate-x-1/2 whitespace-nowrap rounded-sm bg-surface px-1.5 py-0.5 text-[10px] text-ink-1 group-hover:block">
                    {d.day} · {pct(d.tapRatePct)}
                  </span>
                </div>
              ))}
            </div>
            <div className="numeral mt-1.5 flex justify-between text-[10px] text-ink-3">
              <span>{start}</span>
              <span>{end}</span>
            </div>
          </div>
        </section>

        <section
          aria-label="Tapped tickers and engaged users"
          className="mt-8 grid gap-x-10 gap-y-8 divide-y divide-hairline/60 border-t border-hairline/60 pt-8 md:grid-cols-2 md:divide-x md:divide-y-0"
        >
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Most tapped tickers</h2>
            <div className="mt-3">
              <BarList
                items={tickers.map(t => ({ label: t.ticker, value: t.taps }))}
                formatValue={compact}
              />
            </div>
          </div>
          <div className="pt-8 md:pl-10 md:pt-0">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Most engaged users</h2>
            <div className="mt-3">
              <BarList
                items={users.map(u => ({
                  label: shortWallet(u.walletAddress),
                  value: u.taps,
                  sublabel: `${pct(u.tapRatePct)} rate`,
                }))}
                formatValue={compact}
              />
            </div>
          </div>
        </section>

        <section aria-label="Time to trade after tap" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">
            Time to trade after notification tap
          </h2>
          <div className="mt-4 grid gap-x-6 divide-y divide-hairline/60 sm:grid-cols-4 sm:divide-x sm:divide-y-0">
            <StatTile label="Taps" value={timeToTrade.totalTaps} format={compact} />
            <div className="sm:pl-6">
              <StatTile label="Traded within 1h" value={timeToTrade.tradedWithin1h} format={compact} />
            </div>
            <div className="sm:pl-6">
              <StatTile label="Traded within 6h" value={timeToTrade.tradedWithin6h} format={compact} />
            </div>
            <div className="sm:pl-6">
              <StatTile label="Traded within 24h" value={timeToTrade.tradedWithin24h} format={compact} />
            </div>
          </div>
          {timeToTrade.medianDelaySeconds !== null && (
            <p className="numeral mt-4 text-sm text-ink-2">
              Median delay to trade: <span className="text-ink-1">{formatDuration(timeToTrade.medianDelaySeconds)}</span>
            </p>
          )}
          <div className="mt-4">
            <BarList
              items={timeToTrade.distribution.map(b => ({ label: b.label, value: b.count }))}
              formatValue={compact}
              emptyLabel="No notification→trade conversions found within 24 hours"
            />
          </div>
        </section>

        <section aria-label="Engaged users detail" className="mt-8 border-t border-hairline/60 pt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Engaged users detail</h2>
          <div className="mt-3">
            <DataTable
              rowKey={row => row.walletAddress}
              rows={users}
              columns={[
                { key: 'wallet', header: 'Wallet', render: r => shortWallet(r.walletAddress) },
                { key: 'taps', header: 'Taps', align: 'right', render: r => compact(r.taps) },
                { key: 'sent', header: 'Sent', align: 'right', render: r => compact(r.sent) },
                { key: 'rate', header: 'Tap rate', align: 'right', render: r => pct(r.tapRatePct) },
              ]}
            />
          </div>
        </section>
      </div>
    </main>
  )
}
