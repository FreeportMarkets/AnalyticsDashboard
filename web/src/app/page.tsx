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
import { RANGES, addDays, isRangeKey, rangeSpanDays, rangeStart, todayNy, type RangeKey } from '@/lib/ranges'
import { hlVolumeKpi } from '@/lib/metrics/hlVolumeRead'
import { AutoRefresh } from '@/components/AutoRefresh'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { StatTile } from '@/components/StatTile'
import { BarList } from '@/components/BarList'
import { DataTable } from '@/components/DataTable'
import { StalenessBadge } from '@/components/StalenessBadge'
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

const usd = (n: number) =>
  `$${Math.round(n).toLocaleString('en-US')}`
const compact = (n: number) => Math.round(n).toLocaleString('en-US')

/**
 * Abbreviated USD for width-constrained slots: the KPI tile, and the value
 * column of the three-up volume bar lists.
 *
 * A full `$89,794,594` at the KPI tile's type size does not fit one fifth
 * of the content width and ellipsised, so the headline volume figure was
 * the one number on the page you could not read. In the three-up row the
 * same string ate ~110px of a ~430px column and squeezed the bar to a stub.
 *
 * The `Daily detail` table still renders `usd()` in full -- it has the
 * width, and it is the surface you go to for exact figures -- and every
 * abbreviated slot carries the exact value in a hover title.
 */
const usdAbbrev = (n: number) => {
  const abs = Math.abs(n)
  if (abs >= 1_000_000_000) return `$${(n / 1_000_000_000).toFixed(2)}B`
  if (abs >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`
  if (abs >= 10_000) return `$${Math.round(n / 1_000).toLocaleString('en-US')}K`
  return usd(n)
}

/**
 * Every volume figure on this page is the DB reconstruction of perps
 * notional (`VOLUME_USD_EXPR` in lib/metrics/trades.ts, imported by
 * overview.ts) -- it multiplies margin by leverage, not Hyperliquid's
 * authoritative per-fill data, and runs ~15% high as a result. Same "est."
 * tag as the Trades page, so the two pages read consistently.
 *
 * Styled as quiet secondary text (ink-2, no amber, no filled background),
 * NOT the alert color -- `--color-alert` is reserved exclusively for
 * staleness/alert states (see StalenessBadge). This is a routine caveat
 * about methodology, not a warning, and shouldn't read like one.
 */
const EST_TAG_TITLE =
  "Perps stores amount_usd as margin, not notional, so volume here is reconstructed as margin × leverage (opens) or size × price (closes). That runs roughly 15% above Hyperliquid's actual per-fill data, which reflects filled size rather than the intended order size this reconstruction uses. Swap volume is exact."

function EstTag() {
  return (
    <span
      className="numeral ml-1.5 align-middle text-[10px] uppercase tracking-wide text-ink-2"
      title={EST_TAG_TITLE}
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

  const [session, end] = [await auth(), todayNy()]
  const start = rangeStart(end, range)
  const prevEnd = addDays(start, -1)
  const prevStart = rangeStart(prevEnd, range)

  const [kpis, platforms, events, series, hourly, trades, ages, hlVol] = await Promise.all([
    kpiSummary(start, end),
    platformSplit(start, end),
    topEvents(start, end, 8),
    dailySeries(start, end),
    hourlyActivity(start, end),
    tradeSummary(start, end),
    watermarkAge(),
    hlVolumeKpi(start, end, prevStart, prevEnd),
  ])

  const maxHour = Math.max(...hourly.map(h => h.count), 1)

  // Prefer the HL builder-fee-authoritative volume (exact, per-fill) once the
  // backfill has populated wallet_volume_daily; fall back to the DB
  // reconstruction (labeled "est.") until then. See docs/volume-tracking.md.
  const volume = hlVol.hasData
    ? { current: hlVol.current, previous: hlVol.previous, source: 'hl' as const }
    : { current: kpis.volumeUsd.current, previous: kpis.volumeUsd.previous, source: 'est' as const }

  return (
    <main className="mx-auto w-full max-w-[1600px] px-8 py-8">
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
            <AutoRefresh intervalMs={60_000} />
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
        <section aria-label="Key metrics" className="mt-8 grid gap-x-6 divide-y divide-hairline sm:grid-cols-5 sm:divide-x sm:divide-y-0">
          <StatTile
            label="Events"
            value={kpis.events.current}
            previousValue={kpis.events.previous}
            format={compact}
          />
          <div className="sm:pl-6">
            <StatTile
              label="Active users"
              value={kpis.users.current}
              previousValue={kpis.users.previous}
              format={compact}
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
            />
          </div>
          <div className="sm:pl-6">
            <StatTile
              /* Authoritative (HL per-fill) drops the "est."; the DB-
                 reconstruction fallback keeps it. */
              label={volume.source === 'hl' ? 'Volume' : 'Volume (est.)'}
              value={volume.current}
              previousValue={volume.previous}
              format={usdAbbrev}
              valueTitle={usd(volume.current)}
            />
          </div>
        </section>
        {/* Stated once for the whole row. Each tile used to repeat "vs prior
            period" under its own delta -- five identical captions competing
            with the five figures they annotate. */}
        <p className="mt-3 text-xs text-ink-3">
          Deltas compare against the prior {rangeSpanDays(range)} days.{' '}
          {volume.source === 'hl' ? (
            <>Volume is Hyperliquid per-fill data attributed by builder fee — every fill through Freeport, exact.</>
          ) : (
            <>
              <span className="text-ink-2" title={EST_TAG_TITLE}>est.</span>{' '}
              volume is reconstructed from the trade log (authoritative HL volume not yet backfilled).
            </>
          )}
        </p>

        <section aria-label="Daily events" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading meta={NY_TZ}>Daily events</SectionHeading>
          <div className="mt-4">
            <TimeSeriesBars
              data={series.map(d => ({ day: d.day, value: d.events }))}
              formatValue={compact}
              formatDay={shortDay}
            />
          </div>
        </section>

        {/*
          Grouped by ROW HEIGHT, not by topic. These three breakdowns each
          return ~3 rows, so they balance in a three-up row. Previously
          Platform split (3 rows) was paired against Top events (8 rows) in
          a two-column grid, which left a ~200px hole under the left column
          on every render, and Top events -- the densest list on the page --
          was squeezed into half width. Top events now runs full width
          below, where 8 rows and a long event name both fit.
        */}
        <section
          aria-label="Breakdowns"
          className="mt-10 grid items-start gap-x-10 gap-y-8 divide-y divide-hairline border-t border-hairline pt-8 xl:grid-cols-3 xl:divide-x xl:divide-y-0"
        >
          <div>
            <SectionHeading>Platform split</SectionHeading>
            <div className="mt-3">
              <BarList
                items={platforms.map(p => ({ label: p.platform, value: p.events, sublabel: `${compact(p.users)}u` }))}
                formatValue={compact}
              />
            </div>
          </div>
          <div className="pt-8 xl:pl-10 xl:pt-0">
            <SectionHeading meta={<EstTag />}>Volume by type</SectionHeading>
            <div className="mt-3">
              <BarList
                items={trades.byType.map(t => ({ label: t.type, value: t.volumeUsd, sublabel: `${compact(t.count)}x` }))}
                formatValue={usdAbbrev}
                formatValueTitle={usd}
              />
            </div>
          </div>
          <div className="pt-8 xl:pl-10 xl:pt-0">
            <SectionHeading meta={<EstTag />}>Volume by client</SectionHeading>
            <div className="mt-3">
              <BarList
                items={trades.byClient.map(t => ({ label: t.client, value: t.volumeUsd, sublabel: `${compact(t.count)}x` }))}
                formatValue={usdAbbrev}
                formatValueTitle={usd}
              />
            </div>
          </div>
        </section>

        <section aria-label="Top events" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Top events</SectionHeading>
          <div className="mt-3">
            <BarList
              items={events.map(e => ({ label: e.event, value: e.count }))}
              formatValue={compact}
            />
          </div>
        </section>

        <section aria-label="Hourly activity" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading meta={NY_TZ}>Hourly activity</SectionHeading>
          <div className="mt-4 flex h-20 items-end gap-[3px]">
            {hourly.map(h => (
              <div key={h.hour} className="group relative h-full flex-1">
                <div
                  className="absolute inset-x-0 bottom-0 rounded-t-sm bg-accent-bar transition-colors group-hover:bg-accent"
                  /* A zero hour renders NOTHING, not a 1% stub. The old
                     floor drew a sliver for count === 0 that was visually
                     identical to the 3% floor for a real trace value, so
                     "nobody used the app at 4am" and "a handful of people
                     did" looked the same. */
                  style={{ height: h.count > 0 ? `${Math.max((h.count / maxHour) * 100, 3)}%` : '0%' }}
                />
                <span className="pointer-events-none absolute -top-6 left-1/2 hidden -translate-x-1/2 whitespace-nowrap rounded-sm border border-hairline bg-raised px-1.5 py-0.5 text-[10px] text-ink-1 group-hover:block">
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

        <section aria-label="Daily detail" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Daily detail</SectionHeading>
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
