import { auth, signOut } from '@/auth'
import {
  volumeSummary,
  dailyVolume,
  topAssets,
  venueSplit,
  recentTrades,
  depositSummary,
} from '@/lib/metrics/trades'
import { watermarkAge } from '@/lib/metrics/staleness'
import { NY_TZ } from '@/lib/time'
import { RANGES, isRangeKey, rangeStart, todayNy, type RangeKey } from '@/lib/ranges'
import { sql } from '@/lib/db'
import { fetchWalletIdentities } from '@/lib/privyIdentities'
import { hlVolumeTotal, hlVolumeDaily } from '@/lib/metrics/hlVolumeRead'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { StatTile } from '@/components/StatTile'
import { BarList } from '@/components/BarList'
import { DataTable } from '@/components/DataTable'
import { TimeSeriesLine } from '@/components/TimeSeriesLine'
import { StalenessBadge } from '@/components/StalenessBadge'
import { TraderCell } from '@/components/TraderCell'
import { AutoRefresh } from '@/components/AutoRefresh'

export const dynamic = 'force-dynamic'

function formatDateRange(start: string, end: string): string {
  const fmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' })
  const startLabel = fmt.format(new Date(`${start}T12:00:00Z`))
  const endLabel = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    .format(new Date(`${end}T12:00:00Z`))
  return `${startLabel} – ${endLabel}`
}

function formatTradeTs(iso: string): string {
  return new Intl.DateTimeFormat('en-US', {
    timeZone: NY_TZ, month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true,
  }).format(new Date(iso))
}

const usd = (n: number) => `$${Math.round(n).toLocaleString('en-US')}`
const compact = (n: number) => Math.round(n).toLocaleString('en-US')
const pct1 = (n: number) => `${(n * 100).toFixed(1)}%`
const num2 = (n: number) => n.toLocaleString('en-US', { maximumFractionDigits: 2 })

/**
 * The DB reconstruction of perps notional (`_volume_usd` in app.py) runs
 * ~15% hot vs Hyperliquid's authoritative per-fill volume -- it multiplies
 * intended order size by leverage, not filled size. This dashboard doesn't
 * call the HL API (follow-up), so every perps-derived volume figure carries
 * this tag rather than being presented as ground truth. Swap volume is exact
 * (amount_usd, untouched) and is never tagged.
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

export default async function TradesPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const params = await searchParams
  const rawRange = typeof params.range === 'string' ? params.range : undefined
  const range: RangeKey = isRangeKey(rawRange) ? rawRange : '7d'

  const [session, end] = [await auth(), todayNy()]
  const start = rangeStart(end, range)

  const [vol, daily, assets, venues, recent, deposits, ages, hlTotal, hlDaily] = await Promise.all([
    volumeSummary(start, end),
    dailyVolume(start, end),
    topAssets(start, end, 15),
    venueSplit(start, end),
    recentTrades(start, end, 50),
    depositSummary(start, end),
    watermarkAge(),
    hlVolumeTotal(start, end).catch(() => ({ notionalUsd: 0, builderFeeUsd: 0, fillCount: 0 })),
    hlVolumeDaily(start, end).catch(() => [] as Array<{ day: string; notionalUsd: number; fillCount: number }>),
  ])

  // Prefer HL builder-fee-authoritative PERPS volume once the backfill has
  // populated the table; swap volume is always exact from our own DB. Falls
  // back to the reconstruction (labeled "est.") until then. Mirrors Overview.
  const hlHasData = hlTotal.fillCount > 0
  const hlDailyByDay = new Map(hlDaily.map(d => [d.day, d.notionalUsd]))
  // Identity lookup is scoped to just the wallets on this page (a single
  // indexed query against the privy_identities mirror), not a 14s live Privy
  // fetch of all ~5,830 wallets -- see src/lib/privyIdentities.ts. Runs after
  // `recent` resolves since it needs those wallet addresses.
  const privyMap = await fetchWalletIdentities(sql, recent.map(r => r.walletAddress))

  const swap = vol.byType.find(t => t.type === 'swap') ?? { type: 'swap', count: 0, volumeUsd: 0 }
  const perpsDb = vol.byType.find(t => t.type === 'perps') ?? { type: 'perps', count: 0, volumeUsd: 0 }

  // Authoritative when available, reconstruction otherwise. `perpsVolume` and
  // `totalVolume` drive every headline figure; the surface breakdown below
  // stays reconstruction-based (HL fills carry no client tag) and keeps "est."
  const perpsVolume = hlHasData ? hlTotal.notionalUsd : perpsDb.volumeUsd
  const totalVolume = perpsVolume + swap.volumeUsd
  const volSource: 'hl' | 'est' = hlHasData ? 'hl' : 'est'
  const perpsLabel = volSource === 'hl' ? 'Perps Volume' : 'Perps Volume (est.)'
  const totalLabel = volSource === 'hl' ? 'Trading Volume' : 'Trading Volume (est.)'

  // Web Terminal perps volume. HL fills carry no client tag, so the mobile/web
  // split comes from our trade log (the x-client header). We take web's SHARE
  // of perps volume from the log and apply it to the authoritative perps total,
  // so the terminal number is consistent with the headline rather than a raw
  // (lower) reconstruction figure.
  const perpsLogTotal = vol.perpsByClient.reduce((s, c) => s + c.volumeUsd, 0)
  const webLog = vol.perpsByClient.find(c => c.client === 'web')
  const webShare = perpsLogTotal > 0 ? (webLog?.volumeUsd ?? 0) / perpsLogTotal : 0
  const webTerminalVolume = perpsVolume * webShare
  const webTerminalTrades = webLog?.count ?? 0

  return (
    <main className="mx-auto w-full max-w-[1600px] px-8 py-8">
      <PageHeader
        title="Trades & Volume"
        subtitle={
          <div className="flex flex-wrap items-center gap-3">
            <span>{formatDateRange(start, end)} · {NY_TZ}</span>
            <nav aria-label="Date range" className="flex items-center gap-1">
              {(Object.keys(RANGES) as RangeKey[]).map(key => (
                <a
                  key={key}
                  href={key === '7d' ? '/trades' : `/trades?range=${key}`}
                  aria-current={key === range ? 'true' : undefined}
                  className={`numeral rounded-sm px-2 py-0.5 text-xs outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent ${
                    key === range ? 'bg-raised font-medium text-ink-1' : 'text-ink-3 hover:text-ink-1'
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
        {/* --- Volume KPIs --- */}
        <section aria-label="Volume KPIs" className="mt-8 grid gap-x-6 divide-y divide-hairline sm:grid-cols-4 sm:divide-x sm:divide-y-0">
          <StatTile
            label={totalLabel}
            value={totalVolume}
            format={usd}
          />
          <div className="sm:pl-6">
            <StatTile label="Total Trades" value={vol.totalTrades} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Unique Traders" value={vol.uniqueTraders} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label={volSource === 'hl' ? 'Avg Trade Size' : 'Avg Trade Size (est.)'}
              value={vol.totalTrades > 0 ? totalVolume / vol.totalTrades : 0}
              format={usd}
            />
          </div>
        </section>
        <p className="mt-2 text-xs text-ink-3">
          {volSource === 'hl' ? (
            <>Perps volume is Hyperliquid per-fill data (builder-fee attributed); swap volume is exact.</>
          ) : (
            <>
              <span className="text-ink-2" title={EST_TAG_TITLE}>est.</span>{' '}
              figures include reconstructed perps volume (authoritative HL volume not yet backfilled).
            </>
          )}
        </p>

        {/* --- Perps vs Swaps vs Terminal --- */}
        <section aria-label="Volume split" className="mt-8 grid gap-x-6 divide-y divide-hairline border-t border-hairline pt-8 sm:grid-cols-3 sm:divide-x sm:divide-y-0">
          <StatTile label={perpsLabel} value={perpsVolume} format={usd} />
          <div className="sm:pl-6">
            <StatTile label="Swap Volume" value={swap.volumeUsd} format={usd} />
          </div>
          <div className="sm:pl-6">
            <StatTile
              label="Web Terminal Volume"
              value={webTerminalVolume}
              format={usd}
              valueTitle={`${(webShare * 100).toFixed(1)}% of perps volume placed from the web terminal (per the x-client tag), applied to the authoritative perps total.`}
            />
          </div>
        </section>
        <p className="numeral mt-2 text-xs text-ink-3">
          {compact(perpsDb.count)} perps orders · {compact(swap.count)} swaps · {compact(webTerminalTrades)} from web terminal
        </p>

        {/* --- Mobile vs Web --- */}
        <section aria-label="Perps volume by surface" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>
            Perps volume by surface <EstTag />
          </SectionHeading>
          <p className="mt-1 text-xs text-ink-3">
            Tagged via the x-client header at order placement. &ldquo;Untagged&rdquo; = orders placed before the tag shipped, plus liquidations / TP-SL auto-fills.
          </p>
          <div className="mt-3">
            <BarList
              items={vol.perpsByClient.map(c => ({
                label: c.client,
                value: c.volumeUsd,
                sublabel: `${compact(c.count)}x`,
              }))}
              formatValue={usd}
            />
          </div>
        </section>

        {/* --- Daily volume --- */}
        <section aria-label="Daily volume" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading meta={volSource === 'hl' ? undefined : <EstTag />}>
            Daily trades & volume
          </SectionHeading>
          <div className="mt-4">
            <TimeSeriesLine
              data={daily.map(d => {
                // Authoritative HL perps per day when backfilled, else the
                // reconstruction. Swap is always exact.
                const perpsForDay = hlHasData ? (hlDailyByDay.get(d.day) ?? 0) : d.perpsVolumeUsd
                const totalForDay = perpsForDay + d.swapVolumeUsd
                return {
                  day: d.day,
                  value: totalForDay,
                  tooltip: (
                    <>
                      <span className="numeral">{usd(totalForDay)} total</span>
                      <span className="text-ink-3">{compact(d.tradeCount)} trades</span>
                      <span className="text-ink-3">perps {usd(perpsForDay)} · swap {usd(d.swapVolumeUsd)}</span>
                    </>
                  ),
                }
              })}
              formatValue={usd}
              formatDay={d => new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' }).format(new Date(`${d}T12:00:00Z`))}
            />
          </div>
          <div className="mt-5">
            <DataTable
              rowKey={row => row.day}
              rows={[...daily].reverse()}
              columns={[
                { key: 'day', header: 'Day', render: r => r.day },
                { key: 'trades', header: 'Trades', align: 'right', render: r => compact(r.tradeCount) },
                { key: 'swap', header: 'Swap Volume', align: 'right', render: r => usd(r.swapVolumeUsd) },
                {
                  key: 'perps',
                  header: volSource === 'hl' ? 'Perps Volume' : 'Perps Volume (est.)',
                  align: 'right',
                  render: r => usd(hlHasData ? (hlDailyByDay.get(r.day) ?? 0) : r.perpsVolumeUsd),
                },
                {
                  key: 'total',
                  header: volSource === 'hl' ? 'Total' : 'Total (est.)',
                  align: 'right',
                  render: r => usd((hlHasData ? (hlDailyByDay.get(r.day) ?? 0) : r.perpsVolumeUsd) + r.swapVolumeUsd),
                },
              ]}
            />
          </div>
        </section>

        {/* --- Top assets / Venue split --- */}
        <section
          aria-label="Perps breakdowns"
          className="mt-8 grid items-start gap-x-10 gap-y-8 divide-y divide-hairline border-t border-hairline pt-8 md:grid-cols-2 md:divide-x md:divide-y-0"
        >
          <div>
            <SectionHeading>
              Top assets by volume <EstTag />
            </SectionHeading>
            <div className="mt-3">
              <BarList
                items={assets.map(a => ({ label: a.asset, value: a.volumeUsd, sublabel: `${compact(a.tradeCount)}x` }))}
                formatValue={usd}
              />
            </div>
          </div>
          <div className="pt-8 md:pl-10 md:pt-0">
            <SectionHeading>
              Volume by venue <EstTag />
            </SectionHeading>
            <div className="mt-3">
              <BarList
                items={venues.map(v => ({ label: v.venue, value: v.volumeUsd, sublabel: `${compact(v.tradeCount)}x` }))}
                formatValue={usd}
              />
            </div>
          </div>
        </section>

        {/* --- Recent trades --- */}
        <section aria-label="Recent trades" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Recent trades</SectionHeading>
          <div className="mt-3">
            <DataTable
              /* The one table that keeps an inner scroll: 50 rows of trades
                 sitting between two other sections would push the Deposits
                 funnel roughly 1800px down the page. Every other table in
                 the app now scrolls with the page instead. */
              maxHeight="32rem"
              rowKey={row => `${row.ts}-${row.walletAddress}-${row.asset}-${row.side ?? ''}-${row.volumeUsd}`}
              rows={recent}
              columns={[
                { key: 'ts', header: 'Time', render: r => formatTradeTs(r.ts) },
                { key: 'trader', header: 'Trader', render: r => <TraderCell wallet={r.walletAddress} privyMap={privyMap} /> },
                { key: 'type', header: 'Type', render: r => r.type },
                { key: 'asset', header: 'Asset', render: r => r.asset },
                { key: 'side', header: 'Side', render: r => r.side ?? '—' },
                { key: 'size', header: 'Size', align: 'right', render: r => (r.size == null ? '—' : num2(r.size)) },
                { key: 'price', header: 'Price', align: 'right', render: r => (r.price == null ? '—' : usd(r.price)) },
                { key: 'leverage', header: 'Lev', align: 'right', render: r => (r.leverage == null ? '—' : `${num2(r.leverage)}x`) },
                { key: 'client', header: 'Client', render: r => r.client },
                { key: 'venue', header: 'Venue', render: r => r.venue ?? '—' },
                {
                  key: 'volume',
                  header: 'Volume',
                  align: 'right',
                  render: r => <>{usd(r.volumeUsd)}{r.type === 'perps' && <EstTag />}</>,
                },
              ]}
            />
          </div>
        </section>

        {/* --- Deposits funnel --- */}
        <section aria-label="Deposits" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Deposits</SectionHeading>
          <div className="mt-4 grid gap-x-6 divide-y divide-hairline sm:grid-cols-4 sm:divide-x sm:divide-y-0">
            <StatTile label="Initiated" value={deposits.initiated} format={compact} />
            <div className="sm:pl-6">
              <StatTile label="Success" value={deposits.success} format={compact} />
            </div>
            <div className="sm:pl-6">
              <StatTile label="Error" value={deposits.error} format={compact} />
            </div>
            <div className="sm:pl-6">
              <StatTile label="Conversion" value={deposits.conversionRate * 100} format={n => pct1(n / 100)} />
            </div>
          </div>
          <div className="mt-6">
            <SectionHeading as="h3">By provider</SectionHeading>
            <div className="mt-3">
              <DataTable
                rowKey={row => row.provider}
                rows={deposits.byProvider}
                columns={[
                  { key: 'provider', header: 'Provider', render: r => r.provider },
                  { key: 'initiated', header: 'Initiated', align: 'right', render: r => compact(r.initiated) },
                  { key: 'success', header: 'Success', align: 'right', render: r => compact(r.success) },
                  { key: 'error', header: 'Error', align: 'right', render: r => compact(r.error) },
                  {
                    key: 'conversion',
                    header: 'Conversion',
                    align: 'right',
                    render: r => (r.initiated > 0 ? pct1(r.success / r.initiated) : '—'),
                  },
                ]}
              />
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}
