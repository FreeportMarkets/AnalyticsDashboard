import { sql } from '@/lib/db'
import { nyDateExpr } from '@/lib/time'
import { nyRangeToUtc } from './nyRange'
import { VOLUME_USD_EXPR } from './trades'

/**
 * Same two rules as queries.ts: bucket with nyDateExpr('ts'), filter `ts`
 * with nyRangeToUtc bounds. Never GROUP BY or range-filter on `date` --
 * scripts/lint-no-date-grouping.ts enforces this.
 *
 * System-wallet exclusion mirrors the existing dailyActiveUsers filter
 * exactly, and is applied to every person-shaped count (unique users,
 * sessions, trades, volume). Raw event/hourly totals are left unfiltered,
 * matching dailyEventCounts's existing behavior.
 *
 * VOLUME: every volume figure below sums `VOLUME_USD_EXPR` (imported from
 * trades.ts), never raw `amount_usd`. `amount_usd` on a perps row is MARGIN,
 * not notional -- summing it directly understates volume by ~40%+ (fixed
 * 2026-07-20; Overview and Trades disagreed on the same window). Like the
 * Trades page, this reconstruction is an ESTIMATE (~15% high vs Hyperliquid's
 * authoritative per-fill volume) and must be labeled "est." wherever shown.
 *
 * Every trade/volume query here also filters `type IN ('swap', 'perps')`,
 * matching trades.ts's `trades_only = pd.concat([swap_df, perps_df])` scope
 * (app.py) -- the `trades` table also carries `type = 'deposit'` rows, which
 * are not trades and must not be counted as trade volume (they were the
 * second half of the Overview-vs-Trades mismatch: 85 deposit rows / ~$13k
 * inflated the 30d "Trades" KPI before this fix).
 */
const SYSTEM_WALLETS = ['server', 'unknown', 'system', '']

function addDays(date: string, days: number): string {
  const d = new Date(`${date}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

function daysBetweenInclusive(startDate: string, endDate: string): number {
  const s = new Date(`${startDate}T00:00:00Z`).getTime()
  const e = new Date(`${endDate}T00:00:00Z`).getTime()
  return Math.round((e - s) / 86_400_000) + 1
}

/** Every NY calendar date from startDate to endDate, inclusive. */
function dateRange(startDate: string, endDate: string): string[] {
  const out: string[] = []
  let cur = startDate
  while (cur <= endDate) {
    out.push(cur)
    cur = addDays(cur, 1)
  }
  return out
}

/** The immediately preceding period of equal length, as an NY calendar range. */
function previousPeriod(startDate: string, endDate: string): { start: string; end: string } {
  const days = daysBetweenInclusive(startDate, endDate)
  const end = addDays(startDate, -1)
  const start = addDays(end, -(days - 1))
  return { start, end }
}

export interface KpiValue {
  current: number
  previous: number
}

export interface KpiSummary {
  events: KpiValue
  users: KpiValue
  sessions: KpiValue
  trades: KpiValue
  volumeUsd: KpiValue
}

export async function kpiSummary(startDate: string, endDate: string): Promise<KpiSummary> {
  const { start: prevStart, end: prevEnd } = previousPeriod(startDate, endDate)
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const { fromUtc: prevFromUtc } = nyRangeToUtc(prevStart, prevEnd)

  const eventRows = (await sql(
    `SELECT
        count(*) FILTER (WHERE ts >= $3)::int AS events_current,
        count(*) FILTER (WHERE ts < $3)::int AS events_previous,
        count(DISTINCT wallet_address) FILTER (
          WHERE ts >= $3 AND wallet_address IS NOT NULL AND wallet_address <> ALL($4::text[])
            AND (platform IS NULL OR platform <> 'server')
        )::int AS users_current,
        count(DISTINCT wallet_address) FILTER (
          WHERE ts < $3 AND wallet_address IS NOT NULL AND wallet_address <> ALL($4::text[])
            AND (platform IS NULL OR platform <> 'server')
        )::int AS users_previous,
        count(DISTINCT session_id) FILTER (
          WHERE ts >= $3 AND session_id IS NOT NULL AND wallet_address <> ALL($4::text[])
            AND (platform IS NULL OR platform <> 'server')
        )::int AS sessions_current,
        count(DISTINCT session_id) FILTER (
          WHERE ts < $3 AND session_id IS NOT NULL AND wallet_address <> ALL($4::text[])
            AND (platform IS NULL OR platform <> 'server')
        )::int AS sessions_previous
       FROM events
      WHERE ts >= $1 AND ts < $2`,
    [prevFromUtc.toISOString(), toUtc.toISOString(), fromUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{
    events_current: number
    events_previous: number
    users_current: number
    users_previous: number
    sessions_current: number
    sessions_previous: number
  }>

  const tradeRows = (await sql(
    `SELECT
        count(*) FILTER (WHERE ts >= $3 AND wallet_address <> ALL($4::text[]))::int AS trades_current,
        count(*) FILTER (WHERE ts < $3 AND wallet_address <> ALL($4::text[]))::int AS trades_previous,
        coalesce(sum(${VOLUME_USD_EXPR}) FILTER (WHERE ts >= $3 AND wallet_address <> ALL($4::text[])), 0)::float8 AS volume_current,
        coalesce(sum(${VOLUME_USD_EXPR}) FILTER (WHERE ts < $3 AND wallet_address <> ALL($4::text[])), 0)::float8 AS volume_previous
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type IN ('swap', 'perps')`,
    [prevFromUtc.toISOString(), toUtc.toISOString(), fromUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{
    trades_current: number
    trades_previous: number
    volume_current: number
    volume_previous: number
  }>

  const e = eventRows[0]
  const t = tradeRows[0]
  if (!e || !t) throw new Error('kpiSummary: aggregate query returned no rows')

  return {
    events: { current: e.events_current, previous: e.events_previous },
    users: { current: e.users_current, previous: e.users_previous },
    sessions: { current: e.sessions_current, previous: e.sessions_previous },
    trades: { current: t.trades_current, previous: t.trades_previous },
    volumeUsd: { current: t.volume_current, previous: t.volume_previous },
  }
}

export async function platformSplit(
  startDate: string, endDate: string
): Promise<Array<{ platform: string; users: number; events: number }>> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT coalesce(platform, 'unknown') AS platform,
            count(DISTINCT wallet_address) FILTER (
              WHERE wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
            )::int AS users,
            count(*)::int AS events
       FROM events
      WHERE ts >= $1 AND ts < $2
        AND (platform IS NULL OR platform <> 'server')
      GROUP BY 1
      ORDER BY events DESC`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ platform: string; users: number; events: number }>
  return rows
}

export async function topEvents(
  startDate: string, endDate: string, limit: number
): Promise<Array<{ event: string; count: number }>> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT event, count(*)::int AS count
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY count DESC
      LIMIT $3`,
    [fromUtc.toISOString(), toUtc.toISOString(), limit]
  )) as Array<{ event: string; count: number }>
  return rows
}

export interface DailySeriesRow {
  day: string
  events: number
  users: number
  trades: number
  volumeUsd: number
}

export async function dailySeries(startDate: string, endDate: string): Promise<DailySeriesRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const eventRows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day,
            count(*)::int AS events,
            count(DISTINCT wallet_address) FILTER (
              WHERE wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
                AND (platform IS NULL OR platform <> 'server')
            )::int AS users
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; events: number; users: number }>

  const tradeRows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day,
            count(*) FILTER (WHERE wallet_address <> ALL($3::text[]))::int AS trades,
            coalesce(sum(${VOLUME_USD_EXPR}) FILTER (WHERE wallet_address <> ALL($3::text[])), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type IN ('swap', 'perps')
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; trades: number; volume_usd: number }>

  const dayKey = (d: string | Date) => (typeof d === 'string' ? d : d.toISOString().slice(0, 10))
  const events = new Map(eventRows.map(r => [dayKey(r.day), r]))
  const trades = new Map(tradeRows.map(r => [dayKey(r.day), r]))

  return dateRange(startDate, endDate).map(day => ({
    day,
    events: events.get(day)?.events ?? 0,
    users: events.get(day)?.users ?? 0,
    trades: trades.get(day)?.trades ?? 0,
    volumeUsd: trades.get(day)?.volume_usd ?? 0,
  }))
}

export async function hourlyActivity(
  startDate: string, endDate: string
): Promise<Array<{ hour: number; count: number }>> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT extract(hour FROM (ts AT TIME ZONE 'America/New_York'))::int AS hour,
            count(*)::int AS count
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ hour: number; count: number }>

  const byHour = new Map(rows.map(r => [r.hour, r.count]))
  return Array.from({ length: 24 }, (_, hour) => ({ hour, count: byHour.get(hour) ?? 0 }))
}

export interface TradeSummary {
  byType: Array<{ type: string; count: number; volumeUsd: number }>
  byClient: Array<{ client: string; count: number; volumeUsd: number }>
}

export async function tradeSummary(startDate: string, endDate: string): Promise<TradeSummary> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const byType = (await sql(
    `SELECT coalesce(type, 'unknown') AS type,
            count(*)::int AS count,
            coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2 AND wallet_address <> ALL($3::text[])
        AND type IN ('swap', 'perps')
      GROUP BY 1
      ORDER BY volume_usd DESC`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ type: string; count: number; volume_usd: number }>

  const byClient = (await sql(
    `SELECT coalesce(client, 'unknown') AS client,
            count(*)::int AS count,
            coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2 AND wallet_address <> ALL($3::text[])
        AND type IN ('swap', 'perps')
      GROUP BY 1
      ORDER BY volume_usd DESC`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ client: string; count: number; volume_usd: number }>

  return {
    byType: byType.map(r => ({ type: r.type, count: r.count, volumeUsd: r.volume_usd })),
    byClient: byClient.map(r => ({ client: r.client, count: r.count, volumeUsd: r.volume_usd })),
  }
}
