import { sql } from '@/lib/db'
import { nyDateExpr } from '@/lib/time'
import { nyRangeToUtc } from './nyRange'

/**
 * Trades & Volume metrics -- ported verbatim from app.py's `with tab_trades:`
 * block (lines ~2050-2403) and its `apply_perps_leverage` / `_detect_ostium`
 * helpers (~136-173). Same two rules as queries.ts and overview.ts: bucket
 * with nyDateExpr('ts'), filter `ts` with nyRangeToUtc bounds, never GROUP BY
 * or range-filter on the `date` column.
 *
 * ------------------------------------------------------------------------
 * THE TRAP (read before touching VOLUME_USD_EXPR):
 *
 * `amount_usd` on a perps row is MARGIN, not notional. The Streamlit source
 * reconstructs notional with `apply_perps_leverage`:
 *   - opens (non-Ostium):  amount_usd * leverage
 *   - closes (non-Ostium): |size| * price          (full position notional)
 *   - Ostium is special: `size` is already USD notional there (not base
 *     units like HL/Lighter), so opens keep amount_usd as-is and closes use
 *     |size| directly instead of |size| * price.
 *
 * This reconstruction (`_volume_usd` in app.py, `VOLUME_USD_EXPR` here) is
 * NOT the authoritative number. The real figure comes from Hyperliquid
 * `userFillsByTime` (hl_perp_volume_usd / hl_volume.py), summed live per
 * wallet and deduped by fill id -- that is what HL actually charges builder
 * fees on. The DB reconstruction runs ~15% HOT because trade rows store
 * *intended* order size, not *filled* size (partial/IOC fills drift it).
 *
 * Streamlit only uses the HL-accurate number on the Overview tab. Every
 * function below -- like ~20 call sites in app.py's Trades tab -- uses the
 * DB reconstruction. This port does the same and does NOT call the HL API
 * (out of scope for this pass; see the trades page for the "est." labels
 * this requires). Do not "fix" this by unifying the two numbers -- that
 * silently changes published figures and is a decision for the owner, not
 * an implementation detail.
 * ------------------------------------------------------------------------
 */

const SYSTEM_WALLETS = ['server', 'unknown', 'system', '']

/**
 * SQL CASE expression reconstructing per-row notional exactly like
 * `apply_perps_leverage` + `_detect_ostium` in app.py. Constant string, no
 * user input -- safe to inline directly into query text.
 *
 * `amount_usd` on a perps row is MARGIN, not notional -- this expression
 * multiplies by leverage (or falls back to size * price / raw size for
 * Ostium) to reconstruct notional. THIS IS THE ONLY VOLUME EXPRESSION IN
 * THE APP -- every volume figure, on every page, must sum this, not raw
 * `amount_usd`. A second copy of this CASE expression anywhere else is how
 * the Overview-vs-Trades volume mismatch bug (fixed 2026-07-20) recurs.
 */
export const VOLUME_USD_EXPR = `
    CASE
      WHEN type = 'perps' THEN
        CASE
          WHEN coalesce(is_close, false) THEN
            CASE WHEN lower(coalesce(category, '')) = 'ostium'
                 THEN abs(coalesce(size, 0))
                 ELSE abs(coalesce(size, 0)) * coalesce(price, 0)
            END
          ELSE
            CASE WHEN lower(coalesce(category, '')) = 'ostium'
                 THEN coalesce(amount_usd, 0)
                 ELSE coalesce(amount_usd, 0) * coalesce(leverage, 1)
            END
        END
      ELSE coalesce(amount_usd, 0)
    END
`.trim()

export interface VolumeSummary {
  totalVolumeUsd: number
  totalTrades: number
  uniqueTraders: number
  avgTradeSize: number
  byType: Array<{ type: string; count: number; volumeUsd: number }>
  /** Perps only -- client tag ('mobile'/'web') is only set on perp orders. */
  perpsByClient: Array<{ client: string; count: number; volumeUsd: number }>
}

/**
 * Total trading volume + count (swaps + perps only, deposits excluded --
 * matches `trades_only = pd.concat([swap_df, perps_df])` in app.py), plus
 * the by-type and perps-by-client splits. Every volume figure here is the
 * DB reconstruction (`_volume_usd`), not HL-authoritative -- callers must
 * label it "est."
 */
export async function volumeSummary(startDate: string, endDate: string): Promise<VolumeSummary> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const totalRows = (await sql(
    `SELECT
        count(*)::int AS total_trades,
        count(DISTINCT wallet_address)::int AS unique_traders,
        coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS total_volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type IN ('swap', 'perps')
        AND wallet_address <> ALL($3::text[])`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ total_trades: number; unique_traders: number; total_volume_usd: number }>

  const byTypeRows = (await sql(
    `SELECT type,
            count(*)::int AS count,
            coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type IN ('swap', 'perps')
        AND wallet_address <> ALL($3::text[])
      GROUP BY 1
      ORDER BY volume_usd DESC`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ type: string; count: number; volume_usd: number }>

  const byClientRows = (await sql(
    `SELECT coalesce(nullif(lower(client), ''), 'untagged') AS client,
            count(*)::int AS count,
            coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type = 'perps'
        AND wallet_address <> ALL($3::text[])
      GROUP BY 1
      ORDER BY volume_usd DESC`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ client: string; count: number; volume_usd: number }>

  const t = totalRows[0]
  if (!t) throw new Error('volumeSummary: aggregate query returned no rows')

  return {
    totalVolumeUsd: t.total_volume_usd,
    totalTrades: t.total_trades,
    uniqueTraders: t.unique_traders,
    avgTradeSize: t.total_trades > 0 ? t.total_volume_usd / t.total_trades : 0,
    byType: byTypeRows.map(r => ({ type: r.type, count: r.count, volumeUsd: r.volume_usd })),
    perpsByClient: byClientRows.map(r => ({ client: r.client, count: r.count, volumeUsd: r.volume_usd })),
  }
}

export interface DailyVolumeRow {
  day: string
  swapVolumeUsd: number
  perpsVolumeUsd: number
  tradeCount: number
}

/** Daily swap/perps volume (DB reconstruction, "est.") + trade count. */
export async function dailyVolume(startDate: string, endDate: string): Promise<DailyVolumeRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day,
            count(*)::int AS trade_count,
            coalesce(sum(${VOLUME_USD_EXPR}) FILTER (WHERE type = 'swap'), 0)::float8 AS swap_volume_usd,
            coalesce(sum(${VOLUME_USD_EXPR}) FILTER (WHERE type = 'perps'), 0)::float8 AS perps_volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type IN ('swap', 'perps')
        AND wallet_address <> ALL($3::text[])
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; trade_count: number; swap_volume_usd: number; perps_volume_usd: number }>

  return rows.map(r => ({
    day: typeof r.day === 'string' ? r.day : r.day.toISOString().slice(0, 10),
    swapVolumeUsd: r.swap_volume_usd,
    perpsVolumeUsd: r.perps_volume_usd,
    tradeCount: r.trade_count,
  }))
}

export interface TopAssetRow {
  asset: string
  volumeUsd: number
  tradeCount: number
}

/** Top perps assets by (estimated) volume. Mirrors "Volume by Asset" in app.py. */
export async function topAssets(startDate: string, endDate: string, limit = 15): Promise<TopAssetRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT coalesce(asset, 'unknown') AS asset,
            count(*)::int AS trade_count,
            coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type = 'perps'
        AND wallet_address <> ALL($3::text[])
      GROUP BY 1
      ORDER BY volume_usd DESC
      LIMIT $4`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS, limit]
  )) as Array<{ asset: string; trade_count: number; volume_usd: number }>

  return rows.map(r => ({ asset: r.asset, volumeUsd: r.volume_usd, tradeCount: r.trade_count }))
}

export interface VenueSplitRow {
  venue: string
  volumeUsd: number
  tradeCount: number
}

/**
 * Perps volume + count by venue (the `category` column: hl/lighter/ostium,
 * plus HIP-3 sector tags observed in the data). Category is only populated
 * on perps rows -- swaps carry no venue.
 */
export async function venueSplit(startDate: string, endDate: string): Promise<VenueSplitRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT coalesce(category, 'unknown') AS venue,
            count(*)::int AS trade_count,
            coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND type = 'perps'
        AND wallet_address <> ALL($3::text[])
      GROUP BY 1
      ORDER BY volume_usd DESC`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ venue: string; trade_count: number; volume_usd: number }>

  return rows.map(r => ({ venue: r.venue, volumeUsd: r.volume_usd, tradeCount: r.trade_count }))
}

export interface RecentTradeRow {
  ts: string
  type: string
  asset: string
  side: string | null
  size: number | null
  price: number | null
  leverage: number | null
  client: string
  status: string | null
  venue: string | null
  volumeUsd: number
  walletAddress: string
}

/**
 * Last N swap+perps trades. `asset` falls back to `to_token` for swaps
 * (which have no `asset`/`display_symbol`). `volumeUsd` is the same
 * reconstruction as everywhere else in this file -- "est." for perps rows.
 * `walletAddress` is selected (it's already used in the WHERE clause) so
 * callers can enrich the row with a Privy identity label -- see
 * `@/lib/privy`.
 *
 * `leverage` on a perps row falls back to the position's OPEN when the row
 * itself has none. The perps writer stopped logging `leverage` on close
 * fills around 2026-04-10..13 (verified against the raw DynamoDB dump in
 * AnalyticsDashboard/analytics_output/_raw_trades.json: every close through
 * 2026-04-09 carries leverage; from 2026-04-14 none do -- opens are
 * unaffected). A close has no leverage of its own anyway -- the position's
 * leverage is fixed at open -- so inheriting from the most recent prior
 * open of the same (wallet, symbol, side) is the true value, not an
 * estimate. In the April sample this resolves 213 of 224 leverage-less
 * closes; the misses are positions opened before the sample window. The
 * correlated subquery only runs for perps rows whose own leverage is NULL
 * (COALESCE short-circuits) and seeks the (wallet_address, timestamp) PK,
 * bounded by the outer LIMIT -- fine for a 50-row widget. Volume math is
 * untouched: closes never used leverage (`VOLUME_USD_EXPR` uses size*price).
 */
export async function recentTrades(startDate: string, endDate: string, limit = 50): Promise<RecentTradeRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT ts,
            type,
            coalesce(display_symbol, asset, to_token, 'unknown') AS asset,
            side,
            size::float8 AS size,
            price::float8 AS price,
            coalesce(
              t.leverage,
              CASE WHEN t.type = 'perps' THEN (
                SELECT o.leverage
                  FROM trades o
                 WHERE o.wallet_address = t.wallet_address
                   AND o.type = 'perps'
                   AND NOT coalesce(o.is_close, false)
                   AND o.leverage IS NOT NULL
                   AND coalesce(o.display_symbol, o.asset) = coalesce(t.display_symbol, t.asset)
                   AND coalesce(o.side, '') = coalesce(t.side, '')
                   AND o.ts <= t.ts
                 ORDER BY o.ts DESC
                 LIMIT 1
              ) END
            )::float8 AS leverage,
            coalesce(nullif(lower(client), ''), 'untagged') AS client,
            status,
            category AS venue,
            ${VOLUME_USD_EXPR} AS volume_usd,
            wallet_address
       FROM trades t
      WHERE ts >= $1 AND ts < $2
        AND type IN ('swap', 'perps')
        AND wallet_address <> ALL($3::text[])
      ORDER BY ts DESC
      LIMIT $4`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS, limit]
  )) as Array<{
    ts: Date
    type: string
    asset: string
    side: string | null
    size: number | null
    price: number | null
    leverage: number | null
    client: string
    status: string | null
    venue: string | null
    volume_usd: number
    wallet_address: string
  }>

  return rows.map(r => ({
    ts: r.ts.toISOString(),
    type: r.type,
    asset: r.asset,
    side: r.side,
    size: r.size,
    price: r.price,
    leverage: r.leverage,
    client: r.client,
    status: r.status,
    venue: r.venue,
    volumeUsd: r.volume_usd,
    walletAddress: r.wallet_address,
  }))
}

export interface DepositSummary {
  initiated: number
  success: number
  error: number
  conversionRate: number
  byProvider: Array<{ provider: string; initiated: number; success: number; error: number }>
}

/**
 * Deposit funnel, sourced from `events` (deposit_initiated / deposit_success
 * / deposit_error), not `trades` -- deposits are not written to the trades
 * table in this schema. `metadata->>'provider'` gives the provider split
 * where present.
 */
export async function depositSummary(startDate: string, endDate: string): Promise<DepositSummary> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const totalRows = (await sql(
    `SELECT
        count(*) FILTER (WHERE event = 'deposit_initiated')::int AS initiated,
        count(*) FILTER (WHERE event = 'deposit_success')::int AS success,
        count(*) FILTER (WHERE event = 'deposit_error')::int AS error
       FROM events
      WHERE ts >= $1 AND ts < $2
        AND event IN ('deposit_initiated', 'deposit_success', 'deposit_error')`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ initiated: number; success: number; error: number }>

  const byProviderRows = (await sql(
    `SELECT coalesce(metadata->>'provider', 'unknown') AS provider,
            count(*) FILTER (WHERE event = 'deposit_initiated')::int AS initiated,
            count(*) FILTER (WHERE event = 'deposit_success')::int AS success,
            count(*) FILTER (WHERE event = 'deposit_error')::int AS error
       FROM events
      WHERE ts >= $1 AND ts < $2
        AND event IN ('deposit_initiated', 'deposit_success', 'deposit_error')
      GROUP BY 1
      ORDER BY initiated DESC`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ provider: string; initiated: number; success: number; error: number }>

  const t = totalRows[0]
  if (!t) throw new Error('depositSummary: aggregate query returned no rows')

  return {
    initiated: t.initiated,
    success: t.success,
    error: t.error,
    conversionRate: t.initiated > 0 ? t.success / t.initiated : 0,
    byProvider: byProviderRows.map(r => ({
      provider: r.provider, initiated: r.initiated, success: r.success, error: r.error,
    })),
  }
}
