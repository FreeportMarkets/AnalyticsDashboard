import type { MapResult } from './types'
import { isValidCalendarDate } from '../time'

export interface TradeRow {
  wallet_address: string
  timestamp: string
  ts: Date
  trade_date: string | null
  id: string | null
  type: string | null
  amount_usd: number | null
  status: string | null
  source: string | null
  client: string
  from_token: string | null
  from_mint: string | null
  to_token: string | null
  to_mint: string | null
  amount_from_token: number | null
  amount_to_token: number | null
  tx_signature: string | null
  request_id: string | null
  tweet_handle: string | null
  tweet_ticker: string | null
  tweet_timestamp: string | null
  asset: string | null
  display_symbol: string | null
  side: string | null
  size: number | null
  price: number | null
  leverage: number | null
  order_type: string | null
  is_close: boolean | null
  is_hip3: boolean | null
  category: string | null
  trace_id: string | null
  raw: Record<string, unknown>
}

function optString(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null
}
function optNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}
function optBool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null
}

/**
 * Map a raw item from `freeport-trades-history`.
 *
 * Two writers produce this table with different field sets:
 *   perps -- freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts:167-192
 *   swaps -- Swap_Server/src/services/dynamodb.ts:67-107
 *
 * The full raw item is retained in `raw` because `amount_usd` on perps rows is
 * MARGIN rather than notional (trade-logger.ts:39-46), and volume math may need
 * fields this schema does not name explicitly.
 */
export function mapTrade(item: unknown): MapResult<TradeRow> {
  if (typeof item !== 'object' || item === null) {
    return { ok: false, reason: 'item is not an object' }
  }
  const o = item as Record<string, unknown>

  for (const f of ['wallet_address', 'timestamp'] as const) {
    if (typeof o[f] !== 'string' || (o[f] as string).length === 0) {
      return { ok: false, reason: `missing or non-string required field: ${f}` }
    }
  }

  const ts = new Date(o.timestamp as string)
  if (Number.isNaN(ts.getTime())) {
    return { ok: false, reason: `unparseable timestamp: ${String(o.timestamp)}` }
  }

  // `trade_date` is optional -- an absent value maps to `null` and succeeds.
  // But `trades.trade_date` is a Postgres `date` column (same class of
  // exposure `mapEvent` was hardened against for `events.date`), so a
  // PRESENT-but-malformed value must be caught here rather than passing
  // mapping and blowing up the insert -- or worse, wedging the whole insert
  // batch (see insertTrades.ts's bisection).
  if (typeof o.trade_date === 'string' && o.trade_date.length > 0) {
    if (!isValidCalendarDate(o.trade_date)) {
      return { ok: false, reason: `invalid trade_date: ${String(o.trade_date)}` }
    }
  }

  return {
    ok: true,
    value: {
      wallet_address: o.wallet_address as string,
      timestamp: o.timestamp as string,
      ts,
      trade_date: optString(o.trade_date),
      id: optString(o.id),
      type: optString(o.type),
      amount_usd: optNumber(o.amount_usd),
      status: optString(o.status),
      source: optString(o.source),
      // Matches the writer's own default at trade-logger.ts:189.
      client: optString(o.client) ?? 'unknown',
      from_token: optString(o.from_token),
      from_mint: optString(o.from_mint),
      to_token: optString(o.to_token),
      to_mint: optString(o.to_mint),
      amount_from_token: optNumber(o.amount_from_token),
      amount_to_token: optNumber(o.amount_to_token),
      tx_signature: optString(o.tx_signature),
      request_id: optString(o.request_id),
      tweet_handle: optString(o.tweet_handle),
      tweet_ticker: optString(o.tweet_ticker),
      tweet_timestamp: optString(o.tweet_timestamp),
      asset: optString(o.asset),
      display_symbol: optString(o.display_symbol),
      side: optString(o.side),
      size: optNumber(o.size),
      price: optNumber(o.price),
      leverage: optNumber(o.leverage),
      order_type: optString(o.order_type),
      is_close: optBool(o.is_close),
      is_hip3: optBool(o.is_hip3),
      category: optString(o.category),
      trace_id: optString(o.trace_id),
      raw: o,
    },
  }
}
