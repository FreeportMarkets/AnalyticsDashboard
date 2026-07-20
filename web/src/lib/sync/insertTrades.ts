import type { neon } from '@neondatabase/serverless'
import type { TradeRow } from './mapTrade'
import { isDataError, pgErrorMessage } from './pgError'

type SqlTag = ReturnType<typeof neon>

const COLUMNS = [
  'wallet_address', 'timestamp', 'ts', 'trade_date', 'id', 'type', 'amount_usd',
  'status', 'source', 'client', 'from_token', 'from_mint', 'to_token', 'to_mint',
  'amount_from_token', 'amount_to_token', 'tx_signature', 'request_id',
  'tweet_handle', 'tweet_ticker', 'tweet_timestamp', 'asset', 'display_symbol',
  'side', 'size', 'price', 'leverage', 'order_type', 'is_close', 'is_hip3',
  'category', 'trace_id', 'raw',
] as const

async function runInsert(sql: SqlTag, chunk: TradeRow[]) {
  const values = chunk
    .map((_, r) => `(${COLUMNS.map((_, c) => `$${r * COLUMNS.length + c + 1}`).join(',')})`)
    .join(',')
  const params = chunk.flatMap(r => [
    r.wallet_address, r.timestamp, r.ts.toISOString(), r.trade_date, r.id, r.type,
    r.amount_usd, r.status, r.source, r.client, r.from_token, r.from_mint,
    r.to_token, r.to_mint, r.amount_from_token, r.amount_to_token, r.tx_signature,
    r.request_id, r.tweet_handle, r.tweet_ticker, r.tweet_timestamp, r.asset,
    r.display_symbol, r.side, r.size, r.price, r.leverage, r.order_type,
    r.is_close, r.is_hip3, r.category, r.trace_id, JSON.stringify(r.raw),
  ])
  const updates = COLUMNS
    .filter(c => c !== 'wallet_address' && c !== 'timestamp')
    .map(c => `${c} = EXCLUDED.${c}`)
    .join(', ')
  // No cast here -- see insertEvents.ts's `runInsert` for why leaving the
  // driver's own `fullResults: true` overload type flow through (instead of
  // casting) is load-bearing: it keeps `result.rowCount` a compile error, not
  // a silent `undefined`, if that option is ever dropped.
  return sql(
    `INSERT INTO trades (${COLUMNS.join(',')}) VALUES ${values}
     ON CONFLICT (wallet_address, timestamp) DO UPDATE SET ${updates}, synced_at = now()`,
    params,
    { fullResults: true }
  )
}

/**
 * Insert one chunk, bisecting on failure so one poisoned row cannot sink its
 * neighbors. See insertEvents.ts's `insertChunk` for the full rationale --
 * same multi-row-statement-is-atomic hazard, same fix.
 */
async function insertChunk(
  sql: SqlTag,
  chunk: TradeRow[],
  onRowFailure?: (row: TradeRow, reason: string) => Promise<void>
): Promise<number> {
  if (chunk.length === 0) return 0
  try {
    const result = await runInsert(sql, chunk)
    return result.rowCount
  } catch (err) {
    if (!isDataError(err)) throw err

    if (chunk.length === 1) {
      const reason = pgErrorMessage(err)
      if (onRowFailure) await onRowFailure(chunk[0]!, reason)
      return 0
    }

    const mid = Math.ceil(chunk.length / 2)
    const left = await insertChunk(sql, chunk.slice(0, mid), onRowFailure)
    const right = await insertChunk(sql, chunk.slice(mid), onRowFailure)
    return left + right
  }
}

/**
 * Upsert trades.
 *
 * DO UPDATE, not DO NOTHING: both writers use PutCommand, which overwrites in
 * place on a repeated (wallet_address, timestamp) key. DO NOTHING would pin the
 * mirror to a superseded version of the row (spec risk #7).
 *
 * `onRowFailure`, if given, is invoked once per row a chunk insert rejects
 * for a data reason, after bisection has isolated it. The returned count is
 * always the number of rows ACTUALLY inserted/updated.
 */
export async function insertTrades(
  sql: SqlTag,
  rows: TradeRow[],
  onRowFailure?: (row: TradeRow, reason: string) => Promise<void>
): Promise<number> {
  if (rows.length === 0) return 0
  let total = 0
  for (let i = 0; i < rows.length; i += 500) {
    total += await insertChunk(sql, rows.slice(i, i + 500), onRowFailure)
  }
  return total
}
