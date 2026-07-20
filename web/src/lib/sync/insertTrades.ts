import type { neon } from '@neondatabase/serverless'
import type { TradeRow } from './mapTrade'

type SqlTag = ReturnType<typeof neon>

const COLUMNS = [
  'wallet_address', 'timestamp', 'ts', 'trade_date', 'id', 'type', 'amount_usd',
  'status', 'source', 'client', 'from_token', 'from_mint', 'to_token', 'to_mint',
  'amount_from_token', 'amount_to_token', 'tx_signature', 'request_id',
  'tweet_handle', 'tweet_ticker', 'tweet_timestamp', 'asset', 'display_symbol',
  'side', 'size', 'price', 'leverage', 'order_type', 'is_close', 'is_hip3',
  'category', 'trace_id', 'raw',
] as const

/**
 * Upsert trades.
 *
 * DO UPDATE, not DO NOTHING: both writers use PutCommand, which overwrites in
 * place on a repeated (wallet_address, timestamp) key. DO NOTHING would pin the
 * mirror to a superseded version of the row (spec risk #7).
 */
export async function insertTrades(sql: SqlTag, rows: TradeRow[]): Promise<number> {
  if (rows.length === 0) return 0
  let total = 0
  for (let i = 0; i < rows.length; i += 500) {
    const chunk = rows.slice(i, i + 500)
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
    const result = (await sql(
      `INSERT INTO trades (${COLUMNS.join(',')}) VALUES ${values}
       ON CONFLICT (wallet_address, timestamp) DO UPDATE SET ${updates}, synced_at = now()`,
      params,
      { fullResults: true }
    )) as unknown as { rowCount: number }
    total += result.rowCount
  }
  return total
}
