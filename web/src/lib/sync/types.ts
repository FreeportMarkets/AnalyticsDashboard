export type MapResult<T> =
  | { ok: true; value: T }
  | { ok: false; reason: string }

export interface EventRow {
  date: string            // YYYY-MM-DD, UTC-derived. Storage only -- never a GROUP BY key.
  sk: string
  ts: Date
  event: string
  screen: string | null
  component: string | null
  wallet_address: string | null
  session_id: string | null
  platform: string | null
  app_version: string | null
  metadata: Record<string, unknown> | null
  /**
   * The original, unmapped item as fetched from DynamoDB. Not written to any
   * column -- `events` has no `raw` column -- it exists so the insert-time
   * quarantine path (bisection failures in insertEvents.ts) can store the
   * true raw item instead of the already-normalized row, matching the
   * project's "quarantine the raw item" contract that map-time failures
   * already satisfy. TradeRow carries the equivalent field for the same
   * reason (see mapTrade.ts).
   */
  raw: Record<string, unknown>
}

export interface SyncResult {
  scanned: number
  inserted: number
  quarantined: number
  watermark: string
}
