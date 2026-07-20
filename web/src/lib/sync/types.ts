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
}

export interface SyncResult {
  scanned: number
  inserted: number
  quarantined: number
  watermark: string
}
