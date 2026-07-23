import type { neon } from '@neondatabase/serverless'
import type { DayBucket } from './aggregate'

type SqlTag = ReturnType<typeof neon<boolean, boolean>>

/**
 * All DB reads/writes for the volume tracking tables. Every address is
 * lowercased at this boundary -- HL, our trades table, and the Privy registry
 * all use different casing, and the whole system only joins correctly if
 * addresses are canonicalized in exactly one place. This is that place.
 */

const lc = (addr: string) => addr.toLowerCase()

/** Register wallets to scan. Idempotent; existing rows keep their first_seen_at. */
export async function upsertTrackedWallets(
  sql: SqlTag,
  addresses: string[],
  source: string
): Promise<number> {
  const uniq = [...new Set(addresses.map(lc))]
  if (uniq.length === 0) return 0
  // Chunk to keep parameter counts sane for the HTTP driver.
  const CHUNK = 500
  let written = 0
  for (let i = 0; i < uniq.length; i += CHUNK) {
    const chunk = uniq.slice(i, i + CHUNK)
    const values = chunk.map((_, j) => `($${j + 1}, $${chunk.length + 1})`).join(', ')
    await sql(
      `INSERT INTO tracked_wallets (evm_address, source)
       VALUES ${values}
       ON CONFLICT (evm_address) DO UPDATE SET source = EXCLUDED.source, updated_at = now()`,
      [...chunk, source]
    )
    written += chunk.length
  }
  return written
}

export async function listTrackedWallets(sql: SqlTag): Promise<string[]> {
  const rows = (await sql`SELECT evm_address FROM tracked_wallets`) as Array<{ evm_address: string }>
  return rows.map(r => r.evm_address)
}

/**
 * Wallets that have ever produced a volume row -- the "active traders". The
 * every-few-minutes cron syncs only these (a few dozen), keeping today's
 * number live without sweeping all 4k wallets against HL's rate limit.
 */
export async function activeTraderWallets(sql: SqlTag): Promise<string[]> {
  const rows = (await sql`SELECT DISTINCT evm_address FROM wallet_volume_daily`) as Array<{
    evm_address: string
  }>
  return rows.map(r => r.evm_address)
}

/**
 * Tracked wallets that have NEVER been scanned (no sync-state row). The
 * discovery phase works through these in batches so new signups get picked up
 * without rescanning the whole registry every run. Once scanned (even with
 * zero fills) a wallet gets a sync-state row via markScanned and drops out of
 * this set.
 */
export async function unscannedWallets(sql: SqlTag, limit: number): Promise<string[]> {
  const rows = (await sql(
    `SELECT t.evm_address
       FROM tracked_wallets t
       LEFT JOIN hl_fill_sync_state s ON s.evm_address = t.evm_address
      WHERE s.evm_address IS NULL
      LIMIT $1`,
    [limit]
  )) as Array<{ evm_address: string }>
  return rows.map(r => r.evm_address)
}

/**
 * Mark a wallet scanned by seeding its watermark, even when it had zero fills.
 * Without this an empty wallet keeps no state row and is rescanned from
 * scratch every discovery run. The seeded watermark means the next
 * incremental sync fetches only fills after this instant, so a wallet that
 * starts trading later is still caught.
 */
export async function markScanned(sql: SqlTag, address: string, watermarkMs: number): Promise<void> {
  await sql(
    `INSERT INTO hl_fill_sync_state (evm_address, last_fill_ms, updated_at)
     VALUES ($1, $2, now())
     ON CONFLICT (evm_address) DO UPDATE
       SET last_fill_ms = GREATEST(hl_fill_sync_state.last_fill_ms, EXCLUDED.last_fill_ms),
           updated_at   = now()`,
    [address.toLowerCase(), watermarkMs]
  )
}

export async function readWatermark(sql: SqlTag, address: string): Promise<number> {
  const rows = (await sql(
    `SELECT last_fill_ms FROM hl_fill_sync_state WHERE evm_address = $1`,
    [lc(address)]
  )) as Array<{ last_fill_ms: string | number }>
  return rows.length ? Number(rows[0]!.last_fill_ms) : 0
}

export async function readWatermarks(sql: SqlTag): Promise<Map<string, number>> {
  const rows = (await sql`SELECT evm_address, last_fill_ms FROM hl_fill_sync_state`) as Array<{
    evm_address: string
    last_fill_ms: string | number
  }>
  return new Map(rows.map(r => [r.evm_address, Number(r.last_fill_ms)]))
}

/**
 * Persist one wallet's aggregation: add each day's delta into
 * wallet_volume_daily and advance the watermark -- in that order, so a crash
 * between them re-adds the same fills next run and double-counts. To stay
 * safe under retries, the daily upsert is written as an ABSOLUTE set for a
 * full-backfill (sinceMs=0) and an ADDITIVE delta for incremental runs; the
 * caller signals which via `additive`. The watermark's GREATEST guard makes
 * the advance itself idempotent.
 */
export async function writeWalletBuckets(
  sql: SqlTag,
  address: string,
  buckets: DayBucket[],
  newWatermarkMs: number,
  additive: boolean
): Promise<void> {
  const addr = lc(address)
  for (const b of buckets) {
    if (additive) {
      await sql(
        `INSERT INTO wallet_volume_daily (evm_address, day, notional_usd, builder_fee_usd, fill_count)
         VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT (evm_address, day) DO UPDATE SET
           notional_usd    = wallet_volume_daily.notional_usd + EXCLUDED.notional_usd,
           builder_fee_usd = wallet_volume_daily.builder_fee_usd + EXCLUDED.builder_fee_usd,
           fill_count      = wallet_volume_daily.fill_count + EXCLUDED.fill_count,
           updated_at      = now()`,
        [addr, b.day, b.notionalUsd, b.builderFeeUsd, b.fillCount]
      )
    } else {
      await sql(
        `INSERT INTO wallet_volume_daily (evm_address, day, notional_usd, builder_fee_usd, fill_count)
         VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT (evm_address, day) DO UPDATE SET
           notional_usd    = EXCLUDED.notional_usd,
           builder_fee_usd = EXCLUDED.builder_fee_usd,
           fill_count      = EXCLUDED.fill_count,
           updated_at      = now()`,
        [addr, b.day, b.notionalUsd, b.builderFeeUsd, b.fillCount]
      )
    }
  }
  await sql(
    `INSERT INTO hl_fill_sync_state (evm_address, last_fill_ms, updated_at)
     VALUES ($1, $2, now())
     ON CONFLICT (evm_address) DO UPDATE
       SET last_fill_ms = GREATEST(hl_fill_sync_state.last_fill_ms, EXCLUDED.last_fill_ms),
           updated_at   = now()`,
    [addr, newWatermarkMs]
  )
}

/** Σ(builder_fee_usd) across all stored daily rows -- the bottom-up total. */
export async function bottomUpBuilderFee(sql: SqlTag): Promise<number> {
  const rows = (await sql`SELECT coalesce(sum(builder_fee_usd), 0)::float8 AS fee FROM wallet_volume_daily`) as Array<{
    fee: number
  }>
  return rows[0]?.fee ?? 0
}

export async function logReconciliation(
  sql: SqlTag,
  r: { bottomUpFeeUsd: number; topDownFeeUsd: number; walletsTracked: number; note?: string }
): Promise<void> {
  const ratio = r.topDownFeeUsd > 0 ? r.bottomUpFeeUsd / r.topDownFeeUsd : 0
  await sql(
    `INSERT INTO volume_reconciliation (bottom_up_fee_usd, top_down_fee_usd, ratio, wallets_tracked, note)
     VALUES ($1, $2, $3, $4, $5)`,
    [r.bottomUpFeeUsd, r.topDownFeeUsd, ratio, r.walletsTracked, r.note ?? null]
  )
}
