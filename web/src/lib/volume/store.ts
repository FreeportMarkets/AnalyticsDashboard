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
 * A rotating batch of NON-ACTIVE wallets (no volume rows yet) to re-check,
 * least-recently-checked first (never-checked wallets first of all).
 *
 * This is the fix for the "dormant wallet goes dark" bug: marking a
 * zero-fill wallet scanned must NOT exclude it forever, or a user who signs
 * up, sits idle a day, then trades would never be recorded. Instead every
 * non-active wallet stays in this rotation and is re-checked on a cycle
 * (batch size × run interval), so a wallet that starts trading is picked up
 * within one cycle and then graduates to the active set (Phase 1) once it has
 * a volume row. Active traders are excluded here because Phase 1 already
 * syncs them every run.
 */
export async function staleNonActiveWallets(sql: SqlTag, limit: number): Promise<string[]> {
  const rows = (await sql(
    `SELECT t.evm_address
       FROM tracked_wallets t
       LEFT JOIN hl_fill_sync_state s ON s.evm_address = t.evm_address
      WHERE NOT EXISTS (
        SELECT 1 FROM wallet_volume_daily v WHERE v.evm_address = t.evm_address
      )
      ORDER BY s.updated_at ASC NULLS FIRST
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
 * Persist one wallet's aggregation ATOMICALLY: all day buckets AND the
 * watermark advance in a single Postgres transaction.
 *
 * The atomicity is the whole point. An earlier version wrote the buckets, then
 * the watermark, in separate statements. A crash between them left the fills
 * counted but the watermark unmoved, so the next run re-fetched the same fills
 * and (in additive mode) ADDED them again -- a permanent overstatement that
 * reconciliation could only detect, not repair. Bundling them means either
 * both land or neither does; a retry then re-fetches from the OLD watermark
 * and re-applies the SAME delta onto a table that never received the first
 * one. No double count.
 *
 * `additive` (incremental: only fills after the watermark were fetched, so
 * they're deltas) vs absolute (backfill: the full day was recomputed, so
 * overwrite). The watermark's GREATEST guard keeps the advance monotonic.
 */
export async function writeWalletBuckets(
  sql: SqlTag,
  address: string,
  buckets: DayBucket[],
  newWatermarkMs: number,
  additive: boolean
): Promise<void> {
  const addr = lc(address)
  const bucketSet = additive
    ? `notional_usd    = wallet_volume_daily.notional_usd + EXCLUDED.notional_usd,
       builder_fee_usd = wallet_volume_daily.builder_fee_usd + EXCLUDED.builder_fee_usd,
       fill_count      = wallet_volume_daily.fill_count + EXCLUDED.fill_count,
       updated_at      = now()`
    : `notional_usd    = EXCLUDED.notional_usd,
       builder_fee_usd = EXCLUDED.builder_fee_usd,
       fill_count      = EXCLUDED.fill_count,
       updated_at      = now()`

  const queries = buckets.map(b =>
    sql(
      `INSERT INTO wallet_volume_daily (evm_address, day, notional_usd, builder_fee_usd, fill_count)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (evm_address, day) DO UPDATE SET ${bucketSet}`,
      [addr, b.day, b.notionalUsd, b.builderFeeUsd, b.fillCount]
    )
  )
  queries.push(
    sql(
      `INSERT INTO hl_fill_sync_state (evm_address, last_fill_ms, updated_at)
       VALUES ($1, $2, now())
       ON CONFLICT (evm_address) DO UPDATE
         SET last_fill_ms = GREATEST(hl_fill_sync_state.last_fill_ms, EXCLUDED.last_fill_ms),
             updated_at   = now()`,
      [addr, newWatermarkMs]
    )
  )

  // One HTTP round-trip, one transaction: all buckets + the watermark commit
  // together or not at all.
  await sql.transaction(queries)
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
