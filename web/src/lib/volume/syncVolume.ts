import { aggregateWalletFills } from './aggregate'
import type { HlFill } from '@/lib/hl/volume'

/**
 * Per-wallet volume sync orchestration, expressed against injected primitives
 * so it is unit-testable without a DB or the network. The cron wires the real
 * store + HL fetcher; tests wire fakes.
 *
 *   watermark -> fetch fills after it -> aggregate -> write buckets + advance
 *
 * `additive` is the single most important flag:
 *   - backfill  (additive=false): the wallet has no prior rows, so each day's
 *     bucket is written as an ABSOLUTE value. Re-running a backfill for a
 *     wallet overwrites, never doubles.
 *   - incremental (additive=true): we only fetched fills AFTER the watermark,
 *     so those buckets are DELTAS to add onto whatever the day already holds.
 * Getting this wrong is how volume silently doubles, so it is explicit at
 * every call site rather than inferred.
 */

export interface SyncDeps {
  readWatermark: (address: string) => Promise<number>
  fetchFills: (address: string, sinceMs: number) => Promise<HlFill[]>
  writeBuckets: (
    address: string,
    buckets: ReturnType<typeof aggregateWalletFills>['buckets'],
    newWatermarkMs: number,
    additive: boolean
  ) => Promise<void>
}

export interface WalletSyncResult {
  address: string
  attributedFills: number
  notionalUsd: number
  builderFeeUsd: number
  daysTouched: number
  watermarkMs: number
  error?: string
}

export async function syncWalletVolume(
  address: string,
  deps: SyncDeps,
  opts: { additive: boolean }
): Promise<WalletSyncResult> {
  const base = {
    address,
    attributedFills: 0,
    notionalUsd: 0,
    builderFeeUsd: 0,
    daysTouched: 0,
    watermarkMs: 0,
  }
  try {
    const sinceMs = opts.additive ? await deps.readWatermark(address) : 0
    const fills = await deps.fetchFills(address, sinceMs)
    const agg = aggregateWalletFills(fills, sinceMs)

    // Nothing new and no watermark movement -> skip the write entirely.
    if (agg.buckets.length === 0 && agg.maxFillMs <= sinceMs) {
      return { ...base, watermarkMs: sinceMs }
    }

    await deps.writeBuckets(address, agg.buckets, agg.maxFillMs, opts.additive)

    return {
      address,
      attributedFills: agg.attributedFills,
      notionalUsd: agg.buckets.reduce((s, b) => s + b.notionalUsd, 0),
      builderFeeUsd: agg.buckets.reduce((s, b) => s + b.builderFeeUsd, 0),
      daysTouched: agg.buckets.length,
      watermarkMs: agg.maxFillMs,
    }
  } catch (err) {
    // One wallet's failure must never abort the batch -- record and move on.
    return { ...base, error: err instanceof Error ? err.message : String(err) }
  }
}

/**
 * Run `syncWalletVolume` across many wallets with a bounded number in flight.
 * A time budget (`deadlineMs`) lets the cron stop cleanly before its platform
 * timeout: wallets not reached this run are simply picked up next run (their
 * watermark hasn't moved), so partial progress is always safe.
 */
export async function syncWalletsBatch(
  addresses: string[],
  deps: SyncDeps,
  opts: { additive: boolean; concurrency?: number; deadlineMs?: number; now?: () => number }
): Promise<{ results: WalletSyncResult[]; processed: number; skippedForTime: number }> {
  const concurrency = opts.concurrency ?? 8
  const now = opts.now ?? (() => Date.now())
  const deadline = opts.deadlineMs ?? Infinity
  const start = now()

  const results: WalletSyncResult[] = []
  let next = 0
  let skippedForTime = 0

  async function worker() {
    for (;;) {
      if (now() - start >= deadline) {
        // Count what we're leaving for the next run, once.
        return
      }
      const i = next++
      if (i >= addresses.length) return
      results.push(await syncWalletVolume(addresses[i]!, deps, opts))
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, addresses.length) }, worker))
  const processed = results.length
  skippedForTime = addresses.length - processed
  return { results, processed, skippedForTime }
}
