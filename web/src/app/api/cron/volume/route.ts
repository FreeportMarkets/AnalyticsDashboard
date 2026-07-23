import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { hlFillsFetcher, type HlFill } from '@/lib/hl/volume'
import { defaultWalletSource } from '@/lib/volume/walletSource'
import {
  upsertTrackedWallets,
  listTrackedWallets,
  readWatermarks,
  writeWalletBuckets,
} from '@/lib/volume/store'
import { syncWalletsBatch, type SyncDeps } from '@/lib/volume/syncVolume'
import { reconcile } from '@/lib/volume/reconcile'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

/**
 * Incremental volume sync.
 *
 * 1. Refresh the tracked-wallet set from the best available source (Privy
 *    registry when creds exist). New signups get scanned automatically.
 * 2. For each wallet, fetch HL fills after its watermark, aggregate, upsert
 *    daily volume, advance the watermark. Bounded by a wall-clock deadline so
 *    we never hit the platform's function timeout -- unreached wallets are
 *    picked up next run (their watermark hasn't moved, so nothing is lost).
 * 3. Reconcile Σ(builder_fee) against the collector's fees and log the ratio.
 *
 * Durability note: HL ages fills out of userFillsByTime, so this must run
 * often enough to capture fills before they expire. That, not dashboard
 * freshness, is why the cron is mandatory.
 */
export async function GET(request: Request) {
  const secret = process.env.CRON_SECRET
  const auth = request.headers.get('authorization')
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 })
  }

  const startedAt = Date.now()

  // 1. Refresh enumeration.
  const source = defaultWalletSource(sql)
  let discovered = 0
  try {
    const wallets = await source.list()
    discovered = await upsertTrackedWallets(sql, wallets, source.name)
  } catch (err) {
    // A registry hiccup shouldn't stop us syncing the wallets we already track.
    // eslint-disable-next-line no-console
    console.warn('volume cron: wallet source failed, using existing tracked set', err)
  }

  const tracked = await listTrackedWallets(sql)
  const watermarks = await readWatermarks(sql)

  // 2. Incremental per-wallet sync. Leave ~8s headroom under maxDuration for
  //    the reconcile + response.
  const deadlineMs = (maxDuration - 12) * 1000
  const deps: SyncDeps = {
    readWatermark: async a => watermarks.get(a.toLowerCase()) ?? 0,
    fetchFills: (a, sinceMs) => hlFillsFetcher(a, sinceMs) as Promise<HlFill[]>,
    writeBuckets: (a, buckets, wm, additive) => writeWalletBuckets(sql, a, buckets, wm, additive),
  }
  const { results, processed, skippedForTime } = await syncWalletsBatch(tracked, deps, {
    additive: true,
    concurrency: 6,
    deadlineMs,
  })

  const errors = results.filter(r => r.error).length
  const newFills = results.reduce((s, r) => s + r.attributedFills, 0)
  const newNotional = results.reduce((s, r) => s + r.notionalUsd, 0)

  // 3. Reconcile (best-effort — never fail the sync on the fee lookup).
  let recon = null
  try {
    recon = await reconcile(sql, { walletsTracked: tracked.length, note: 'cron', log: true })
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('volume cron: reconcile failed', err)
  }

  return NextResponse.json({
    ok: true,
    durationMs: Date.now() - startedAt,
    discovered,
    tracked: tracked.length,
    processed,
    skippedForTime,
    errors,
    newFills,
    newNotionalUsd: Math.round(newNotional),
    reconciliation: recon
      ? {
          ratio: Number(recon.ratio.toFixed(4)),
          bottomUpFeeUsd: Math.round(recon.bottomUpFeeUsd),
          topDownFeeUsd: Math.round(recon.topDownFeeUsd),
          complete: recon.complete,
        }
      : null,
  })
}
