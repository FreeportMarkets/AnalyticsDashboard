import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { hlFillsFetcher, type HlFill } from '@/lib/hl/volume'
import { privyRegistrySource, analyticsWalletSource } from '@/lib/volume/walletSource'
import {
  upsertTrackedWallets,
  activeTraderWallets,
  staleNonActiveWallets,
  readWatermarks,
  writeWalletBuckets,
  markScanned,
} from '@/lib/volume/store'
import { syncWalletsBatch, syncWalletVolume, type SyncDeps } from '@/lib/volume/syncVolume'
import { reconcile } from '@/lib/volume/reconcile'
import { readWatermark as readSyncState, advanceWatermark as writeSyncState } from '@/lib/sync/watermark'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

/** Refresh the Privy registry at most this often (it's ~44 pages, not free). */
const REGISTRY_REFRESH_MS = 60 * 60 * 1000
const REGISTRY_STATE_KEY = 'volume:registry-refresh'
/** How many never-scanned wallets to discover per run, time permitting. */
const DISCOVERY_BATCH = 400

/**
 * Two-phase, deadline-bounded volume sync. Designed so no single run sweeps
 * all ~4k wallets against HL's per-IP rate limit (which corrupts data by
 * dropping wallets):
 *
 *   Phase 1 — sync the ACTIVE traders (wallets with any volume row, a few
 *     dozen) incrementally. Cheap, and keeps today's number live every run.
 *   Phase 2 — with leftover time budget, DISCOVER a batch of never-scanned
 *     wallets so new signups get picked up. Each is marked scanned (even at
 *     zero fills) so it isn't rescanned; a discovered trader gets a volume
 *     row and joins Phase 1 next run.
 *   Registry — refreshed from Privy at most hourly, not every run.
 *
 * Everything is resumable: a wallet not reached this run keeps its watermark
 * (or stays unscanned) and is picked up next run. Nothing is ever lost.
 */
export async function GET(request: Request) {
  const secret = process.env.CRON_SECRET
  if (!secret || request.headers.get('authorization') !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 })
  }

  const startedAt = Date.now()
  const budgetMs = (maxDuration - 10) * 1000
  const timeLeft = () => budgetMs - (Date.now() - startedAt)

  // Registry refresh (hourly). Best-effort: a Privy hiccup must not stop sync.
  let discovered = 0
  try {
    const last = await readSyncState(sql, REGISTRY_STATE_KEY, new Date(0))
    if (Date.now() - last.getTime() > REGISTRY_REFRESH_MS) {
      const source =
        process.env.PRIVY_APP_ID && process.env.PRIVY_APP_SECRET
          ? privyRegistrySource()
          : analyticsWalletSource(sql)
      const wallets = await source.list()
      discovered = await upsertTrackedWallets(sql, wallets, source.name)
      await writeSyncState(sql, REGISTRY_STATE_KEY, new Date())
    }
  } catch (err) {
    console.warn('volume cron: registry refresh failed', err)
  }

  const watermarks = await readWatermarks(sql)
  const deps: SyncDeps = {
    readWatermark: async a => watermarks.get(a.toLowerCase()) ?? 0,
    fetchFills: (a, sinceMs) => hlFillsFetcher(a, sinceMs) as Promise<HlFill[]>,
    writeBuckets: (a, buckets, wm, additive) => writeWalletBuckets(sql, a, buckets, wm, additive),
  }

  // Phase 1 — active traders, always.
  const active = await activeTraderWallets(sql)
  const phase1 = await syncWalletsBatch(active, deps, {
    additive: true,
    concurrency: 6,
    deadlineMs: Math.max(0, budgetMs - 15_000), // reserve time for discovery + reconcile
  })

  // Phase 2 — re-check a rotating batch of non-active wallets with the
  // remaining time budget. Incremental (additive) so a re-checked wallet that
  // has traded since its last check gets those fills added, and a
  // never-checked wallet (watermark 0) gets its full history. This is what
  // catches a wallet that signs up, sits idle, then starts trading -- it stays
  // in the rotation and graduates to Phase 1 once it has a volume row.
  let rechecked = 0
  let discoveredTraders = 0
  if (timeLeft() > 8_000) {
    const candidates = await staleNonActiveWallets(sql, DISCOVERY_BATCH)
    let next = 0
    const nowMs = Date.now()
    async function recheck() {
      while (timeLeft() > 5_000) {
        const i = next++
        if (i >= candidates.length) return
        const addr = candidates[i]!
        const r = await syncWalletVolume(addr, deps, { additive: true })
        if (r.error) continue // leave as-is -> retried next rotation
        rechecked++
        if (r.attributedFills > 0) discoveredTraders++
        // No new fills -> bump the check time so the rotation advances but the
        // wallet is NOT dropped; it'll be re-checked again next cycle.
        if (r.daysTouched === 0) await markScanned(sql, addr, nowMs)
      }
    }
    await Promise.all([recheck(), recheck(), recheck()])
  }

  // Reconcile (best-effort). The <1% gate; logged for the trend + dashboard.
  let recon = null
  try {
    const trackedCount = active.length + rechecked
    recon = await reconcile(sql, { walletsTracked: trackedCount, note: 'cron', log: true })
  } catch (err) {
    console.warn('volume cron: reconcile failed', err)
  }

  return NextResponse.json({
    ok: true,
    durationMs: Date.now() - startedAt,
    registryDiscovered: discovered,
    activeTraders: active.length,
    activeProcessed: phase1.processed,
    activeErrors: phase1.results.filter(r => r.error).length,
    newFills: phase1.results.reduce((s, r) => s + r.attributedFills, 0),
    rechecked,
    discoveryNewTraders: discoveredTraders,
    reconciliation: recon
      ? {
          ratio: Number(recon.ratio.toFixed(4)),
          complete: recon.complete,
          bottomUpFeeUsd: Math.round(recon.bottomUpFeeUsd),
          topDownFeeUsd: Math.round(recon.topDownFeeUsd),
        }
      : null,
  })
}
