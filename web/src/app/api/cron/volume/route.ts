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

/**
 * Hard wall-clock deadline for ALL work in this function, well under the 60s
 * platform timeout. Every HL fetch is passed this deadline so it aborts
 * rather than hanging in rate-limit backoff -- the fix for the
 * FUNCTION_INVOCATION_TIMEOUT that froze the data. Nothing here may run past
 * it; partial progress is safe because watermarks only advance on committed
 * writes, so the next run resumes cleanly.
 */
const HARD_BUDGET_MS = 45_000

const REGISTRY_REFRESH_MS = 60 * 60 * 1000
const REGISTRY_STATE_KEY = 'volume:registry-refresh'
/** Small so discovery can never dominate the budget; the rotation still covers everyone over time. */
const DISCOVERY_BATCH = 120

/**
 * Every-run volume sync, ordered by priority so the critical path always
 * finishes within budget:
 *
 *   1. Active traders (Phase 1) -- keeps today's number live. Runs first.
 *   2. Reconcile -- the health/completeness signal.
 *   3. Registry refresh (hourly) + discovery rotation -- best-effort, only
 *      with leftover time. New signups are picked up here.
 *
 * Every HL call carries the hard deadline, so no single stuck fetch can blow
 * the function timeout.
 */
export async function GET(request: Request) {
  const secret = process.env.CRON_SECRET
  if (!secret || request.headers.get('authorization') !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 })
  }

  const startedAt = Date.now()
  const deadlineMs = startedAt + HARD_BUDGET_MS
  const timeLeft = () => deadlineMs - Date.now()

  const watermarks = await readWatermarks(sql)
  const deps: SyncDeps = {
    readWatermark: async a => watermarks.get(a.toLowerCase()) ?? 0,
    fetchFills: (a, sinceMs) => hlFillsFetcher(a, sinceMs, undefined, deadlineMs) as Promise<HlFill[]>,
    writeBuckets: (a, buckets, wm, additive) => writeWalletBuckets(sql, a, buckets, wm, additive),
  }

  // 1. Active traders first -- the live number. Reserve ~12s for reconcile.
  const active = await activeTraderWallets(sql)
  const phase1 = await syncWalletsBatch(active, deps, {
    additive: true,
    concurrency: 6,
    deadlineMs: HARD_BUDGET_MS - 12_000,
  })

  // 2. Reconcile (best-effort, deadline-bounded HL call).
  let recon = null
  try {
    recon = await reconcile(sql, {
      walletsTracked: active.length,
      note: 'cron',
      log: true,
      deadlineMs: startedAt + HARD_BUDGET_MS - 3_000,
    })
  } catch (err) {
    console.warn('volume cron: reconcile failed', err)
  }

  // 3. Registry refresh (hourly) + discovery -- only with real time left.
  let registryDiscovered = 0
  let rechecked = 0
  let discoveredTraders = 0
  if (timeLeft() > 10_000) {
    try {
      const last = await readSyncState(sql, REGISTRY_STATE_KEY, new Date(0))
      if (Date.now() - last.getTime() > REGISTRY_REFRESH_MS) {
        const source =
          process.env.PRIVY_APP_ID && process.env.PRIVY_APP_SECRET
            ? privyRegistrySource()
            : analyticsWalletSource(sql)
        const wallets = await source.list()
        registryDiscovered = await upsertTrackedWallets(sql, wallets, source.name)
        await writeSyncState(sql, REGISTRY_STATE_KEY, new Date())
      }
    } catch (err) {
      console.warn('volume cron: registry refresh failed', err)
    }

    if (timeLeft() > 6_000) {
      const candidates = await staleNonActiveWallets(sql, DISCOVERY_BATCH)
      const nowMs = Date.now()
      let next = 0
      const recheck = async () => {
        while (timeLeft() > 4_000) {
          const i = next++
          if (i >= candidates.length) return
          const addr = candidates[i]!
          const r = await syncWalletVolume(addr, deps, { additive: true })
          if (r.error) continue
          rechecked++
          if (r.attributedFills > 0) discoveredTraders++
          if (r.daysTouched === 0) await markScanned(sql, addr, nowMs)
        }
      }
      await Promise.all([recheck(), recheck(), recheck()])
    }
  }

  return NextResponse.json({
    ok: true,
    durationMs: Date.now() - startedAt,
    activeTraders: active.length,
    activeProcessed: phase1.processed,
    activeErrors: phase1.results.filter(r => r.error).length,
    newFills: phase1.results.reduce((s, r) => s + r.attributedFills, 0),
    registryDiscovered,
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
