/**
 * One-time backfill of HL builder-fee-authoritative perps volume.
 *
 * Enumerates every Freeport wallet (Privy registry), scans all HL fills back
 * to builder-code launch (2026-05-07 — nothing earlier carries a builder fee),
 * and writes per-wallet-per-day rows into `wallet_volume_daily`, seeding each
 * watermark to the wallet's max fill time so the incremental cron takes over
 * cleanly.
 *
 * Resumable + idempotent: a wallet whose watermark is already set is skipped
 * on re-run (unless --force), and each wallet writes absolute daily values
 * (additive=false), so re-running a wallet overwrites rather than doubles.
 * Safe to Ctrl-C and restart.
 *
 * HL's public endpoint rate-limits hard against a 4k-wallet sweep, so this
 * paces itself (low concurrency + backoff inside hlFillsFetcher) and prints
 * progress. Expect it to take a while; that's fine — it runs once.
 *
 * Usage:
 *   npx tsx scripts/backfill-volume.ts [--concurrency N] [--force] [--limit N]
 *
 * Requires DATABASE_URL, PRIVY_APP_ID, PRIVY_APP_SECRET.
 */
import { neon } from '@neondatabase/serverless'
import { hlFillsFetcher, type HlFill } from '../src/lib/hl/volume'
import { privyRegistrySource, analyticsWalletSource } from '../src/lib/volume/walletSource'
import {
  upsertTrackedWallets,
  listTrackedWallets,
  readWatermarks,
  writeWalletBuckets,
} from '../src/lib/volume/store'
import { syncWalletVolume, type SyncDeps } from '../src/lib/volume/syncVolume'
import { reconcile } from '../src/lib/volume/reconcile'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')
const sql = neon<boolean, boolean>(url)

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 ? process.argv[i + 1] : undefined
}
const flag = (name: string) => process.argv.includes(`--${name}`)

async function main() {
  const concurrency = Number(arg('concurrency') ?? 5)
  const force = flag('force')
  const limit = arg('limit') ? Number(arg('limit')) : undefined

  // 1. Enumerate + register wallets.
  const source =
    process.env.PRIVY_APP_ID && process.env.PRIVY_APP_SECRET
      ? privyRegistrySource()
      : analyticsWalletSource(sql)
  console.log(`enumerating wallets via ${source.name}...`)
  const wallets = await source.list()
  console.log(`  ${wallets.length} wallets`)
  await upsertTrackedWallets(sql, wallets, source.name)

  let tracked = await listTrackedWallets(sql)
  if (limit) tracked = tracked.slice(0, limit)
  const watermarks = await readWatermarks(sql)

  const deps: SyncDeps = {
    readWatermark: async a => watermarks.get(a.toLowerCase()) ?? 0,
    fetchFills: (a, sinceMs) => hlFillsFetcher(a, sinceMs) as Promise<HlFill[]>,
    writeBuckets: (a, buckets, wm, additive) => writeWalletBuckets(sql, a, buckets, wm, additive),
  }

  // 2. Scan. Full-backfill semantics (additive=false, from 0). Skip wallets
  //    already watermarked unless --force.
  const todo = force ? tracked : tracked.filter(a => (watermarks.get(a.toLowerCase()) ?? 0) === 0)
  console.log(`scanning ${todo.length} wallets (concurrency ${concurrency}, ${force ? 'force' : 'resume'})...`)

  let done = 0
  let traded = 0
  let errored = 0
  let totalNotional = 0
  let next = 0
  const failedWallets: string[] = []
  async function worker() {
    for (;;) {
      const i = next++
      if (i >= todo.length) return
      const w = todo[i]!
      const r = await syncWalletVolume(w, deps, { additive: false })
      done++
      if (r.error) {
        errored++
        failedWallets.push(w)
      } else if (r.notionalUsd > 0) {
        traded++
        totalNotional += r.notionalUsd
      }
      if (done % 200 === 0 || done === todo.length) {
        console.log(
          `  ${done}/${todo.length}  traded=${traded}  errored=${errored}  Σnotional=$${(totalNotional / 1e6).toFixed(1)}M`
        )
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, todo.length) }, worker))

  // Errored wallets never advanced their watermark, so re-running resumes
  // them -- but a backfill that "completed" with errors is NOT done. Surface
  // it loudly rather than reconciling against a partial dataset.
  if (errored > 0) {
    console.log(`\n⚠️  ${errored} wallets FAILED (rate limit / network) and were NOT counted.`)
    console.log(`    Re-run to resume them: npx tsx scripts/backfill-volume.ts --concurrency ${concurrency}`)
  }

  // 3. Reconcile against the collector's fees.
  const recon = await reconcile(sql, { walletsTracked: tracked.length, note: 'backfill', log: true })
  console.log('\n=== reconciliation ===')
  console.log(`bottom-up builder fee : $${recon.bottomUpFeeUsd.toFixed(2)}`)
  console.log(`top-down builder fee  : $${recon.topDownFeeUsd.toFixed(2)}`)
  console.log(`ratio                 : ${(recon.ratio * 100).toFixed(1)}%`)
  console.log(`gap                   : $${recon.gapUsd.toFixed(2)}`)
  if (recon.complete) {
    console.log('COMPLETE (<1% gap) ✓')
  } else if (errored > 0) {
    console.log(`INCOMPLETE — ${errored} wallets failed this run; RE-RUN to resume before trusting the total.`)
    process.exitCode = 2
  } else {
    console.log('INCOMPLETE — all wallets scanned but fees still short; enumeration is missing wallets.')
    process.exitCode = 2
  }
}

main().then(() => process.exit(0)).catch(e => { console.error(e); process.exit(1) })
