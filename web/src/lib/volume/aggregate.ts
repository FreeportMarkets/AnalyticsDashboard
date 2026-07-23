import { NY_TZ } from '@/lib/time'
import type { HlFill } from '@/lib/hl/volume'

/**
 * Pure aggregation core: turn a wallet's raw HL fills into per-NY-day volume
 * buckets and the new watermark. No I/O, no DB, no network -- so it is fully
 * unit-testable, and every correctness rule lives here rather than smeared
 * across the sync orchestration.
 *
 * The authoritative volume rule (see docs/volume-tracking.md):
 *   - Only perp fills (open/close, long/short); spot + settlement excluded.
 *   - Only fills with builderFee > 0 -- HL's own stamp that the order went
 *     through OUR builder code. This is the attribution.
 *   - Deduped by `tid`.
 *   - notional = |sz| * px, read directly from the fill (rate-independent).
 *   - Bucketed by New York calendar day, matching every other metric.
 */

export interface DayBucket {
  day: string // YYYY-MM-DD, NY calendar
  notionalUsd: number
  builderFeeUsd: number
  fillCount: number
}

export interface AggregateResult {
  /** Per-day buckets, ascending by day. Only days with ≥1 attributed fill. */
  buckets: DayBucket[]
  /** Max fill `time` (epoch ms) seen across ALL fills considered -- the new
   *  watermark. This advances even when no fill was builder-attributed, so a
   *  wallet that trades only outside Freeport still moves its watermark
   *  forward and isn't rescanned from scratch every run. */
  maxFillMs: number
  /** Count of builder-attributed perp fills that contributed to buckets. */
  attributedFills: number
}

/**
 * Non-perp fill directions to EXCLUDE. This is a denylist, not a perp
 * allowlist, on purpose: `builderFee > 0` is HL's own stamp that the order
 * went through our builder code, and Freeport is perps-only, so a positive
 * builder fee already means a Freeport perp trade. Gating on an allowlist of
 * {Open,Close}×{Long,Short} would silently DROP advanced-order fills that use
 * other perp directions -- position flips (`Long > Short`, `Short > Long`),
 * and any future dir HL introduces -- which is exactly the "handle all order
 * types" failure mode. We exclude only the known spot / settlement dirs,
 * which never carry our fee anyway (verified empirically), as belt-and-braces
 * against HL ever attaching a builder fee to a spot conversion.
 *
 * The reconciliation gate is the ultimate backstop: if this filter ever drops
 * a fee-bearing fill, Σ(builderFee) bottom-up won't match the collector's
 * total and the run is flagged incomplete.
 */
const NON_PERP_DIRS = new Set(['Buy', 'Sell', 'Spot Dust Conversion', 'Settlement'])

/**
 * True for a Freeport-attributed perp fill: it carried our builder fee and is
 * not a spot/settlement fill. Captures every perp direction, including
 * position flips and TWAP/trigger fills (which use the normal perp dirs).
 */
export function isFreeportPerpFill(fill: HlFill): boolean {
  if (Number(fill.builderFee ?? 0) <= 0) return false
  return !NON_PERP_DIRS.has(fill.dir ?? '')
}

/** One fill's notional. |sz| * px, always positive. */
function notionalOf(fill: HlFill): number {
  return Math.abs(Number(fill.sz)) * Number(fill.px)
}

const nyDayFormatter = new Intl.DateTimeFormat('en-CA', {
  timeZone: NY_TZ,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
})

/** Epoch ms -> NY calendar date (YYYY-MM-DD). */
export function nyDayOf(fillTimeMs: number): string {
  return nyDayFormatter.format(new Date(fillTimeMs))
}

/**
 * Aggregate one wallet's fills.
 *
 * `sinceMs` is the wallet's current watermark: fills with `time <= sinceMs`
 * are ignored (already ingested on a prior run). Pass 0 for a full backfill.
 * The dedup-by-tid is defensive -- the fetcher already dedupes across pages,
 * but a wallet appearing twice in an enumeration, or an overlapping refetch,
 * must never double-count.
 */
export function aggregateWalletFills(fills: HlFill[], sinceMs = 0): AggregateResult {
  const byDay = new Map<string, DayBucket>()
  const seenTid = new Set<string>()
  let maxFillMs = sinceMs
  let attributedFills = 0

  for (const fill of fills) {
    const t = Number(fill.time)
    if (!Number.isFinite(t) || t <= sinceMs) continue

    if (fill.tid !== undefined && fill.tid !== null) {
      const tid = String(fill.tid)
      if (seenTid.has(tid)) continue
      seenTid.add(tid)
    }

    // Watermark advances on every in-window fill, attributed or not.
    if (t > maxFillMs) maxFillMs = t

    if (!isFreeportPerpFill(fill)) continue

    const day = nyDayOf(t)
    const bucket = byDay.get(day) ?? { day, notionalUsd: 0, builderFeeUsd: 0, fillCount: 0 }
    bucket.notionalUsd += notionalOf(fill)
    bucket.builderFeeUsd += Number(fill.builderFee ?? 0)
    bucket.fillCount += 1
    byDay.set(day, bucket)
    attributedFills += 1
  }

  const buckets = [...byDay.values()].sort((a, b) => a.day.localeCompare(b.day))
  return { buckets, maxFillMs, attributedFills }
}
