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
 * Non-perp fill directions to EXCLUDE. Denylist, not an allowlist: Freeport is
 * perps-only, so any fill on an embedded wallet that ISN'T a spot conversion
 * or settlement is a Freeport perp trade. This captures every perp direction
 * -- opens, closes, position flips (`Long > Short`), and any future dir HL
 * adds -- without an allowlist that would silently drop advanced orders.
 */
const NON_PERP_DIRS = new Set(['Buy', 'Sell', 'Spot Dust Conversion', 'Settlement'])

/**
 * True for a Freeport perp fill (any perp direction; excludes spot/settlement).
 *
 * IMPORTANT: this is NOT gated on `builderFee > 0`. It counts ALL perp fills
 * of the wallet, including LIQUIDATIONS and TP/SL auto-closes -- which HL
 * executes itself, so they carry no builder fee (no `cloid`, no builder
 * param) even though they close a genuine Freeport position. Excluding them
 * undercounted vs the Streamlit dashboard by exactly the liquidation volume
 * (measured: 69 forced-close fills / ~$145k, almost all one SNDK liquidation).
 * Enumeration is Privy-embedded wallets, which only trade through Freeport, so
 * all their perp fills are Freeport volume.
 *
 * `builderFee` is still summed per fill (see aggregate) and reconciled against
 * the collector's fees -- that remains the check that every FEE-bearing fill
 * was captured. Volume is the superset (fills + liquidations); fees are the
 * attributed subset.
 */
export function isFreeportPerpFill(fill: HlFill): boolean {
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
