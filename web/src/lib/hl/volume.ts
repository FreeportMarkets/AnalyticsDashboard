/**
 * HL-sourced perp volume -- 1:1 with Hyperliquid per-fill data.
 *
 * Direct port of `hl_volume.py`. Keep the two in lockstep: the whole point
 * of this file is that the Vercel dashboard and the Streamlit dashboard
 * publish the SAME volume figure, and they only do that while both compute
 * it the same way. `test/hlVolume.test.ts` mirrors `test_hl_volume.py`
 * case for case.
 *
 * Volume is computed from HL `userFillsByTime` (every fill's price * size),
 * the same per-fill notional HL charges builder fees on. This is exact,
 * unlike reconstructing from our trade log (which stores intended order
 * size, not filled size, so it drifts high on partial/IOC fills).
 *
 * Only PERP fills count (open/close, long/short). Spot fills (dir "Buy" /
 * "Sell") and "Settlement" are excluded. Fills are deduped by trade id
 * (`tid`).
 */

import { hlInfoPost } from './client'

export const PERP_DIRS = new Set(['Open Long', 'Close Long', 'Open Short', 'Close Short'])

/** HL returns at most 2000 fills per `userFillsByTime` call. */
export const HL_PAGE_CAP = 2000

export interface HlFill {
  /** Size, as a decimal string. */
  sz: string
  /** Price, as a decimal string. */
  px: string
  /** Direction, e.g. "Open Long". Absent/other values are not perp fills. */
  dir?: string
  /** Trade id -- the dedup key. */
  tid?: number | string
  /** Fill time, epoch milliseconds. */
  time?: number | string
  /**
   * Builder fee charged on this fill, as a decimal string. Present and > 0
   * only when the order carried a builder code. For Freeport attribution
   * this is HL's own stamp that the fill went through our builder
   * (see lib/volume/aggregate.ts). Absent/0 = not a Freeport-attributed fill.
   */
  builderFee?: string
  /** TWAP order id, set on fills that are sub-executions of a TWAP order. */
  twapId?: number | string | null
}

/** One fill's notional = |size| * price. Always positive. */
export function fillNotional(fill: HlFill): number {
  return Math.abs(Number(fill.sz)) * Number(fill.px)
}

/** True only for the four perp directions; excludes spot + settlement. */
export function isPerpFill(fill: HlFill): boolean {
  return fill.dir !== undefined && PERP_DIRS.has(fill.dir)
}

/** Sum |size|*price over perp fills, deduped by `tid`. */
export function perpVolumeFromFills(fills: HlFill[]): number {
  const seen = new Set<string>()
  let total = 0
  for (const f of fills) {
    if (f.tid !== undefined && f.tid !== null) {
      const tid = String(f.tid)
      if (seen.has(tid)) continue
      seen.add(tid)
    }
    if (isPerpFill(f)) total += fillNotional(f)
  }
  return total
}

/**
 * Fetch each wallet's fills, keep those in [startMs, endMs), and sum the
 * perp notional. `fetcher(wallet, startMs)` returns HL fill objects --
 * injected so this is testable without network and swappable for caching.
 */
export async function computePerpVolume(
  wallets: readonly string[],
  startMs: number,
  endMs: number,
  fetcher: (wallet: string, startMs: number) => Promise<HlFill[]>
): Promise<number> {
  const seen = new Set<string>()
  const deduped: HlFill[] = []
  for (const wallet of wallets) {
    for (const f of await fetcher(wallet, startMs)) {
      if (f.time !== undefined && f.time !== null) {
        const t = Number(f.time)
        if (!(t >= startMs && t < endMs)) continue
      }
      if (f.tid !== undefined && f.tid !== null) {
        const tid = String(f.tid)
        if (seen.has(tid)) continue
        seen.add(tid)
      }
      deduped.push(f)
    }
  }
  return perpVolumeFromFills(deduped)
}

// --- real network fetcher (not exercised by the pure unit tests) ---

async function hlPost(body: unknown): Promise<HlFill[]> {
  return hlInfoPost<HlFill[]>(body)
}

/**
 * Live HL `userFillsByTime` for one wallet, paginated past the 2000-fill cap.
 *
 * HL caps each response at 2000 fills; a busy wallet over a multi-day range
 * has more, so a single call silently truncates (undercounts volume). We
 * page by advancing `startTime` to the last fill's time and dedup by `tid`
 * to absorb the boundary overlap. `postFn` is injected for testing.
 * Terminates when a page is under the cap or no new fills appear
 * (degenerate same-timestamp page).
 */
export async function hlFillsFetcher(
  wallet: string,
  startMs: number,
  postFn: (body: unknown) => Promise<HlFill[]> = hlPost
): Promise<HlFill[]> {
  const out: HlFill[] = []
  const seen = new Set<string>()
  let cur = startMs

  for (;;) {
    const batch = await postFn({ type: 'userFillsByTime', user: wallet, startTime: cur })
    if (!batch || batch.length === 0) break

    const fresh = batch.filter(f => !seen.has(String(f.tid)))
    for (const f of fresh) seen.add(String(f.tid))
    out.push(...fresh)

    if (batch.length < HL_PAGE_CAP || fresh.length === 0) break

    const last = Math.max(...batch.map(f => Number(f.time)))
    // Advance; re-fetch the boundary and let dedup absorb the overlap.
    cur = last > cur ? last : cur + 1
  }
  return out
}
