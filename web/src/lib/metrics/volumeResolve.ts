/**
 * Pure volume-resolution logic, split out from overview.ts (which imports
 * `@/lib/db` at module scope and throws if DATABASE_URL isn't set) so this
 * can be unit-tested with no database at all -- same reasoning as
 * staleness-format.ts.
 */

export interface KpiValueLike {
  current: number
  previous: number
}

export interface VolumeResult extends KpiValueLike {
  source: 'hl' | 'est'
}

/**
 * Resolve the Overview headline "Volume" figure.
 *
 * `hlVol` (from `wallet_volume_daily`, see hlVolumeRead.ts) is Hyperliquid
 * PERPS volume only -- authoritative, but it has no concept of swaps (see
 * docs/volume-tracking.md). Once `hasData` is true, it must be added to
 * swap volume (exact, straight from `amount_usd`), never used alone -- the
 * Trades page already does this (`perpsVolume + swap.volumeUsd` in
 * trades/page.tsx). Returning `hlVol` alone here previously made the
 * headline tile silently perps-only the moment the HL backfill landed,
 * permanently dropping every swap dollar while Trades kept reporting the
 * true total for the same range. See volume-parity.integration.test.ts,
 * which pins this the same way it already pins the "est." path.
 */
export function resolveVolume(
  hlVol: { current: number; previous: number; hasData: boolean },
  kpis: { volumeUsd: KpiValueLike; swapVolumeUsd: KpiValueLike }
): VolumeResult {
  if (hlVol.hasData) {
    return {
      current: hlVol.current + kpis.swapVolumeUsd.current,
      previous: hlVol.previous + kpis.swapVolumeUsd.previous,
      source: 'hl',
    }
  }
  return { current: kpis.volumeUsd.current, previous: kpis.volumeUsd.previous, source: 'est' }
}
