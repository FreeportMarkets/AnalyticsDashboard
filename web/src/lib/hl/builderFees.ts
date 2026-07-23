/**
 * Top-down builder-fee total for the Freeport collector wallet.
 *
 * This is the reconciliation target: the exact USDC HL has paid us for orders
 * that carried our builder code, whether already swept to the wallet
 * (`claimed`) or still accruing in HL's rewards pool (`pending`). The
 * bottom-up sum of every fill's `builderFee` must equal this to within <1%;
 * that equality is the proof that we enumerated every Freeport wallet.
 *
 * Source: HL `info { type: "referral", user: <builder> }`.
 *
 * CRITICAL FIELD SEMANTICS (verified against the live collector, do not
 * "simplify"): in `tokenToState[0]` (USDC),
 *
 *     builderRewards === claimedRewards + unclaimedRewards
 *     20945.20        === 18818.22      + 2126.98
 *
 * `builderRewards` is the CUMULATIVE total builder fees ever earned; claimed
 * and unclaimed are its two halves, NOT separate pools to add on top. An
 * earlier version summed claimed + builderRewards and double-counted the
 * claimed half, producing a bogus $39,763 target that made a complete
 * backfill look like it reconciled at 53%. The total is `builderRewards`.
 *
 * Referral rewards share the same HL pool, but for this collector referral
 * volume is negligible (`cumVlm` ~ $90), so `builderRewards` is effectively
 * pure builder fees.
 */

import { hlInfoPost } from './client'

export const FREEPORT_BUILDER_ADDRESS =
  process.env.HL_BUILDER_ADDRESS ?? '0x9f4e80F17Ddb4A7efC1dc07fAE6B34AbAb77d6Df'

export interface BuilderFeeTotals {
  /** USDC already claimed into the collector wallet. */
  claimedUsdc: number
  /** USDC accrued but not yet claimed. */
  pendingUsdc: number
  /**
   * Cumulative builder fees earned, ever (= claimed + pending). Read straight
   * from HL's `builderRewards`, NOT computed as claimed + builderRewards --
   * see the field-semantics note above. This is the reconciliation target.
   */
  totalUsdc: number
}

interface TokenState {
  cumVlm?: string
  unclaimedRewards?: string
  claimedRewards?: string
  /** Cumulative builder fees ever (= claimedRewards + unclaimedRewards). */
  builderRewards?: string
}
interface ReferralResponse {
  builderRewards?: string
  tokenToState?: Array<[number, TokenState]>
}

/**
 * Read the collector's builder-fee totals. `postFn` is injected for testing.
 *
 * USDC is token index 0 in `tokenToState`. `builderRewards` there is the
 * pending builder pool; `claimedRewards` is the historical claimed total.
 * The top-level `builderRewards` is the same pending figure and is used as a
 * fallback if `tokenToState` is absent.
 */
export async function builderFeeTotals(
  builderAddress: string = FREEPORT_BUILDER_ADDRESS,
  postFn: (body: unknown) => Promise<unknown> = body => hlInfoPost(body)
): Promise<BuilderFeeTotals> {
  const ref = (await postFn({ type: 'referral', user: builderAddress })) as ReferralResponse
  const usdc = ref.tokenToState?.find(([token]) => token === 0)?.[1]

  // `builderRewards` is the cumulative total; claimed + unclaimed are its
  // halves. total = builderRewards (never claimed + builderRewards).
  const totalUsdc = Number(usdc?.builderRewards ?? ref.builderRewards ?? 0)
  const claimedUsdc = Number(usdc?.claimedRewards ?? 0)
  const pendingUsdc = Number(usdc?.unclaimedRewards ?? Math.max(0, totalUsdc - claimedUsdc))

  return { claimedUsdc, pendingUsdc, totalUsdc }
}
