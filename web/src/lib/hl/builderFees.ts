/**
 * Top-down builder-fee total for the Freeport collector wallet.
 *
 * This is the reconciliation target: the exact USDC HL has paid us for orders
 * that carried our builder code, whether already swept to the wallet
 * (`claimed`) or still accruing in HL's rewards pool (`pending`). The
 * bottom-up sum of every fill's `builderFee` must equal this to within <1%;
 * that equality is the proof that we enumerated every Freeport wallet.
 *
 * Source: HL `info { type: "referral", user: <builder> }`. Builder fees and
 * referral rewards share one pool (see the backend CLAUDE.md gotcha), so this
 * separates them: `builderRewards` is the pure builder accrual (pending), and
 * the ledger's `rewardsClaim` entries are what was claimed. For the Freeport
 * collector, referral volume is negligible (`cumVlm` ~ $90), so `claimed`
 * here is effectively all builder fees; we still read it from the dedicated
 * fields rather than assume.
 */

import { hlInfoPost } from './client'

export const FREEPORT_BUILDER_ADDRESS =
  process.env.HL_BUILDER_ADDRESS ?? '0x9f4e80F17Ddb4A7efC1dc07fAE6B34AbAb77d6Df'

export interface BuilderFeeTotals {
  /** USDC already claimed into the collector wallet. */
  claimedUsdc: number
  /** USDC accrued but not yet claimed (HL rewards pool). */
  pendingUsdc: number
  /** claimed + pending: total builder fees earned, ever. */
  totalUsdc: number
}

interface TokenState {
  cumVlm?: string
  unclaimedRewards?: string
  claimedRewards?: string
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

  const pendingUsdc = Number(usdc?.builderRewards ?? ref.builderRewards ?? 0)
  const claimedUsdc = Number(usdc?.claimedRewards ?? 0)

  return {
    claimedUsdc,
    pendingUsdc,
    totalUsdc: claimedUsdc + pendingUsdc,
  }
}
