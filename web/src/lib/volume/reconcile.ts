import type { neon } from '@neondatabase/serverless'
import { builderFeeTotals, FREEPORT_BUILDER_ADDRESS } from '@/lib/hl/builderFees'
import { hlInfoPost } from '@/lib/hl/client'
import { bottomUpBuilderFee, logReconciliation } from './store'

type SqlTag = ReturnType<typeof neon<boolean, boolean>>

export interface Reconciliation {
  bottomUpFeeUsd: number
  topDownFeeUsd: number
  ratio: number
  gapUsd: number
  /** True when |1 - ratio| < tolerance -- enumeration is provably complete. */
  complete: boolean
}

/**
 * Compare the fees we summed bottom-up (Σ over stored fills) against the fees
 * HL actually paid our collector (top-down). This is the correctness gate for
 * the whole system: they can only match if every Freeport wallet was scanned
 * and every fill counted, because the fee total cannot be reproduced any
 * other way. A ratio well under 1.0 means wallets are missing from
 * enumeration; over 1.0 would mean double-counting.
 */
export async function reconcile(
  sql: SqlTag,
  opts: {
    walletsTracked: number
    tolerance?: number
    note?: string
    log?: boolean
    /** Absolute epoch-ms deadline for the HL fee lookup (cron safety). */
    deadlineMs?: number
  } = { walletsTracked: 0 }
): Promise<Reconciliation> {
  const tolerance = opts.tolerance ?? 0.01
  // Bound the single HL fee call so reconcile can't hang the cron: short
  // retry budget and the shared deadline.
  const feePost = (body: unknown) =>
    hlInfoPost<unknown>(body, { deadlineMs: opts.deadlineMs, maxRetries: 3 })
  const [bottomUp, top] = await Promise.all([
    bottomUpBuilderFee(sql),
    builderFeeTotals(FREEPORT_BUILDER_ADDRESS, feePost),
  ])
  const topDown = top.totalUsdc
  const ratio = topDown > 0 ? bottomUp / topDown : 0

  if (opts.log !== false) {
    await logReconciliation(sql, {
      bottomUpFeeUsd: bottomUp,
      topDownFeeUsd: topDown,
      walletsTracked: opts.walletsTracked,
      note: opts.note,
    })
  }

  return {
    bottomUpFeeUsd: bottomUp,
    topDownFeeUsd: topDown,
    ratio,
    gapUsd: topDown - bottomUp,
    complete: Math.abs(1 - ratio) < tolerance,
  }
}
