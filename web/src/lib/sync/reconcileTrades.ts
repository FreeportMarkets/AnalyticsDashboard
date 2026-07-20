import { mapTrade, type TradeRow } from './mapTrade'
import { datesToScan } from './watermark'
import type { TradeFetcher } from './syncTrades'
import type { SyncResult } from './types'
import { NOT_ADVANCED } from './reconcileEvents'

export interface ReconcileTradesDeps {
  now: Date
  fetchTrades: TradeFetcher
  insertTrades: (
    rows: TradeRow[],
    onRowFailure?: (row: TradeRow, reason: string) => Promise<void>
  ) => Promise<number>
  quarantine: (raw: unknown, reason: string) => Promise<void>
  /** Present only so a test can prove it is never invoked. See reconcileEvents.ts. */
  advanceWatermark: (source: string, ts: Date) => Promise<void>
}

/**
 * Trades counterpart of `reconcileEvents` -- full re-read of the last two UTC
 * trade_date partitions, watermark untouched. See reconcileEvents.ts's doc
 * comment for the full rationale. Idempotent via trades' DO UPDATE (both
 * writers use PutCommand, which overwrites in place -- see insertTrades.ts).
 */
export async function reconcileTrades(deps: ReconcileTradesDeps): Promise<SyncResult> {
  const dates = datesToScan(deps.now)

  let scanned = 0
  let inserted = 0
  let quarantined = 0

  for (const date of dates) {
    const items = await deps.fetchTrades(date, '')
    scanned += items.length

    const rows: TradeRow[] = []
    for (const item of items) {
      const mapped = mapTrade(item)
      if (mapped.ok) rows.push(mapped.value)
      else {
        await deps.quarantine(item, mapped.reason)
        quarantined += 1
      }
    }
    if (rows.length > 0) {
      inserted += await deps.insertTrades(rows, (row, reason) => deps.quarantine(row, reason))
    }
  }

  return { scanned, inserted, quarantined, watermark: NOT_ADVANCED }
}
