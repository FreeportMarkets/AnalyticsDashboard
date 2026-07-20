import { mapTrade, type TradeRow } from './mapTrade'
import { computeNextWatermark, datesToScan } from './watermark'
import type { SyncResult } from './types'

export type TradeFetcher = (tradeDate: string, afterTimestamp: string) => Promise<unknown[]>

export interface SyncTradesDeps {
  now: Date
  readWatermark: () => Promise<Date>
  advanceWatermark: (source: string, ts: Date) => Promise<void>
  fetchTrades: TradeFetcher
  insertTrades: (
    rows: TradeRow[],
    onRowFailure?: (row: TradeRow, reason: string) => Promise<void>
  ) => Promise<number>
  quarantine: (raw: unknown, reason: string) => Promise<void>
}

export const TRADES_SOURCE = 'trades'

export async function syncTrades(deps: SyncTradesDeps): Promise<SyncResult> {
  const watermark = await deps.readWatermark()
  const dates = datesToScan(deps.now)
  const after = watermark.toISOString()

  let scanned = 0
  let inserted = 0
  let quarantined = 0

  for (const date of dates) {
    const items = await deps.fetchTrades(date, after)
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

  const next = computeNextWatermark(deps.now)
  await deps.advanceWatermark(TRADES_SOURCE, next)

  return { scanned, inserted, quarantined, watermark: next.toISOString() }
}
