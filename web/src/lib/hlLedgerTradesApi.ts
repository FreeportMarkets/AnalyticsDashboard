import { nyRangeToUtc } from './metrics/nyRange'

export interface HlLedgerTrade {
  fillId: string
  orderId: string | null
  walletAddress: string
  market: string
  side: 'B' | 'A'
  direction: string | null
  size: string
  price: string
  notionalUsd: string
  filledAt: string
}

export function parseHlLedgerTrades(value: unknown, from: string, to: string, limit: number): HlLedgerTrade[] {
  const report = value as Record<string, unknown> | null
  const range = report?.range as Record<string, unknown> | undefined
  if (!report || report.schemaVersion !== 1 || report.timezone !== 'America/New_York' ||
    report.completeness !== 'provisional' || range?.from !== from || range?.to !== to ||
    !Array.isArray(report.trades) || report.trades.length > limit) throw Error('HL trades contract mismatch')
  const ids = new Set<string>()
  const { fromUtc, toUtc } = nyRangeToUtc(from, to)
  const money = (v: unknown) => typeof v === 'string' && /^\d+(?:\.\d+)?$/.test(v) && Number.isFinite(Number(v)) && Number(v) >= 0
  return report.trades.map((row: HlLedgerTrade) => {
    if (!row || typeof row.fillId !== 'string' || !/^\d+$/.test(row.fillId) || ids.has(row.fillId) ||
      !(row.orderId === null || (typeof row.orderId === 'string' && /^\d+$/.test(row.orderId))) ||
      typeof row.walletAddress !== 'string' || !/^0x[0-9a-f]{40}$/i.test(row.walletAddress) ||
      typeof row.market !== 'string' || !row.market || !['B', 'A'].includes(row.side) ||
      !(row.direction === null || typeof row.direction === 'string') ||
      !money(row.size) || !money(row.price) || !money(row.notionalUsd) ||
      typeof row.filledAt !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(row.filledAt) ||
      !Number.isFinite(Date.parse(row.filledAt)) || new Date(row.filledAt).toISOString() !== row.filledAt ||
      Date.parse(row.filledAt) < fromUtc.getTime() || Date.parse(row.filledAt) >= toUtc.getTime()) throw Error('HL trades contract mismatch')
    ids.add(row.fillId)
    return row
  })
}

export async function fetchHlLedgerTrades(from: string, to: string, limit: number): Promise<HlLedgerTrade[]> {
  const secret = process.env.ANALYTICS_FUNNEL_READ_SECRET
  if (!secret) throw Error('HL trades read secret is not configured')
  const base = process.env.TRADING_API_BASE_URL ?? 'https://trading-api.freeportmarkets.com'
  const response = await fetch(`${base}/v1/analytics/hl-ledger/trades?${new URLSearchParams({ from, to, limit: String(limit) })}`, {
    headers: { 'x-funnel-secret': secret }, signal: AbortSignal.timeout(10_000), cache: 'no-store',
  })
  if (!response.ok) throw Error(`HL trades read failed (${response.status})`)
  return parseHlLedgerTrades(await response.json(), from, to, limit)
}
