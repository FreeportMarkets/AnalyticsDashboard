export interface HlLedgerDay {
  day: string
  fillCount: number
  notionalUsd: string
  confirmedFeeUsd: string
  unresolvedBuilderFeeUsd: string
  unresolvedFillCount: number
  latestFillAt: string | null
  latestIngestAt: string | null
}

export interface HlLedgerMetrics {
  schemaVersion: 1
  timezone: 'America/New_York'
  completeness: 'provisional'
  range: { from: string; to: string }
  days: HlLedgerDay[]
}

const decimal = (value: unknown): value is string =>
  typeof value === 'string' && /^\d+(?:\.\d+)?$/.test(value) && Number.isFinite(Number(value))
const date = (value: unknown): value is string => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)

export function isHlLedgerMetrics(value: unknown, from: string, to: string): value is HlLedgerMetrics {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const report = value as Record<string, unknown>
  if (report.schemaVersion !== 1 || report.timezone !== 'America/New_York' ||
    report.completeness !== 'provisional' || !report.range ||
    typeof report.range !== 'object' || Array.isArray(report.range) ||
    (report.range as Record<string, unknown>).from !== from ||
    (report.range as Record<string, unknown>).to !== to || !Array.isArray(report.days)) return false
  const days = new Set<string>()
  return report.days.every((item: unknown) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return false
    const row = item as Record<string, unknown>
    if (!date(row.day) || row.day < from || row.day > to || days.has(row.day) ||
      !Number.isSafeInteger(row.fillCount) || Number(row.fillCount) < 0 ||
      !Number.isSafeInteger(row.unresolvedFillCount) || Number(row.unresolvedFillCount) < 0 ||
      Number(row.unresolvedFillCount) > Number(row.fillCount) ||
      !decimal(row.notionalUsd) || !decimal(row.confirmedFeeUsd) ||
      !decimal(row.unresolvedBuilderFeeUsd)) return false
    days.add(row.day)
    return true
  })
}

export async function fetchHlLedgerMetrics(from: string, to: string): Promise<HlLedgerMetrics> {
  const secret = process.env.ANALYTICS_FUNNEL_READ_SECRET
  if (!secret) throw Error('HL ledger read secret is not configured')
  const base = process.env.TRADING_API_BASE_URL ?? 'https://trading-api.freeportmarkets.com'
  const response = await fetch(`${base}/v1/analytics/hl-ledger?${new URLSearchParams({ from, to })}`, {
    headers: { 'x-funnel-secret': secret }, signal: AbortSignal.timeout(10_000), next: { revalidate: 60 },
  })
  if (!response.ok) throw Error(`HL ledger read failed (${response.status})`)
  const payload: unknown = await response.json()
  if (!isHlLedgerMetrics(payload, from, to)) throw Error('HL ledger response contract mismatch')
  return payload
}
