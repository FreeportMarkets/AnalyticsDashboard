import { sql } from '@/lib/db'
import { nyDateExpr } from '@/lib/time'
import { nyRangeToUtc } from './nyRange'

/**
 * Ported from app.py's `tab_notifications` (lines ~2404-2554).
 *
 * Same two rules as every other metrics module: bucket with nyDateExpr('ts'),
 * filter `ts` with nyRangeToUtc bounds. Never GROUP BY or range-filter on the
 * `date` column -- scripts/lint-no-date-grouping.ts enforces this.
 *
 * Notification event shapes (inspected against live data on 2026-07-20):
 *   notification_sent            -- server-attributed (platform='server'), real
 *                                    per-user wallet_address. metadata:
 *                                    { type, title, ticker, channel, subtype,
 *                                      success, threshold_source }. `type` is
 *                                    NOT limited to the two values app.py
 *                                    special-cased ("price_alert","lifecycle")
 *                                    -- topic-based sends ("Global Conflict",
 *                                    "Fed & Macro") dominate volume in
 *                                    practice, so sendsByType groups on the
 *                                    real `type` value dynamically rather than
 *                                    replicating app.py's stale hardcoded list.
 *   notification_sent_broadcast  -- wallet_address='server'. metadata:
 *                                    { type, title, handle, ticker, channel,
 *                                      sent_count, failed_count, total_recipients }.
 *   notification_received        -- client-attributed, wallet_address is the
 *                                    real user wallet. metadata: {} (empty).
 *   notification_tap             -- client-attributed. metadata:
 *                                    { type, action, handle?, ticker? }.
 *                                    `ticker` only present for trade_alert taps.
 *
 * System-wallet exclusion is applied only to per-user aggregates (received,
 * tapped, per-wallet breakdowns) per the project's Global Constraints --
 * notification_sent/notification_sent_broadcast are server-attributed and
 * must NOT be filtered on wallet_address or the send counts vanish.
 */
const SYSTEM_WALLETS = ['server', 'unknown', 'system', '']

function addDays(date: string, days: number): string {
  const d = new Date(`${date}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

function dateRange(startDate: string, endDate: string): string[] {
  const out: string[] = []
  let cur = startDate
  while (cur <= endDate) {
    out.push(cur)
    cur = addDays(cur, 1)
  }
  return out
}

function dayKey(d: string | Date): string {
  return typeof d === 'string' ? d : d.toISOString().slice(0, 10)
}

/** "price_alert" -> "Price Alert"; "Global Conflict" -> unchanged. */
function titleCaseType(type: string): string {
  return type
    .split('_')
    .map(w => (w.length === 0 ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(' ')
}

export interface SendSummary {
  sent: number
  received: number
  tapped: number
  tapRatePct: number
  deliveryRatePct: number
}

/** Sent = targeted (success) + broadcast recipients. Mirrors app.py's total_server_sent. */
export async function sendSummary(startDate: string, endDate: string): Promise<SendSummary> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `WITH targeted AS (
        SELECT count(*) FILTER (WHERE (metadata->>'success')::boolean IS TRUE)::int AS sent
          FROM events
         WHERE event = 'notification_sent' AND ts >= $1 AND ts < $2
      ),
      broadcast AS (
        SELECT coalesce(sum((metadata->>'sent_count')::int), 0)::int AS sent
          FROM events
         WHERE event = 'notification_sent_broadcast' AND ts >= $1 AND ts < $2
      ),
      received AS (
        SELECT count(*)::int AS n
          FROM events
         WHERE event = 'notification_received' AND ts >= $1 AND ts < $2
           AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
      ),
      tapped AS (
        SELECT count(*)::int AS n
          FROM events
         WHERE event = 'notification_tap' AND ts >= $1 AND ts < $2
           AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
      )
      SELECT targeted.sent AS targeted_sent, broadcast.sent AS broadcast_sent,
             received.n AS received, tapped.n AS tapped
        FROM targeted, broadcast, received, tapped`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ targeted_sent: number; broadcast_sent: number; received: number; tapped: number }>

  const r = rows[0]
  if (!r) throw new Error('sendSummary: aggregate query returned no rows')

  const sent = r.targeted_sent + r.broadcast_sent
  return {
    sent,
    received: r.received,
    tapped: r.tapped,
    tapRatePct: sent > 0 ? (r.tapped / sent) * 100 : 0,
    deliveryRatePct: sent > 0 ? (r.received / sent) * 100 : 0,
  }
}

export interface SendsByType {
  type: string
  sends: number
  recipients: number
}

export async function sendsByType(startDate: string, endDate: string): Promise<SendsByType[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const targetedRows = (await sql(
    `SELECT coalesce(metadata->>'type', 'unknown') AS type, count(*)::int AS sends
       FROM events
      WHERE event = 'notification_sent' AND ts >= $1 AND ts < $2
        AND (metadata->>'success')::boolean IS TRUE
      GROUP BY 1
      ORDER BY sends DESC`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ type: string; sends: number }>

  const broadcastRows = (await sql(
    `SELECT count(*)::int AS events, coalesce(sum((metadata->>'sent_count')::int), 0)::int AS recipients
       FROM events
      WHERE event = 'notification_sent_broadcast' AND ts >= $1 AND ts < $2`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ events: number; recipients: number }>

  const out: SendsByType[] = targetedRows.map(r => ({
    type: titleCaseType(r.type),
    sends: r.sends,
    recipients: r.sends,
  }))

  const b = broadcastRows[0]
  if (b && b.events > 0) {
    out.push({ type: 'Trade Alerts (broadcast)', sends: b.events, recipients: b.recipients })
  }

  return out.sort((a, c) => c.recipients - a.recipients)
}

export interface TapRateDay {
  day: string
  sent: number
  tapped: number
  tapRatePct: number
}

export async function tapRateByDay(startDate: string, endDate: string): Promise<TapRateDay[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const targetedRows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day,
            count(*) FILTER (WHERE (metadata->>'success')::boolean IS TRUE)::int AS sent
       FROM events
      WHERE event = 'notification_sent' AND ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ day: string | Date; sent: number }>

  const broadcastRows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day,
            coalesce(sum((metadata->>'sent_count')::int), 0)::int AS sent
       FROM events
      WHERE event = 'notification_sent_broadcast' AND ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ day: string | Date; sent: number }>

  const tappedRows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day, count(*)::int AS tapped
       FROM events
      WHERE event = 'notification_tap' AND ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; tapped: number }>

  const targeted = new Map(targetedRows.map(r => [dayKey(r.day), r.sent]))
  const broadcast = new Map(broadcastRows.map(r => [dayKey(r.day), r.sent]))
  const tapped = new Map(tappedRows.map(r => [dayKey(r.day), r.tapped]))

  return dateRange(startDate, endDate).map(day => {
    const sent = (targeted.get(day) ?? 0) + (broadcast.get(day) ?? 0)
    const t = tapped.get(day) ?? 0
    return { day, sent, tapped: t, tapRatePct: sent > 0 ? (t / sent) * 100 : 0 }
  })
}

export interface TimeToTradeResult {
  totalTaps: number
  tradedWithin1h: number
  tradedWithin6h: number
  tradedWithin24h: number
  medianDelaySeconds: number | null
  distribution: Array<{ label: string; count: number }>
}

const DELAY_BUCKETS: Array<{ label: string; maxSeconds: number }> = [
  { label: '<1m', maxSeconds: 60 },
  { label: '1-5m', maxSeconds: 5 * 60 },
  { label: '5-15m', maxSeconds: 15 * 60 },
  { label: '15-60m', maxSeconds: 60 * 60 },
  { label: '1-6h', maxSeconds: 6 * 3600 },
  { label: '6-24h', maxSeconds: 24 * 3600 },
]

function bucketDelay(seconds: number): string {
  for (const b of DELAY_BUCKETS) {
    if (seconds < b.maxSeconds) return b.label
  }
  return DELAY_BUCKETS[DELAY_BUCKETS.length - 1]!.label
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!
}

/**
 * For each notification_tap, whether the same wallet traded within 1h / 6h /
 * 24h of the tap. Trade sources are unioned per the spec: trade_success and
 * trade_initiated events (Solana spot -- same wallet_address namespace as
 * notification_tap) plus the `trades` table (also covers HL perps, executed
 * via a derived EVM sub-account address so those rows only match wallets that
 * trade perps directly under that address).
 */
export async function timeToTradeAfterTap(startDate: string, endDate: string): Promise<TimeToTradeResult> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const tradeUpperBound = new Date(toUtc.getTime() + 24 * 3600 * 1000)

  const rows = (await sql(
    `WITH taps AS (
        SELECT wallet_address, ts AS tap_ts
          FROM events
         WHERE event = 'notification_tap' AND ts >= $1 AND ts < $2
           AND wallet_address IS NOT NULL AND wallet_address <> ALL($4::text[])
      ),
      trade_ts AS (
        SELECT wallet_address, ts FROM events
         WHERE event IN ('trade_success', 'trade_initiated') AND ts >= $1 AND ts < $3
        UNION ALL
        SELECT wallet_address, ts FROM trades
         WHERE ts >= $1 AND ts < $3
      )
      SELECT taps.tap_ts,
             (SELECT min(tt.ts) FROM trade_ts tt
               WHERE tt.wallet_address = taps.wallet_address
                 AND tt.ts > taps.tap_ts
                 AND tt.ts < taps.tap_ts + interval '24 hours'
             ) AS first_trade_ts
        FROM taps`,
    [fromUtc.toISOString(), toUtc.toISOString(), tradeUpperBound.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ tap_ts: Date | string; first_trade_ts: Date | string | null }>

  let tradedWithin1h = 0
  let tradedWithin6h = 0
  let tradedWithin24h = 0
  const delays: number[] = []
  const bucketCounts = new Map<string, number>(DELAY_BUCKETS.map(b => [b.label, 0]))

  for (const r of rows) {
    if (!r.first_trade_ts) continue
    const tapTs = new Date(r.tap_ts).getTime()
    const tradeTs = new Date(r.first_trade_ts).getTime()
    const delaySeconds = (tradeTs - tapTs) / 1000
    delays.push(delaySeconds)
    if (delaySeconds <= 3600) tradedWithin1h++
    if (delaySeconds <= 6 * 3600) tradedWithin6h++
    tradedWithin24h++
    const label = bucketDelay(delaySeconds)
    bucketCounts.set(label, (bucketCounts.get(label) ?? 0) + 1)
  }

  return {
    totalTaps: rows.length,
    tradedWithin1h,
    tradedWithin6h,
    tradedWithin24h,
    medianDelaySeconds: median(delays),
    distribution: DELAY_BUCKETS.map(b => ({ label: b.label, count: bucketCounts.get(b.label) ?? 0 })),
  }
}

export interface TappedTicker {
  ticker: string
  taps: number
}

export async function mostTappedTickers(
  startDate: string, endDate: string, limit = 10
): Promise<TappedTicker[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT metadata->>'ticker' AS ticker, count(*)::int AS taps
       FROM events
      WHERE event = 'notification_tap' AND ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($4::text[])
        AND metadata->>'ticker' IS NOT NULL AND metadata->>'ticker' <> ''
      GROUP BY 1
      ORDER BY taps DESC
      LIMIT $3`,
    [fromUtc.toISOString(), toUtc.toISOString(), limit, SYSTEM_WALLETS]
  )) as Array<{ ticker: string; taps: number }>
  return rows
}

export interface EngagedUser {
  walletAddress: string
  taps: number
  sent: number
  tapRatePct: number
}

export async function mostEngagedUsers(
  startDate: string, endDate: string, limit = 10
): Promise<EngagedUser[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const tapRows = (await sql(
    `SELECT wallet_address, count(*)::int AS taps
       FROM events
      WHERE event = 'notification_tap' AND ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($4::text[])
      GROUP BY 1
      ORDER BY taps DESC
      LIMIT $3`,
    [fromUtc.toISOString(), toUtc.toISOString(), limit, SYSTEM_WALLETS]
  )) as Array<{ wallet_address: string; taps: number }>

  if (tapRows.length === 0) return []

  const wallets = tapRows.map(r => r.wallet_address)
  const sentRows = (await sql(
    `SELECT wallet_address, count(*) FILTER (WHERE (metadata->>'success')::boolean IS TRUE)::int AS sent
       FROM events
      WHERE event = 'notification_sent' AND ts >= $1 AND ts < $2
        AND wallet_address = ANY($3::text[])
      GROUP BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), wallets]
  )) as Array<{ wallet_address: string; sent: number }>
  const sentByWallet = new Map(sentRows.map(r => [r.wallet_address, r.sent]))

  return tapRows.map(r => {
    const sent = sentByWallet.get(r.wallet_address) ?? 0
    return {
      walletAddress: r.wallet_address,
      taps: r.taps,
      sent,
      tapRatePct: sent > 0 ? (r.taps / sent) * 100 : 0,
    }
  })
}
