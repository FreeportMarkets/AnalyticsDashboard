/**
 * Deposit totals, read from the trading backend.
 *
 * The second metric on this console that does not come from Neon or DynamoDB,
 * and for the same reason as the first (see `funnelApi.ts`): deposit events are
 * written by the mobile app to `trading-api/v1/events/batch` -> RDS, which
 * lives in a private subnet with ECS-only ingress. Vercel cannot query it, so
 * the backend serves the aggregate.
 *
 * This exists because the Deposits page previously read Neon's `events` table,
 * which the mobile app stopped writing to on 2026-08-08. The query and the
 * event names were correct; the table was not, and the page answered 0 for
 * weeks without anything appearing broken.
 */

const BASE = process.env.TRADING_API_BASE_URL ?? 'https://trading-api.freeportmarkets.com'

export interface DepositsProviderRow {
  provider: string
  initiated: number
  success: number
  error: number
  conversion: number
}

export interface DepositsResponse {
  trace_id?: string
  range: { from: string; to: string }
  totals: { initiated: number; success: number; error: number; conversion: number }
  by_provider: DepositsProviderRow[]
}

export type DepositsFailure = 'not_configured' | 'unauthorized' | 'bad_range' | 'unreachable'

export type DepositsResult =
  | { ok: true; data: DepositsResponse }
  /**
   * `traceId` is the backend's, echoed in every response body including the
   * 400 and 401. It is what turns "the page is erroring" into a single
   * CloudWatch lookup (saved query `analytics/deposits-read-rejected`), so it
   * is carried through to the thrown error rather than discarded here.
   */
  | { ok: false; reason: DepositsFailure; traceId?: string }

export async function fetchDeposits(opts: {
  /** ISO `YYYY-MM-DD`. The API rejects malformed dates rather than widening. */
  from?: string
  to?: string
  days?: number
}): Promise<DepositsResult> {
  // Shares the funnel secret: one caller, one class of data, one credential to
  // rotate. Missing config fails closed rather than reading as an empty range.
  const secret = process.env.ANALYTICS_FUNNEL_READ_SECRET
  if (!secret) return { ok: false, reason: 'not_configured' }

  const params = new URLSearchParams({ days: String(opts.days ?? 30) })
  if (opts.from) params.set('from', opts.from)
  if (opts.to) params.set('to', opts.to)

  try {
    const res = await fetch(`${BASE}/v1/analytics/deposits?${params.toString()}`, {
      headers: { 'x-funnel-secret': secret },
      signal: AbortSignal.timeout(10_000),
      // Deposits land continuously rather than on a rollup schedule, so this is
      // shorter than the funnel's cache: a minute-old number would be visibly
      // behind while someone is watching a payment go through.
      next: { revalidate: 30 },
    })
    if (!res.ok) {
      // The API rejects an impossible range rather than quietly widening it, so
      // a 400 is a bad URL, not a backend fault — say so, or the page blames
      // the backend for a typo in the address bar.
      const reason: DepositsFailure =
        res.status === 401 ? 'unauthorized' : res.status === 400 ? 'bad_range' : 'unreachable'
      // Best effort: an error body may not be JSON at all (a gateway's own 502
      // page, say). Losing the trace id must not turn a 401 into a parse crash.
      let traceId: string | undefined
      try {
        traceId = ((await res.json()) as { trace_id?: string }).trace_id
      } catch {
        traceId = undefined
      }
      return { ok: false, reason, traceId }
    }
    return { ok: true, data: (await res.json()) as DepositsResponse }
  } catch {
    return { ok: false, reason: 'unreachable' }
  }
}
