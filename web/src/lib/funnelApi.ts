/**
 * Acquisition funnel, read from the trading backend.
 *
 * This is the ONE metric on the console that does not come from Neon or
 * DynamoDB. The funnel is keyed on `anon_id` — a device identity that exists
 * before there is an account — and those events live in the trading backend's
 * RDS, which is private and ECS-only. So the backend pre-computes the rollup
 * and serves it here rather than us re-deriving it from synced events, which
 * are keyed on `wallet_address` and therefore cannot see the pre-account steps
 * at all.
 *
 * Semantics (what "matured" means, why a step can read higher than the one
 * above it): docs/analytics-funnel.md in freeport-trading-backend.
 */

export interface FunnelStep {
  step_index: number
  step_key: string
  cohort_size: number
  users_reached: number
  /** 0..1, computed by the API so every caller agrees on the number. */
  conversion: number
  p50_ms: number | null
  p90_ms: number | null
}

/**
 * What the response actually covers, as opposed to what was requested.
 *
 * `days` is the range we ASKED for (30). `coverage` is what exists. They are
 * almost never the same: instrumentation started 2026-08-10, and a cohort day
 * is only published once its window has fully elapsed (2 days for 24h, 8 days
 * for 7d). Printing the requested range beside a total summed over the actual
 * one is what made "73 new users / last 30 days" read as "73 users today".
 */
export interface FunnelCoverage {
  cohort_start: string | null
  cohort_end: string | null
  cohort_days: number
}

/**
 * The unmatured tail: the newest cohort days, which have NOT had their full
 * window yet and whose counts can therefore only rise.
 *
 * Kept apart from `steps` on purpose. A partial day's later steps are
 * undercounted by construction — those users have not had their chance — so
 * adding the two together yields a confident-looking number belonging to no
 * real population. Render it as its own block, never folded into the headline.
 *
 * Undefined/null when the backend has the feature off or the range ends in the
 * past; the page then omits the section entirely.
 */
export interface FunnelProvisional {
  coverage: FunnelCoverage
  steps: FunnelStep[]
}

export interface FunnelResponse {
  window: '24h' | '7d'
  segment: { key: string; value: string }
  days: number
  /** The explicit cohort-date range, when one was asked for. */
  range?: { from: string; to: string } | null
  /** Older backends may not send this; callers must tolerate undefined. */
  coverage?: FunnelCoverage
  steps: FunnelStep[]
  /** Older backends may not send this; callers must tolerate undefined. */
  provisional?: FunnelProvisional | null
}

export type FunnelResult =
  | { ok: true; data: FunnelResponse }
  | { ok: false; reason: 'not_configured' | 'unauthorized' | 'unreachable' | 'bad_range' }

const BASE = process.env.TRADING_API_BASE_URL ?? 'https://trading-api.freeportmarkets.com'

/**
 * Server-side only — the secret must never reach the browser, so this is
 * called from a server component and the result is rendered, not the request.
 */
export async function fetchFunnel(opts: {
  window?: '24h' | '7d'
  days?: number
  segmentKey?: 'overall' | 'platform'
  segmentValue?: string
  /** ISO `YYYY-MM-DD`. Sent only when set; the API rejects malformed dates. */
  from?: string
  to?: string
}): Promise<FunnelResult> {
  const secret = process.env.ANALYTICS_FUNNEL_READ_SECRET
  if (!secret) return { ok: false, reason: 'not_configured' }

  const params = new URLSearchParams({
    window: opts.window ?? '7d',
    days: String(opts.days ?? 30),
    segment_key: opts.segmentKey ?? 'overall',
  })
  if (opts.segmentKey === 'platform' && opts.segmentValue) {
    params.set('segment_value', opts.segmentValue)
  }
  if (opts.from) params.set('from', opts.from)
  if (opts.to) params.set('to', opts.to)

  try {
    const res = await fetch(`${BASE}/v1/analytics/funnel?${params.toString()}`, {
      headers: { 'x-funnel-secret': secret },
      signal: AbortSignal.timeout(10_000),
      // The rollup only moves every 15 minutes, so a short cache spares the
      // API on refreshes without ever showing a stale-by-hours number.
      next: { revalidate: 60 },
    })
    if (res.status === 401) return { ok: false, reason: 'unauthorized' }
    // The API rejects a malformed or impossible range rather than quietly
    // widening it, so this is a bad URL, not a backend problem — say so, or the
    // page blames the backend for a typo in the address bar.
    if (res.status === 400) return { ok: false, reason: 'bad_range' }
    if (!res.ok) return { ok: false, reason: 'unreachable' }
    return { ok: true, data: (await res.json()) as FunnelResponse }
  } catch {
    // A dashboard page that throws on a backend blip is worse than one that
    // says the backend is unreachable.
    return { ok: false, reason: 'unreachable' }
  }
}

/** Human labels. The event names are precise but not readable in a chart. */
export const STEP_LABELS: Record<string, string> = {
  app_first_open: 'Opened the app',
  intro_started: 'Started the intro',
  intro_completed: 'Finished the intro',
  auth_succeeded: 'Signed in',
  deposit_opened: 'Opened deposit',
  deposit_method_selected: 'Chose a funding method',
  deposit_started: 'Started payment',
  deposit_completed: 'Completed payment',
  deposit_funds_arrived: 'Funds arrived',
  trade_submitted: 'Placed a trade',
  trade_succeeded: 'Trade filled',
}

/** "10 Aug" / "10-18 Aug" / "28 Jul - 3 Aug". Undated when nothing has matured. */
export function formatCohortRange(c?: FunnelCoverage): string | null {
  if (!c?.cohort_start || !c?.cohort_end) return null
  // Parsed as UTC noon: the API returns a bare date and `new Date('2026-08-10')`
  // is midnight UTC, which renders as the 9th anywhere west of Greenwich.
  const at = (d: string) => new Date(`${d}T12:00:00Z`)
  const day = (d: Date) => d.getUTCDate()
  const mon = (d: Date) =>
    d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' })
  const a = at(c.cohort_start)
  const b = at(c.cohort_end)
  if (c.cohort_start === c.cohort_end) return `${day(a)} ${mon(a)}`
  if (mon(a) === mon(b)) return `${day(a)}\u2013${day(b)} ${mon(b)}`
  return `${day(a)} ${mon(a)} \u2013 ${day(b)} ${mon(b)}`
}

/** How many days behind today the newest cohort is. Null if unknown. */
export function daysBehind(c?: FunnelCoverage): number | null {
  if (!c?.cohort_end) return null
  const end = new Date(`${c.cohort_end}T12:00:00Z`).getTime()
  const today = new Date(`${new Date().toISOString().slice(0, 10)}T12:00:00Z`).getTime()
  return Math.max(0, Math.round((today - end) / 86_400_000))
}

export function formatDuration(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  const s = ms / 1000
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)}s`
  const m = s / 60
  if (m < 60) return `${m < 10 ? m.toFixed(1) : Math.round(m)}m`
  const h = m / 60
  if (h < 24) return `${h < 10 ? h.toFixed(1) : Math.round(h)}h`
  return `${(h / 24).toFixed(1)}d`
}
