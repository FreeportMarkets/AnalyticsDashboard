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

export interface FunnelResponse {
  window: '24h' | '7d'
  segment: { key: string; value: string }
  days: number
  steps: FunnelStep[]
}

export type FunnelResult =
  | { ok: true; data: FunnelResponse }
  | { ok: false; reason: 'not_configured' | 'unauthorized' | 'unreachable' }

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

  try {
    const res = await fetch(`${BASE}/v1/analytics/funnel?${params.toString()}`, {
      headers: { 'x-funnel-secret': secret },
      // The rollup only moves every 15 minutes, so a short cache spares the
      // API on refreshes without ever showing a stale-by-hours number.
      next: { revalidate: 60 },
    })
    if (res.status === 401) return { ok: false, reason: 'unauthorized' }
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
  auth_succeeded: 'Made an account',
  deposit_opened: 'Opened deposit',
  deposit_method_selected: 'Chose a method',
  deposit_started: 'Started paying',
  deposit_completed: 'Paid',
  deposit_funds_arrived: 'Money landed',
  trade_submitted: 'Placed a trade',
  trade_succeeded: 'Trade filled',
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
