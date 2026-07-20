/**
 * Points admin API client -- promo codes + personal-referral leaderboard.
 *
 * Ported from the working Streamlit dashboard (`app.py`, repo root,
 * ~1165-1337): `_admin_key` / `_admin_headers`, `fetch_promo_codes`,
 * `fetch_promo_code_detail`, `create_promo_code`, `disable_promo_code`,
 * `attach_promo_beneficiary`, `fetch_top_referrers`.
 *
 * SERVER-ONLY. `ADMIN_API_KEY` must never reach the client -- every export
 * here is called exclusively from Server Components (page.tsx) or Server
 * Actions (actions.ts), never from 'use client' code.
 *
 * Unlike privy.ts, the READ helpers here do NOT fail soft to an empty
 * array/null on network or auth failure for the *caller* to silently show
 * an empty state without explanation -- they throw, and the page renders
 * the message. This route mutates production money-adjacent state, so a
 * misconfigured key or a downed backend must be loud, not swallowed into
 * "no promo codes have been created yet."
 */

const POINTS_API_BASE = 'https://trading-api.freeportmarkets.com/v1/points'
const READ_TIMEOUT_MS = 10_000
const WRITE_TIMEOUT_MS = 15_000

export class PointsApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number
  ) {
    super(message)
    this.name = 'PointsApiError'
  }
}

function adminKey(): string | null {
  return process.env.ADMIN_API_KEY || null
}

function adminHeaders(): HeadersInit {
  const key = adminKey()
  const h: Record<string, string> = { 'content-type': 'application/json' }
  if (key) h['x-admin-key'] = key
  // Purely audit metadata on the backend -- flags this caller as the
  // dashboard so disabled_by / created_by columns are honest. Mirrors
  // app.py's `_admin_headers`.
  h['x-admin-id'] = 'analytics-dashboard-web'
  return h
}

async function req<T>(
  path: string,
  init: { method?: string; body?: unknown; params?: Record<string, string | number>; timeoutMs: number }
): Promise<T> {
  const key = adminKey()
  if (!key) {
    throw new PointsApiError(
      'ADMIN_API_KEY is not configured on the server (web/.env.local or Vercel env).'
    )
  }

  const url = new URL(`${POINTS_API_BASE}${path}`)
  if (init.params) {
    for (const [k, v] of Object.entries(init.params)) url.searchParams.set(k, String(v))
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), init.timeoutMs)
  let resp: Response
  try {
    resp = await fetch(url.toString(), {
      method: init.method ?? 'GET',
      headers: adminHeaders(),
      body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
      signal: controller.signal,
      cache: 'no-store',
    })
  } catch (e) {
    throw new PointsApiError(`Network error: ${e instanceof Error ? e.message : String(e)}`)
  } finally {
    clearTimeout(timer)
  }

  const contentType = resp.headers.get('content-type') ?? ''
  const isJson = contentType.startsWith('application/json')
  const body = isJson ? await resp.json().catch(() => null) : null

  if (!resp.ok) {
    const errMsg = (body && typeof body === 'object' && 'error' in body && typeof body.error === 'string')
      ? body.error
      : (await safeText(resp, body))
    throw new PointsApiError(`HTTP ${resp.status}: ${errMsg}`, resp.status)
  }

  return (body ?? {}) as T
}

async function safeText(resp: Response, alreadyParsedJson: unknown): Promise<string> {
  if (alreadyParsedJson !== null) return JSON.stringify(alreadyParsedJson).slice(0, 200)
  try {
    return (await resp.text()).slice(0, 200)
  } catch {
    return resp.statusText || 'unknown error'
  }
}

// --- Types (backend response shapes -- fields as observed live) ---------

export interface RewardValue {
  amount?: string
  multiplier?: string
  duration_hours?: number
  tier?: string
  [key: string]: unknown
}

export interface PromoCode {
  code: string
  status: string
  reward_kind: string
  reward_value: RewardValue | null
  max_redemptions: number | null
  current_redemptions: number
  beneficiary_did: string | null
  kickback_rate: string | null
  champion_redeem_bonus: number | null
  kickback_cap_per_referee: number | null
  expires_at: string | null
  created_by: string | null
  created_at: string | null
  disabled_at: string | null
  disabled_reason: string | null
  [key: string]: unknown
}

/** Shape observed live from GET /admin/promo/:code -- redemption rows. */
export interface PromoRedemption {
  id?: string
  privy_did?: string
  redeemed_at?: string
  reward_kind?: string
  reward_value?: RewardValue | null
  points_ledger_id?: string | null
  active_boost_id?: string | null
  chest_id?: string | null
  trace_id?: string | null
  [key: string]: unknown
}

export interface PromoCodeDetail extends PromoCode {
  redemptions?: PromoRedemption[]
  [key: string]: unknown
}

export interface TopReferrer {
  code: string
  referrer_did: string
  code_created_at?: string
  referee_count: number
  first_referral_at?: string | null
  most_recent_referral_at?: string | null
  lifetime_kickback_points?: string | number
  kickback_event_count?: number
  [key: string]: unknown
}

// --- Reads ----------------------------------------------------------------

/** GET /admin/promo -- all promo codes with the projected dynamic status. */
export async function fetchPromoCodes(): Promise<PromoCode[]> {
  const body = await req<{ codes?: PromoCode[] }>('/admin/promo', { timeoutMs: READ_TIMEOUT_MS })
  return body.codes ?? []
}

/** GET /admin/promo/:code -- detail + redemption rows for one code. */
export async function fetchPromoCodeDetail(code: string): Promise<PromoCodeDetail | null> {
  if (!code) return null
  return req<PromoCodeDetail>(`/admin/promo/${encodeURIComponent(code)}`, { timeoutMs: READ_TIMEOUT_MS })
}

/** GET /admin/personal-codes/top -- top users by referee count. */
export async function fetchTopReferrers(limit = 50): Promise<TopReferrer[]> {
  const body = await req<{ top_referrers?: TopReferrer[] }>('/admin/personal-codes/top', {
    params: { limit },
    timeoutMs: READ_TIMEOUT_MS,
  })
  return body.top_referrers ?? []
}

// --- Writes (called only from actions.ts Server Actions) ------------------

export interface CreatePromoPayload {
  code: string
  reward_kind: 'points' | 'multiplier_boost' | 'chest_grant'
  reward_value: RewardValue
  max_redemptions?: number
  expires_at?: string
  kickback_rate?: string
  champion_redeem_bonus?: number
  kickback_cap_per_referee?: number
}

/** POST /admin/promo. Payload shape mirrors app.py's CreatePromoBody. */
export async function createPromoCode(payload: CreatePromoPayload): Promise<{ code: string }> {
  return req<{ code: string }>('/admin/promo', { method: 'POST', body: payload, timeoutMs: WRITE_TIMEOUT_MS })
}

/** POST /admin/promo/:code/disable. Reason is required by the backend. */
export async function disablePromoCode(code: string, reason: string): Promise<void> {
  await req<unknown>(`/admin/promo/${encodeURIComponent(code)}/disable`, {
    method: 'POST',
    body: { reason },
    timeoutMs: WRITE_TIMEOUT_MS,
  })
}

export interface AttachBeneficiaryInput {
  did?: string | null
  wallet?: string | null
  force: boolean
}

/** POST /admin/promo/:code/beneficiary. Either did or wallet must be set. */
export async function attachPromoBeneficiary(code: string, input: AttachBeneficiaryInput): Promise<void> {
  const payload: Record<string, unknown> = { force: input.force }
  if (input.did) {
    payload.beneficiary_did = input.did
  } else if (input.wallet) {
    payload.beneficiary_wallet = input.wallet
  } else {
    throw new PointsApiError('Provide either a DID or a wallet address.')
  }
  await req<unknown>(`/admin/promo/${encodeURIComponent(code)}/beneficiary`, {
    method: 'POST',
    body: payload,
    timeoutMs: WRITE_TIMEOUT_MS,
  })
}
