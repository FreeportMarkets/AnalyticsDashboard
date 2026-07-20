/**
 * Privy wallet -> identity enrichment.
 *
 * Ported from the working Streamlit dashboard (`app.py`, repo root):
 *   - `_privy_credentials`            (~786-791)
 *   - `_extract_privy_identity`       (~793-840)
 *   - `load_privy_users`              (~841-890)
 *   - `label_wallet` / `enrich_wallet_df` (~892-917)
 *   - `_did_lookup_from_privy_map` / `_fetch_privy_user_by_did` (~919-966)
 *
 * `_did_lookup_from_privy_map` / `_fetch_privy_user_by_did` were added in
 * commit dd67951 as a per-DID backstop for `load_privy_users`' bulk listing,
 * which Privy rate-limits past ~30 pages (~3000 users) under sustained
 * request rate -- newer accounts past that window would otherwise render as
 * bare DIDs/addresses forever. Ported here (`fetchPrivyUserByDid`) for
 * completeness and for any future DID-keyed table (referrals, redemptions,
 * etc. -- see `enrich_did_df` in app.py); today's callers (trades/users
 * pages) only have wallet addresses, not DIDs, so they only exercise the
 * bulk wallet-keyed path.
 *
 * FAILS SOFT EVERYWHERE: an unreachable/unauthorized/slow Privy API must
 * never take the dashboard down. Every network call is timeout-guarded and
 * wrapped so failures collapse to an empty map / null, and callers fall
 * back to a truncated wallet address.
 */

const PRIVY_API_BASE = 'https://auth.privy.io/api/v1'
const PAGE_SIZE = 100
// Safety cap on pagination, mirrors app.py's `pages > 200` bound (~20k users).
const MAX_PAGES = 200
// Matches Streamlit's `@st.cache_data(ttl=3600)` on `load_privy_users`.
const BULK_CACHE_TTL_MS = 60 * 60 * 1000
// Matches Streamlit's `@st.cache_data(ttl=86400)` on `_fetch_privy_user_by_did`.
const DID_CACHE_TTL_MS = 24 * 60 * 60 * 1000
const FETCH_TIMEOUT_MS = 20_000
const DID_FETCH_TIMEOUT_MS = 10_000

export type LoginType = 'google' | 'email' | 'apple' | 'twitter' | 'phone' | 'wallet_only'

export interface PrivyIdentity {
  privyDid: string
  label: string | null
  loginType: LoginType
  contact: string | null
  email: string | null
}

/** Keyed by LOWERCASED wallet address. A Privy user can own multiple wallets -- every one is indexed. */
export type PrivyWalletMap = Map<string, PrivyIdentity>

function credentials(): { appId: string; secret: string } | null {
  const appId = process.env.PRIVY_APP_ID
  const secret = process.env.PRIVY_APP_SECRET
  if (!appId || !secret) return null
  return { appId, secret }
}

/**
 * Privy REST auth: HTTP Basic app_id:app_secret, plus a required
 * `privy-app-id` header (mirrors app.py's `auth=(app_id, secret)` +
 * `headers={"privy-app-id": app_id}`).
 */
function authHeaders(appId: string, secret: string): HeadersInit {
  const basic = Buffer.from(`${appId}:${secret}`, 'utf8').toString('base64')
  return {
    Authorization: `Basic ${basic}`,
    'privy-app-id': appId,
  }
}

/** GET + parse JSON. Fails soft to null on any non-200, timeout, or network error. */
async function fetchJson(
  url: string,
  appId: string,
  secret: string,
  timeoutMs: number
): Promise<unknown | null> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const resp = await fetch(url, { headers: authHeaders(appId, secret), signal: controller.signal })
    if (!resp.ok) return null
    return await resp.json()
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}

interface ExtractedIdentity {
  identity: PrivyIdentity
  /** Lowercased linked wallet addresses for this user. */
  wallets: string[]
}

/**
 * Pick the best human-readable label out of a Privy user's linked_accounts.
 * Precedence ported verbatim from app.py's `_extract_privy_identity`
 * (~793-838): google_name -> google_email -> email -> apple_email ->
 * twitter -> phone -> wallet_only (no label; caller falls back to a
 * truncated address).
 */
function extractIdentity(user: Record<string, unknown>): ExtractedIdentity {
  const accounts = Array.isArray(user.linked_accounts) ? (user.linked_accounts as Record<string, unknown>[]) : []

  let email: string | null = null
  let googleEmail: string | null = null
  let googleName: string | null = null
  let appleEmail: string | null = null
  let phone: string | null = null
  let twitter: string | null = null
  const wallets: string[] = []

  for (const acc of accounts) {
    const t = acc.type
    if (t === 'email' && !email) {
      email = typeof acc.address === 'string' ? acc.address : null
    } else if (t === 'google_oauth') {
      googleEmail = googleEmail ?? (typeof acc.email === 'string' ? acc.email : null)
      googleName = googleName ?? (typeof acc.name === 'string' ? acc.name : null)
    } else if (t === 'apple_oauth' && !appleEmail) {
      appleEmail = typeof acc.email === 'string' ? acc.email : null
    } else if (t === 'phone' && !phone) {
      phone = typeof acc.number === 'string' ? acc.number : null
    } else if (t === 'twitter_oauth' && !twitter) {
      twitter = typeof acc.username === 'string' ? acc.username : null
    } else if (t === 'wallet') {
      const addr = acc.address
      if (typeof addr === 'string' && addr) wallets.push(addr.toLowerCase())
    }
  }

  let label: string | null
  let loginType: LoginType
  let contact: string | null

  if (googleName) {
    label = googleName
    loginType = 'google'
    contact = googleEmail
  } else if (googleEmail) {
    label = googleEmail
    loginType = 'google'
    contact = googleEmail
  } else if (email) {
    label = email
    loginType = 'email'
    contact = email
  } else if (appleEmail) {
    label = appleEmail
    loginType = 'apple'
    contact = appleEmail
  } else if (twitter) {
    label = `@${twitter}`
    loginType = 'twitter'
    contact = null
  } else if (phone) {
    label = phone
    loginType = 'phone'
    contact = phone
  } else {
    label = null
    loginType = 'wallet_only'
    contact = null
  }

  return {
    identity: {
      privyDid: typeof user.id === 'string' ? user.id : '',
      label,
      loginType,
      contact,
      email: email ?? googleEmail ?? appleEmail ?? null,
    },
    wallets,
  }
}

let bulkCache: { map: PrivyWalletMap; expiresAt: number } | null = null
let bulkInflight: Promise<PrivyWalletMap> | null = null

/**
 * Paginate GET /users, building wallet_lower -> identity. Cached in module
 * scope for `BULK_CACHE_TTL_MS` (1h) so Server Components rendering per
 * request don't hammer Privy. Never throws -- any failure (not configured,
 * non-200, timeout, network error) resolves to an empty (or partial) map.
 */
export async function fetchPrivyUsers(): Promise<PrivyWalletMap> {
  const now = Date.now()
  if (bulkCache && bulkCache.expiresAt > now) return bulkCache.map
  if (bulkInflight) return bulkInflight

  bulkInflight = loadPrivyUsers()
    .then(map => {
      bulkCache = { map, expiresAt: Date.now() + BULK_CACHE_TTL_MS }
      return map
    })
    .catch(() => new Map<string, PrivyIdentity>())
    .finally(() => {
      bulkInflight = null
    })

  return bulkInflight
}

async function loadPrivyUsers(): Promise<PrivyWalletMap> {
  const map: PrivyWalletMap = new Map()
  const creds = credentials()
  if (!creds) return map

  let cursor: string | undefined
  let pages = 0

  try {
    while (pages < MAX_PAGES) {
      const url = new URL(`${PRIVY_API_BASE}/users`)
      url.searchParams.set('limit', String(PAGE_SIZE))
      if (cursor) url.searchParams.set('cursor', cursor)

      const body = (await fetchJson(url.toString(), creds.appId, creds.secret, FETCH_TIMEOUT_MS)) as {
        data?: Record<string, unknown>[]
        next_cursor?: string
      } | null
      if (!body) break // request failed / non-200 -- stop, return whatever we've built

      const pageUsers = Array.isArray(body.data) ? body.data : []
      for (const u of pageUsers) {
        const { identity, wallets } = extractIdentity(u)
        for (const w of wallets) map.set(w, identity)
      }

      cursor = body.next_cursor
      pages += 1
      if (!cursor) break
    }
  } catch {
    // Fail soft -- return whatever partial map was built before the error.
  }

  return map
}

/** Truncated address fallback, e.g. `0x1234…abcd`. Matches the existing web convention. */
export function shortWallet(addr: string): string {
  if (!addr) return '?'
  return addr.length > 10 ? `${addr.slice(0, 4)}…${addr.slice(-4)}` : addr
}

/** Human label for a wallet. Prefers the Privy identity, falls back to a truncated address. */
export function labelForWallet(addr: string, map: PrivyWalletMap): string {
  if (!addr) return '?'
  const info = map.get(addr.toLowerCase())
  if (info?.label) return info.label
  return shortWallet(addr)
}

export interface DidIdentity {
  label: string
  email: string
  loginType: LoginType | ''
}

/**
 * DID-keyed lookup, derived lazily from the wallet-keyed bulk map. The same
 * Privy DID can have multiple linked wallets, so dedupe on the way in.
 * Ported from `_did_lookup_from_privy_map` (app.py ~919-933).
 */
export function didLookupFromPrivyMap(map: PrivyWalletMap): Map<string, DidIdentity> {
  const out = new Map<string, DidIdentity>()
  if (!map) return out
  for (const ident of map.values()) {
    const did = ident.privyDid
    if (!did || out.has(did)) continue
    out.set(did, {
      label: ident.label ?? '',
      email: ident.email ?? '',
      loginType: ident.loginType ?? '',
    })
  }
  return out
}

const didCache = new Map<string, { identity: PrivyIdentity | null; expiresAt: number }>()

/**
 * Per-DID identity fetch (GET /users/{did}), cached individually for
 * `DID_CACHE_TTL_MS` (24h) -- the backstop for users past the bulk listing's
 * pagination cap. Ported from `_fetch_privy_user_by_did` (app.py ~946-966).
 * Fails soft to null on any error, bad input, or missing credentials.
 */
export async function fetchPrivyUserByDid(did: string): Promise<PrivyIdentity | null> {
  if (!did || !did.startsWith('did:privy:')) return null

  const cached = didCache.get(did)
  if (cached && cached.expiresAt > Date.now()) return cached.identity

  const creds = credentials()
  if (!creds) return null

  const body = await fetchJson(
    `${PRIVY_API_BASE}/users/${encodeURIComponent(did)}`,
    creds.appId,
    creds.secret,
    DID_FETCH_TIMEOUT_MS
  )
  const identity = body ? extractIdentity(body as Record<string, unknown>).identity : null
  didCache.set(did, { identity, expiresAt: Date.now() + DID_CACHE_TTL_MS })
  return identity
}
