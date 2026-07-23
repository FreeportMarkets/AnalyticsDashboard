import type { neon } from '@neondatabase/serverless'

type SqlTag = ReturnType<typeof neon<boolean, boolean>>

/**
 * Where the list of Freeport wallets to scan comes from. Two implementations,
 * because enumeration completeness is the entire ballgame (see
 * docs/volume-tracking.md):
 *
 *   - `privyRegistrySource` — every embedded EVM wallet Privy knows about.
 *     This is the authoritative registry: every user authenticates through
 *     Privy, so this is the complete set. Reconciliation ties to <1% only
 *     with this source. Requires PRIVY_APP_ID / PRIVY_APP_SECRET.
 *   - `analyticsWalletSource` — wallets already present in our Neon
 *     trades ∪ events. A bootstrap/fallback that reaches only ~half the
 *     fees, because the trades table is itself incomplete (that's the bug
 *     this whole system routes around). Use only when Privy creds are absent.
 */

export interface WalletSource {
  readonly name: string
  list(): Promise<string[]>
}

// --- Privy registry (authoritative) ---

const PRIVY_USERS_URL = 'https://auth.privy.io/api/v1/users'
const PRIVY_PAGE_LIMIT = 100
const PRIVY_PAGE_DELAY_MS = Number(process.env.PRIVY_PAGE_DELAY_MS ?? 300)
const PRIVY_MAX_RETRIES = 8

interface PrivyLinkedAccount {
  type?: string
  address?: string
  chain_type?: string
  wallet_client_type?: string
}
interface PrivyUser {
  linked_accounts?: PrivyLinkedAccount[]
}

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms))

/**
 * The embedded Privy ethereum wallet is the one Freeport trades through
 * (`wallet_client_type === 'privy'`, `chain_type === 'ethereum'`). External
 * linked wallets (metamask, phantom) are the user's own and generally do not
 * trade through us; including them only adds empty HL lookups. We take the
 * embedded set and lowercase-dedupe.
 */
export function embeddedEvmWallets(users: PrivyUser[]): string[] {
  const out = new Set<string>()
  for (const u of users) {
    for (const a of u.linked_accounts ?? []) {
      if (
        a.type === 'wallet' &&
        a.chain_type === 'ethereum' &&
        a.wallet_client_type === 'privy' &&
        a.address
      ) {
        out.add(a.address.toLowerCase())
      }
    }
  }
  return [...out]
}

export function privyRegistrySource(
  appId = process.env.PRIVY_APP_ID,
  appSecret = process.env.PRIVY_APP_SECRET,
  fetchImpl: typeof fetch = fetch
): WalletSource {
  return {
    name: 'privy-registry',
    async list() {
      if (!appId || !appSecret) throw new Error('PRIVY_APP_ID / PRIVY_APP_SECRET not set')
      const auth = `Basic ${Buffer.from(`${appId}:${appSecret}`).toString('base64')}`
      const wallets = new Set<string>()
      let cursor: string | undefined

      do {
        const url = new URL(PRIVY_USERS_URL)
        url.searchParams.set('limit', String(PRIVY_PAGE_LIMIT))
        if (cursor) url.searchParams.set('cursor', cursor)

        let res: Response | undefined
        for (let attempt = 0; attempt < PRIVY_MAX_RETRIES; attempt++) {
          res = await fetchImpl(url, { headers: { Authorization: auth, 'privy-app-id': appId } })
          if (res.ok) break
          if (res.status !== 429 && res.status < 500) {
            throw new Error(`privy GET /users ${res.status}: ${await res.text()}`)
          }
          const retryAfter = Number(res.headers.get('retry-after'))
          await sleep(retryAfter > 0 ? retryAfter * 1000 : Math.min(60_000, 2_000 * 2 ** attempt))
        }
        if (!res?.ok) throw new Error(`privy GET /users failed after ${PRIVY_MAX_RETRIES} retries`)

        const body = (await res.json()) as { data?: PrivyUser[]; next_cursor?: string | null }
        for (const w of embeddedEvmWallets(body.data ?? [])) wallets.add(w)
        cursor = body.next_cursor ?? undefined
        if (cursor) await sleep(PRIVY_PAGE_DELAY_MS)
      } while (cursor)

      return [...wallets]
    },
  }
}

// --- Analytics DB (bootstrap / fallback) ---

export function analyticsWalletSource(sql: SqlTag): WalletSource {
  return {
    name: 'analytics-db',
    async list() {
      const rows = (await sql`
        SELECT DISTINCT lower(wallet_address) AS a FROM (
          SELECT wallet_address FROM trades WHERE wallet_address ~* '^0x[0-9a-f]{40}$'
          UNION
          SELECT wallet_address FROM events WHERE wallet_address ~* '^0x[0-9a-f]{40}$'
        ) u
      `) as Array<{ a: string }>
      return rows.map(r => r.a)
    },
  }
}

/** Pick the best available source: Privy registry if creds exist, else analytics DB. */
export function defaultWalletSource(sql: SqlTag): WalletSource {
  if (process.env.PRIVY_APP_ID && process.env.PRIVY_APP_SECRET) return privyRegistrySource()
  return analyticsWalletSource(sql)
}
