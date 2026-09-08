import { unstable_cache } from 'next/cache'

export interface JourneyAccount {
  id: string
  createdAt: string
  wallets: string[]
}

/** EVM addresses are case insensitive; Solana addresses are not. */
export const normalizeWallet = (wallet: string) => /^0x[0-9a-f]+$/i.test(wallet) ? wallet.toLowerCase() : wallet

/**
 * Account counts must include walletless users and deduplicate by Privy DID.
 * The identity-label mirror lacks created_at and updates only nightly.
 * This separate, persistently cached snapshot keeps "today" current without
 * re-listing users on each navigation or changing identity enrichment.
 * Contract: https://docs.privy.io/api-reference/users/get-all
 */
export async function loadJourneyAccounts(): Promise<{ accounts: JourneyAccount[]; fetchedAt: string }> {
  const appId = process.env.PRIVY_APP_ID
  const secret = process.env.PRIVY_APP_SECRET
  if (!appId || !secret) throw new Error('Signup source is not configured')
  const fetchedAt = new Date().toISOString()
  const signal = AbortSignal.timeout(45_000)
  const accounts = new Map<string, JourneyAccount>()
  const cursors = new Set<string>()
  let cursor: string | undefined
  for (let page = 0; page < 200; page++) {
    const url = new URL('https://api.privy.io/v1/users')
    url.searchParams.set('limit', '100')
    if (cursor) url.searchParams.set('cursor', cursor)
    const response = await fetch(url, {
      headers: { Authorization: `Basic ${Buffer.from(`${appId}:${secret}`).toString('base64')}`, 'privy-app-id': appId },
      cache: 'no-store', signal,
    })
    if (!response.ok) throw new Error('Signup source could not be read completely')
    const body = await response.json() as { data?: Record<string, unknown>[]; next_cursor?: string | null }
    if (!Array.isArray(body.data)) throw new Error('Invalid signup source response')
    for (const user of body.data) {
      if (user.is_guest === true) continue
      if (typeof user.id !== 'string' || !user.id.startsWith('did:privy:') ||
          typeof user.created_at !== 'number' || !Number.isFinite(user.created_at) ||
          user.created_at <= 0 || !Array.isArray(user.linked_accounts)) {
        throw new Error('Signup source contains an invalid account')
      }
      accounts.set(user.id, {
        id: user.id,
        createdAt: new Date(user.created_at * 1000).toISOString(),
        wallets: [...new Set(user.linked_accounts.flatMap((linked: Record<string, unknown>) =>
          linked.type === 'wallet' && typeof linked.address === 'string' && linked.address
            ? [normalizeWallet(linked.address)] : [],
        ))],
      })
    }
    if (body.next_cursor === null || body.next_cursor === undefined) return { accounts: [...accounts.values()], fetchedAt }
    if (typeof body.next_cursor !== 'string' || !body.next_cursor || cursors.has(body.next_cursor)) {
      throw new Error('Signup pagination did not complete')
    }
    cursor = body.next_cursor
    cursors.add(cursor)
  }
  // Never publish a partial listing as a complete signup count.
  throw new Error('Signup pagination limit reached')
}

export const fetchJourneyAccounts = unstable_cache(loadJourneyAccounts, ['journey-accounts-v1'], { revalidate: 300 })
