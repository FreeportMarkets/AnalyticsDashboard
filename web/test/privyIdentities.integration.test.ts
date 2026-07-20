import { describe, it, expect, vi, afterEach } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { neon } from '@neondatabase/serverless'
import { fetchWalletIdentities, fetchIdentitiesByDid } from '@/lib/privyIdentities'

/**
 * These tests run against the REAL Neon test database (not a mock) -- same
 * rationale as test/integration.test.ts: privy_identities' read path
 * (fetchWalletIdentities/fetchIdentitiesByDid) is the fix for the 14.4s
 * Privy-in-render-path bug, and refreshPrivyIdentities' UNNEST upsert is
 * exactly the kind of SQL no mock can stand in for.
 *
 * `TEST_DATABASE_URL` is read directly out of `.env.local`, matching every
 * other integration suite in this project. Skipped gracefully (not failed)
 * when absent.
 */
function readEnvLocal(key: string): string | undefined {
  let content: string
  try {
    content = readFileSync(path.join(import.meta.dirname, '..', '.env.local'), 'utf8')
  } catch {
    return undefined
  }
  for (const line of content.split('\n')) {
    const match = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/)
    if (match && match[1] === key) {
      return match[2]!.trim().replace(/^"(.*)"$/, '$1')
    }
  }
  return undefined
}

const TEST_DATABASE_URL = readEnvLocal('TEST_DATABASE_URL')

// Unique per test-file run so concurrent/rerun invocations never collide.
// Wallets are lowercased hex-ish strings so they round-trip through the
// same lowercasing `fetchWalletIdentities` applies to its input.
const MARKER = `itest_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`.toLowerCase()

describe.skipIf(!TEST_DATABASE_URL)('privyIdentities (real Neon test database)', () => {
  const sql = neon<boolean, boolean>(TEST_DATABASE_URL!)

  afterEach(async () => {
    await sql`DELETE FROM privy_identities WHERE wallet_address LIKE ${MARKER + '%'}`
  })

  describe('fetchWalletIdentities', () => {
    it('reads only the requested wallets, lowercases input, and labels the rest via fallback (empty map entry)', async () => {
      const walletLabeled = `${MARKER}-wallet-1`
      const walletUnlabeled = `${MARKER}-wallet-2`
      const walletNotOnPage = `${MARKER}-wallet-3`

      await sql`
        INSERT INTO privy_identities (wallet_address, did, label, login_type, contact)
        VALUES
          (${walletLabeled}, ${'did:privy:' + MARKER + '-a'}, 'Test User One', 'email', 'one@example.com'),
          (${walletUnlabeled}, ${'did:privy:' + MARKER + '-b'}, NULL, 'wallet_only', NULL),
          (${walletNotOnPage}, ${'did:privy:' + MARKER + '-c'}, 'Should not be fetched', 'email', 'x@example.com')
      `

      // Request labeled + unlabeled + a wallet that has NO row at all, with
      // mixed casing to prove the lowercasing behavior. walletNotOnPage is
      // deliberately excluded from the request.
      const map = await fetchWalletIdentities(sql, [
        walletLabeled.toUpperCase(),
        walletUnlabeled,
        `${MARKER}-wallet-missing`,
      ])

      expect(map.get(walletLabeled)?.label).toBe('Test User One')
      expect(map.get(walletLabeled)?.privyDid).toBe(`did:privy:${MARKER}-a`)
      expect(map.get(walletUnlabeled)?.label).toBeNull()
      expect(map.has(`${MARKER}-wallet-missing`)).toBe(false)
      // The page-scoping is the whole point of this function -- a wallet
      // that exists in the table but wasn't requested must never appear.
      expect(map.has(walletNotOnPage)).toBe(false)
    })

    it('returns an empty map (fails soft) for an empty wallet list without querying', async () => {
      const map = await fetchWalletIdentities(sql, [])
      expect(map.size).toBe(0)
    })
  })

  describe('fetchIdentitiesByDid', () => {
    it('reads only the requested DIDs via the did index', async () => {
      const did1 = `did:privy:${MARKER}-referrer-1`
      const did2 = `did:privy:${MARKER}-referrer-2`

      await sql`
        INSERT INTO privy_identities (wallet_address, did, label, login_type, contact)
        VALUES (${`${MARKER}-wallet-did-1`}, ${did1}, 'Referrer One', 'google', 'ref1@example.com')
      `
      await sql`
        INSERT INTO privy_identities (wallet_address, did, label, login_type, contact)
        VALUES (${`${MARKER}-wallet-did-2`}, ${did2}, NULL, 'wallet_only', NULL)
      `

      const map = await fetchIdentitiesByDid(sql, [did1, `did:privy:${MARKER}-not-in-table`])

      expect(map.get(did1)).toEqual({ label: 'Referrer One', email: 'ref1@example.com', loginType: 'google' })
      expect(map.has(did2)).toBe(false) // not requested
      expect(map.has(`did:privy:${MARKER}-not-in-table`)).toBe(false)
    })

    it('ignores non-did-shaped input and returns an empty map for an empty list', async () => {
      const map = await fetchIdentitiesByDid(sql, ['not-a-did', ''])
      expect(map.size).toBe(0)
    })
  })

  describe('refreshPrivyIdentities', () => {
    const originalFetch = globalThis.fetch
    const originalAppId = process.env.PRIVY_APP_ID
    const originalSecret = process.env.PRIVY_APP_SECRET

    afterEach(() => {
      vi.unstubAllGlobals()
      globalThis.fetch = originalFetch
      process.env.PRIVY_APP_ID = originalAppId
      process.env.PRIVY_APP_SECRET = originalSecret
    })

    it('upserts a mocked Privy bulk listing into privy_identities, then updates on re-run', async () => {
      process.env.PRIVY_APP_ID = 'test-app-id'
      process.env.PRIVY_APP_SECRET = 'test-app-secret'

      const wallet = `${MARKER}-refresh-wallet-1`
      const did = `did:privy:${MARKER}-refresh-1`

      // Mutated below between the two refreshPrivyIdentities() calls -- the
      // mock closure reads this each time it's invoked. Shaped loosely
      // (matching what extractIdentity() in privy.ts actually reads off
      // `linked_accounts`) since the two pages use different account types.
      let mockPage: { data: Array<{ id: string; linked_accounts: Record<string, unknown>[] }>; next_cursor?: string } = {
        data: [
          {
            id: did,
            linked_accounts: [
              { type: 'wallet', address: wallet.toUpperCase() },
              { type: 'email', address: 'refresh1@example.com' },
            ],
          },
        ],
        next_cursor: undefined,
      }

      // `globalThis.fetch` is ALSO how the Neon HTTP driver talks to
      // Postgres -- a blanket mock would intercept the test's own `sql`
      // calls, not just the Privy API call inside refreshPrivyIdentities.
      // Only intercept requests to the Privy API host; pass everything else
      // (Neon's driver) through to the real fetch.
      const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input.toString()
        if (url.startsWith('https://auth.privy.io/')) {
          return Promise.resolve({ ok: true, json: async () => mockPage } as Response)
        }
        return originalFetch(input, init)
      })
      vi.stubGlobal('fetch', fetchMock)

      // `fetchPrivyUsers()` (which refreshPrivyIdentities calls) caches its
      // result at module scope for an hour -- fine in production (a cron
      // invocation is a fresh container), but this test calls
      // refreshPrivyIdentities twice with two different mocked pages in the
      // same process, so each call needs a genuinely fresh module instance
      // or the second call would silently serve the first call's cached map.
      vi.resetModules()
      const { refreshPrivyIdentities: refresh1 } = await import('@/lib/privyIdentities')
      const result = await refresh1(sql)
      expect(result.fetchedWallets).toBeGreaterThanOrEqual(1)
      expect(result.upserted).toBeGreaterThanOrEqual(1)

      const rows = (await sql`
        SELECT wallet_address, did, label, login_type, contact FROM privy_identities WHERE wallet_address = ${wallet}
      `) as Array<{ wallet_address: string; did: string; label: string; login_type: string; contact: string }>
      expect(rows).toHaveLength(1)
      expect(rows[0]).toMatchObject({
        wallet_address: wallet,
        did,
        label: 'refresh1@example.com',
        login_type: 'email',
        contact: 'refresh1@example.com',
      })

      // Re-run with an updated label -- ON CONFLICT (wallet_address) DO
      // UPDATE must overwrite, not duplicate the row.
      mockPage = {
        data: [
          {
            id: did,
            linked_accounts: [
              { type: 'wallet', address: wallet },
              { type: 'google_oauth', name: 'Updated Name', email: 'refresh1@example.com' },
            ],
          },
        ],
        next_cursor: undefined,
      }

      vi.resetModules()
      const { refreshPrivyIdentities: refresh2 } = await import('@/lib/privyIdentities')
      await refresh2(sql)

      const updated = (await sql`
        SELECT wallet_address, label, login_type FROM privy_identities WHERE wallet_address = ${wallet}
      `) as Array<{ wallet_address: string; label: string; login_type: string }>
      expect(updated).toHaveLength(1)
      expect(updated[0]).toMatchObject({ wallet_address: wallet, label: 'Updated Name', login_type: 'google' })
    })
  })
})
