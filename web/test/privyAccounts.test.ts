import { afterEach, describe, expect, it, vi } from 'vitest'
import { loadJourneyAccounts } from '@/lib/privyAccounts'

vi.mock('next/cache', () => ({ unstable_cache: (fn: unknown) => fn }))
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals() })
const user = (id: string, wallets: string[] = []) => ({ id: `did:privy:${id}`, created_at: 1788840000, linked_accounts: wallets.map(address => ({ type: 'wallet', address })) })
function setup(pages: unknown[]) {
  vi.stubEnv('PRIVY_APP_ID', 'test-app')
  vi.stubEnv('PRIVY_APP_SECRET', 'test-secret')
  const fetch = vi.fn()
  for (const page of pages) fetch.mockResolvedValueOnce({ ok: true, json: async () => page })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
describe('Daily signup source', () => {
  it('includes walletless accounts, merges duplicate IDs and excludes guests', async () => {
    const fetch = setup([
      { data: [user('one'), { ...user('guest'), is_guest: true }], next_cursor: 'second' },
      { data: [user('one'), user('two', ['0xAB', '0xab', 'SolABC'])], next_cursor: null },
    ])
    const result = await loadJourneyAccounts()
    expect(result.accounts).toHaveLength(2)
    expect(result.accounts[0]?.wallets).toEqual([])
    expect(result.accounts[1]?.wallets).toEqual(['0xab', 'SolABC'])
    expect(new URL(String(fetch.mock.calls[1]?.[0])).searchParams.get('cursor')).toBe('second')
  })
  it('rejects a partial listing when a subsequent page fails', async () => {
    const fetch = setup([{ data: [user('one')], next_cursor: 'second' }])
    fetch.mockResolvedValueOnce({ ok: false })
    await expect(loadJourneyAccounts()).rejects.toThrow('completely')
  })
  it('rejects malformed timestamps and repeated cursors', async () => {
    setup([{ data: [{ ...user('bad'), created_at: 'today' }] }])
    await expect(loadJourneyAccounts()).rejects.toThrow('invalid account')
    setup([{ data: [], next_cursor: 'same' }, { data: [], next_cursor: 'same' }])
    await expect(loadJourneyAccounts()).rejects.toThrow('pagination')
  })
})
