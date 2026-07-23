import { describe, expect, it } from 'vitest'
import { embeddedEvmWallets } from '../src/lib/volume/walletSource'

describe('embeddedEvmWallets', () => {
  const user = (accts: object[]) => ({ linked_accounts: accts })

  it('keeps only embedded (privy) ethereum wallets, lowercased', () => {
    const out = embeddedEvmWallets([
      user([{ type: 'wallet', chain_type: 'ethereum', wallet_client_type: 'privy', address: '0xAbC' }]),
    ])
    expect(out).toEqual(['0xabc'])
  })

  it('excludes external linked wallets (metamask/phantom)', () => {
    const out = embeddedEvmWallets([
      user([
        { type: 'wallet', chain_type: 'ethereum', wallet_client_type: 'metamask', address: '0xEXT' },
        { type: 'wallet', chain_type: 'ethereum', wallet_client_type: 'phantom', address: '0xEXT2' },
      ]),
    ])
    expect(out).toEqual([])
  })

  it('excludes solana embedded wallets', () => {
    const out = embeddedEvmWallets([
      user([{ type: 'wallet', chain_type: 'solana', wallet_client_type: 'privy', address: 'SoLaNa' }]),
    ])
    expect(out).toEqual([])
  })

  it('dedupes the same embedded wallet across users', () => {
    const acct = { type: 'wallet', chain_type: 'ethereum', wallet_client_type: 'privy', address: '0xSAME' }
    expect(embeddedEvmWallets([user([acct]), user([acct])])).toEqual(['0xsame'])
  })

  it('ignores non-wallet accounts (email, oauth, smart_wallet)', () => {
    const out = embeddedEvmWallets([
      user([
        { type: 'email', address: 'a@b.com' },
        { type: 'google_oauth' },
        { type: 'smart_wallet', address: '0xSMART' },
        { type: 'wallet', chain_type: 'ethereum', wallet_client_type: 'privy', address: '0xReal' },
      ]),
    ])
    expect(out).toEqual(['0xreal'])
  })
})
