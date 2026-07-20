import { describe, it, expect, vi } from 'vitest'

// next-auth's ESM entry imports the bare specifier "next/server" (no
// extension); the installed `next` package ships no "exports" map, so
// Node's native ESM resolver (which Vitest uses for externalized deps)
// can't resolve it — even though it resolves fine under Next's own
// bundler at build/runtime. Mock next-auth so importing '@/auth' for its
// pure isAllowedProfile export doesn't drag in that unrelated, environment
// -only resolution failure. auth.ts itself is untouched by this.
vi.mock('next-auth', () => ({
  default: () => ({ handlers: {}, auth: () => {}, signIn: () => {}, signOut: () => {} }),
}))
vi.mock('next-auth/providers/google', () => ({ default: () => ({}) }))

const { isAllowedProfile } = await import('@/auth')

const domainsOnly = { domains: 'freeportmarkets.com', emails: '' }
const emailsOnly = { domains: '', emails: 'alice@freeportmarkets.com' }
const both = { domains: 'freeportmarkets.com', emails: 'exception@other.com' }
const empty = { domains: '', emails: '' }

describe('isAllowedProfile', () => {
  it('allows a verified email whose domain matches ALLOWED_EMAIL_DOMAINS', () => {
    const profile = { email: 'alice@freeportmarkets.com', email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(true)
  })

  it('allows via the hd claim even without parsing the email domain', () => {
    const profile = { email: 'alice@example.com', email_verified: true, hd: 'freeportmarkets.com' }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(true)
  })

  it('allows an exact-match email via ALLOWED_EMAILS when domains is empty', () => {
    const profile = { email: 'alice@freeportmarkets.com', email_verified: true }
    expect(isAllowedProfile(profile, emailsOnly)).toBe(true)
  })

  it('is case-insensitive on both email and domain', () => {
    const profile = { email: 'Alice@FreeportMarkets.COM', email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(true)
  })

  it('is case-insensitive on the hd claim', () => {
    const profile = { email: 'alice@example.com', email_verified: true, hd: 'FreeportMarkets.COM' }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(true)
  })

  it('denies a matching domain when email_verified is false', () => {
    const profile = { email: 'alice@freeportmarkets.com', email_verified: false }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies a matching domain when email_verified is absent', () => {
    const profile = { email: 'alice@freeportmarkets.com' }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies an unrelated domain', () => {
    const profile = { email: 'attacker@evil.com', email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies a domain that merely shares a suffix (notfreeportmarkets.com)', () => {
    const profile = { email: 'attacker@notfreeportmarkets.com', email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies a domain that merely shares a prefix (freeportmarkets.com.evil.com)', () => {
    const profile = { email: 'attacker@freeportmarkets.com.evil.com', email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies an address engineered to fool a naive last-@ split via a query-string decoy', () => {
    const profile = { email: 'attacker@evil.com?x=@freeportmarkets.com', email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies any address containing more than one @', () => {
    const profile = { email: 'attacker@evil.com@freeportmarkets.com', email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies when email is missing entirely', () => {
    const profile = { email: undefined, email_verified: true }
    expect(isAllowedProfile(profile, domainsOnly)).toBe(false)
  })

  it('denies when profile itself is missing', () => {
    expect(isAllowedProfile(undefined, domainsOnly)).toBe(false)
    expect(isAllowedProfile(null, domainsOnly)).toBe(false)
  })

  it('fails closed when both domains and emails are empty', () => {
    const profile = { email: 'alice@freeportmarkets.com', email_verified: true, hd: 'freeportmarkets.com' }
    expect(isAllowedProfile(profile, empty)).toBe(false)
  })

  it('fails closed when config is whitespace/commas only', () => {
    const profile = { email: 'alice@freeportmarkets.com', email_verified: true, hd: 'freeportmarkets.com' }
    expect(isAllowedProfile(profile, { domains: ' , ', emails: ' , , ' })).toBe(false)
  })

  it('denies an exact-match email that is not verified', () => {
    const profile = { email: 'alice@freeportmarkets.com', email_verified: false }
    expect(isAllowedProfile(profile, emailsOnly)).toBe(false)
  })

  it('allows either mechanism to succeed when both are configured', () => {
    const viaDomain = { email: 'someone@freeportmarkets.com', email_verified: true }
    const viaException = { email: 'exception@other.com', email_verified: true }
    expect(isAllowedProfile(viaDomain, both)).toBe(true)
    expect(isAllowedProfile(viaException, both)).toBe(true)
  })

  it('denies an address on neither the domain nor the exception list', () => {
    const profile = { email: 'nobody@other.com', email_verified: true }
    expect(isAllowedProfile(profile, both)).toBe(false)
  })
})
