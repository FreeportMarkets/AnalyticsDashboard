import NextAuth from 'next-auth'
import Google from 'next-auth/providers/google'

/** Parse a comma-separated env value into trimmed, lowercased, non-empty entries. */
function parseList(value: string | undefined): string[] {
  return (value ?? '')
    .split(',')
    .map(s => s.trim().toLowerCase())
    .filter(Boolean)
}

/**
 * Domain part of an email address: the substring after the LAST '@',
 * lowercased and trimmed — but only for a syntactically single-address
 * string. An address containing more than one '@' (e.g. an attacker
 * embedding "@freeportmarkets.com" as a trailing decoy — think
 * "evil.com?x=@freeportmarkets.com" or "attacker@evil.com@freeportmarkets.com")
 * is rejected outright rather than trusting whatever comes after the last
 * '@', since that decoy suffix is exactly what a naive last-'@' split (or
 * an even naiver endsWith('@domain')) would be fooled by. A well-formed
 * email has exactly one '@', so requiring that also makes "last '@'" and
 * "the '@'" the same thing for anything we actually accept.
 */
function emailDomain(email: string): string {
  const parts = email.split('@')
  if (parts.length !== 2) return ''
  const [local, domain] = parts
  if (!local || !domain) return ''
  return domain.trim().toLowerCase()
}

type ProfileLike = {
  email?: string | null
  email_verified?: boolean | null
  hd?: string
} | undefined | null

type AllowlistConfig = {
  domains: string
  emails: string
}

/**
 * Decide whether a Google OIDC profile is allowed to sign in.
 *
 * Fail closed: if both ALLOWED_EMAIL_DOMAINS and ALLOWED_EMAILS are empty
 * or unset (after parsing), nobody is allowed in. A misconfiguration must
 * lock people out rather than admit anyone with a Google account.
 *
 * Domain matching prefers Google's `hd` (hosted domain) claim over parsing
 * the email string: `hd` is populated only for Google Workspace accounts
 * and is asserted by Google itself, so it can't be spoofed by an email
 * string trick the way a parsed domain could be. The email-domain path is
 * still needed for Workspace-less accounts / general OAuth flows, but it
 * is only trusted when `email_verified` is exactly `true`.
 */
export function isAllowedProfile(profile: ProfileLike, config: AllowlistConfig): boolean {
  const allowedDomains = parseList(config.domains)
  const allowedEmails = parseList(config.emails)

  if (allowedDomains.length === 0 && allowedEmails.length === 0) return false
  if (!profile) return false

  const email = typeof profile.email === 'string' ? profile.email.trim().toLowerCase() : ''

  // Exact-email exception list, for occasional non-domain accounts. Still
  // gated on email_verified: an unverified claim can assert any address,
  // including one that happens to sit on the exception list.
  if (email && profile.email_verified === true && allowedEmails.includes(email)) return true

  if (allowedDomains.length > 0) {
    // Strongest signal: Google's own hosted-domain claim, present only for
    // Workspace accounts. Trust it regardless of email_verified, since it
    // comes from Google, not from the (attacker-controllable) email string.
    const hd = typeof profile.hd === 'string' ? profile.hd.trim().toLowerCase() : ''
    if (hd && allowedDomains.includes(hd)) return true

    // Weaker signal: the email string's own domain. Only trust this when
    // Google has verified the email — an unverified claim could assert
    // any address, including one on an allowed domain it doesn't own.
    if (email && profile.email_verified === true) {
      const domain = emailDomain(email)
      if (domain && allowedDomains.includes(domain)) return true
    }
  }

  return false
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [Google],
  pages: { signIn: '/login' },
  callbacks: {
    // Thin wrapper: all decision logic lives in isAllowedProfile so it is
    // unit-testable without a live OAuth flow.
    signIn({ profile }) {
      return isAllowedProfile(profile, {
        domains: process.env.ALLOWED_EMAIL_DOMAINS ?? '',
        emails: process.env.ALLOWED_EMAILS ?? '',
      })
    },
  },
})
