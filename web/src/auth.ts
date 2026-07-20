import NextAuth from 'next-auth'
import Google from 'next-auth/providers/google'

/** Comma-separated allowlist, e.g. "a@x.com,b@x.com". Empty = deny everyone. */
function allowedEmails(): string[] {
  return (process.env.ALLOWED_EMAILS ?? '')
    .split(',')
    .map(e => e.trim().toLowerCase())
    .filter(Boolean)
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [Google],
  pages: { signIn: '/login' },
  callbacks: {
    /**
     * Deny by default. An unset or empty ALLOWED_EMAILS locks everyone out
     * rather than letting anyone with a Google account in — a misconfiguration
     * must fail closed.
     */
    signIn({ profile }) {
      const email = profile?.email?.toLowerCase()
      if (!email) return false
      return allowedEmails().includes(email)
    },
  },
})
