import { auth } from '@/auth'
import { NextResponse } from 'next/server'

export type GateDecision = 'bypass' | 'allow' | 'redirect'

/**
 * Pure routing decision for the auth gate, extracted from the `auth()`
 * wrapper below so it is unit-testable without importing next-auth.
 *
 * This gates on `auth?.user`, NOT on the truthiness of `auth` alone.
 * That distinction is the entire fix: when @auth/core's `assertConfig`
 * fails (e.g. a missing `AUTH_SECRET`, or `trustHost` resolving false
 * when running off-Vercel), `Auth()` does not throw. It returns
 * `Response.json({ message: "..." }, { status: 500 })`. `getSession()`
 * then calls `.json()` on that response and gets back a truthy plain
 * object — `{ message: "..." }` — which `handleAuth` assigns straight to
 * `req.auth`. A bare `if (!req.auth)` check treats that error sentinel as
 * an authenticated session and lets every protected route through with
 * NO real session at all (fail-open). A real session always carries
 * `user`, and the error sentinel never does, so checking `auth?.user`
 * fails closed instead. Do not simplify this back to `!auth`.
 */
export function gateDecision(
  pathname: string,
  auth: { user?: unknown } | null | undefined
): GateDecision {
  if (pathname.startsWith('/api/cron/')) return 'bypass'
  if (pathname.startsWith('/api/auth/')) return 'bypass'
  if (pathname === '/login') return 'bypass'

  return auth?.user ? 'allow' : 'redirect'
}

/**
 * Default-deny gate for the entire app.
 *
 * `/api/cron/*` is deliberately excluded: those routes authenticate with
 * CRON_SECRET and are invoked by Vercel Cron, which cannot complete an OAuth
 * flow. Gating them here would silently kill the 60s sync and the nightly
 * reconciliation job.
 */
export default auth(req => {
  const decision = gateDecision(req.nextUrl.pathname, req.auth)

  if (decision === 'redirect') {
    const url = new URL('/login', req.nextUrl.origin)
    return NextResponse.redirect(url)
  }
  return NextResponse.next()
})

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
