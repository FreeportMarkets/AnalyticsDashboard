import { describe, it, expect, vi } from 'vitest'

// middleware.ts imports `auth` from '@/auth', which imports next-auth.
// next-auth's ESM entry imports the bare specifier "next/server" (no
// extension); the installed `next` package ships no "exports" map, so
// Node's native ESM resolver (which Vitest uses for externalized deps)
// can't resolve it — even though it resolves fine under Next's own
// bundler at build/runtime. Mock next-auth (and its google provider, since
// '@/auth' imports that too) so importing '@/middleware' for its pure
// gateDecision export doesn't drag in that unrelated, environment-only
// resolution failure. Same workaround as test/auth-allowlist.test.ts.
vi.mock('next-auth', () => ({
  default: () => ({ handlers: {}, auth: () => {}, signIn: () => {}, signOut: () => {} }),
}))
vi.mock('next-auth/providers/google', () => ({ default: () => ({}) }))

const { gateDecision } = await import('@/middleware')

// Typed with an (absent) `user` property so it structurally matches
// gateDecision's `{ user?: unknown }` parameter type — this is exactly the
// real-world shape: @auth/core's error response has no `user` field.
const errorSentinel: { user?: unknown; message: string } = {
  message: 'There was a problem with the server configuration.',
}
const realUser = { user: { email: 'a@freeportmarkets.com' } }

describe('gateDecision', () => {
  it('bypasses /api/cron/sync regardless of auth state (null)', () => {
    expect(gateDecision('/api/cron/sync', null)).toBe('bypass')
  })

  it('bypasses /api/cron/sync even under the error sentinel — cron must never be gated', () => {
    expect(gateDecision('/api/cron/sync', errorSentinel)).toBe('bypass')
  })

  it('bypasses /api/cron/reconcile regardless of auth state (null)', () => {
    expect(gateDecision('/api/cron/reconcile', null)).toBe('bypass')
  })

  it('bypasses /api/cron/reconcile even under the error sentinel — cron must never be gated', () => {
    expect(gateDecision('/api/cron/reconcile', errorSentinel)).toBe('bypass')
  })

  it('bypasses /api/auth/callback/google', () => {
    expect(gateDecision('/api/auth/callback/google', null)).toBe('bypass')
  })

  it('bypasses /login', () => {
    expect(gateDecision('/login', null)).toBe('bypass')
  })

  it('allows / with a real authenticated user', () => {
    expect(gateDecision('/', realUser)).toBe('allow')
  })

  it('redirects / with null auth', () => {
    expect(gateDecision('/', null)).toBe('redirect')
  })

  it('redirects / with undefined auth', () => {
    expect(gateDecision('/', undefined)).toBe('redirect')
  })

  it('redirects / with the error sentinel — REGRESSION TEST for the fail-open bypass', () => {
    // Without the fix (`if (!req.auth)` instead of `if (!req.auth?.user)`)
    // this returns 'allow', because { message: "..." } is a truthy object.
    expect(gateDecision('/', errorSentinel)).toBe('redirect')
  })

  it('redirects / with an empty object (no user)', () => {
    expect(gateDecision('/', {})).toBe('redirect')
  })

  it('redirects a nested protected path with null auth', () => {
    expect(gateDecision('/reports/x', null)).toBe('redirect')
  })
})
