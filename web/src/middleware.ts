import { auth } from '@/auth'
import { NextResponse } from 'next/server'

/**
 * Default-deny gate for the entire app.
 *
 * `/api/cron/*` is deliberately excluded: those routes authenticate with
 * CRON_SECRET and are invoked by Vercel Cron, which cannot complete an OAuth
 * flow. Gating them here would silently kill the 60s sync and the nightly
 * reconciliation job.
 */
export default auth(req => {
  const { pathname } = req.nextUrl
  if (pathname.startsWith('/api/cron/')) return NextResponse.next()
  if (pathname.startsWith('/api/auth/')) return NextResponse.next()
  if (pathname === '/login') return NextResponse.next()

  if (!req.auth) {
    const url = new URL('/login', req.nextUrl.origin)
    return NextResponse.redirect(url)
  }
  return NextResponse.next()
})

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
