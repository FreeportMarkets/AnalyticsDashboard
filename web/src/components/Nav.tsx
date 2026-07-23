'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const LIVE_LINKS: Array<{ href: string; label: string }> = [
  { href: '/', label: 'Overview' },
  { href: '/funnels', label: 'Funnels' },
  { href: '/notifications', label: 'Notifications' },
  { href: '/trades', label: 'Trades' },
  { href: '/users', label: 'Users' },
  { href: '/referrals', label: 'Referrals' },
]

const SOON_LINKS: Array<{ label: string }> = []

/**
 * Left rail. Only routes that actually exist are real links (Overview
 * today); the rest render as visibly disabled "soon" items rather than
 * links to pages that 404.
 */
export function Nav() {
  const pathname = usePathname()

  return (
    /*
     * The rail sits on `bg-rail`, one L-step BELOW the canvas, so the
     * content area reads as the lit surface and the chrome recedes. It was
     * previously the same near-black as the page with a barely-visible
     * right border, which made the whole left quarter of the screen
     * indistinguishable from the content.
     */
    <nav className="w-52 shrink-0 border-r border-hairline bg-rail px-3 py-6">
      <div className="px-2 text-sm font-semibold tracking-tight text-ink-1">Freeport</div>
      <div className="px-2 text-xs text-ink-3">Analytics</div>
      <ul className="mt-6 space-y-0.5">
        {LIVE_LINKS.map(link => {
          const active = pathname === link.href
          return (
            <li key={link.href}>
              <Link
                href={link.href}
                aria-current={active ? 'page' : undefined}
                className={`relative block rounded-md px-2.5 py-2 text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-rail ${
                  active
                    ? 'bg-raised font-medium text-ink-1'
                    : 'text-ink-2 hover:bg-surface hover:text-ink-1'
                }`}
              >
                {link.label}
              </Link>
            </li>
          )
        })}
        {SOON_LINKS.map(link => (
          <li key={link.label}>
            <div
              aria-disabled="true"
              className="flex items-center justify-between rounded-md px-2.5 py-2 text-sm text-ink-3"
            >
              <span>{link.label}</span>
              <span className="numeral rounded-sm bg-surface px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-ink-3">
                soon
              </span>
            </div>
          </li>
        ))}
      </ul>
    </nav>
  )
}
