'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

/**
 * Keeps a server-rendered page live without a manual reload. Calls
 * `router.refresh()` on an interval, which re-runs the server component and
 * streams fresh data into the existing DOM -- no full navigation, no scroll
 * jump, no flash. The page stays a fast server component; this is the only
 * client JS on it.
 *
 * Pauses while the tab is hidden (no point polling a backgrounded tab) and
 * refreshes once immediately on becoming visible again, so returning to the
 * tab shows current numbers at once.
 *
 * Renders a tiny "updated Xs ago" indicator so it's visible that the page is
 * live, not stale.
 */
export function AutoRefresh({ intervalMs = 60_000 }: { intervalMs?: number }) {
  const router = useRouter()
  const [lastRefresh, setLastRefresh] = useState(() => Date.now())
  const [now, setNow] = useState(() => Date.now())
  const savedRefresh = useRef(router.refresh)
  savedRefresh.current = router.refresh

  useEffect(() => {
    const tick = () => {
      if (document.visibilityState === 'visible') {
        savedRefresh.current()
        setLastRefresh(Date.now())
      }
    }
    const id = setInterval(tick, intervalMs)
    const onVisible = () => {
      if (document.visibilityState === 'visible') tick()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [intervalMs])

  // A 1s clock purely for the "Xs ago" label; independent of the data refresh.
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1_000)
    return () => clearInterval(id)
  }, [])

  const secondsAgo = Math.max(0, Math.round((now - lastRefresh) / 1000))
  const label = secondsAgo < 5 ? 'just now' : `${secondsAgo}s ago`

  return (
    <span className="flex items-center gap-1.5 text-xs text-ink-3" title={`Auto-refreshing every ${Math.round(intervalMs / 1000)}s`}>
      <span className="relative flex h-1.5 w-1.5" aria-hidden="true">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-accent" />
      </span>
      <span className="numeral">updated {label}</span>
    </span>
  )
}
