'use client'

import { useEffect, useRef, useState, useTransition } from 'react'
import { useRouter } from 'next/navigation'

/** Page-render freshness is separate from the age of each underlying source. */
export function AutoRefresh({ intervalMs = 60_000, renderedAt }: { intervalMs?: number; renderedAt: number }) {
  const router = useRouter()
  const [now, setNow] = useState(renderedAt)
  const [pending, startTransition] = useTransition()
  const savedRefresh = useRef(router.refresh)
  savedRefresh.current = router.refresh
  const pendingRef = useRef(pending)
  pendingRef.current = pending

  useEffect(() => {
    const tick = () => {
      if (document.visibilityState === 'visible' && !pendingRef.current) {
        startTransition(() => savedRefresh.current())
      }
    }
    const id = setInterval(tick, intervalMs)
    const onVisible = () => { if (document.visibilityState === 'visible') tick() }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [intervalMs])

  useEffect(() => {
    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 1_000)
    return () => clearInterval(id)
  }, [renderedAt])

  const secondsAgo = Math.max(0, Math.round((now - renderedAt) / 1000))
  const label = secondsAgo < 5 ? 'just now' : `${secondsAgo}s ago`
  return <span className="flex items-center gap-1.5 text-xs text-ink-2" title={`Completed page render; source freshness is shown separately. Refreshes every ${Math.round(intervalMs / 1000)} seconds.`}>
    <span className={`h-1.5 w-1.5 rounded-full ${pending ? 'bg-alert' : 'bg-accent'}`} aria-hidden="true" />
    <span className="numeral">{pending ? 'Refreshing…' : `View rendered ${label}`}</span>
  </span>
}
