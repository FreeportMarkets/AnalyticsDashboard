'use client'

import { useEffect, useRef } from 'react'

function defaultFormat(n: number): string {
  return Math.round(n).toLocaleString('en-US')
}

/**
 * A numeral that counts up from 0 to `value` on mount, then settles.
 *
 * Server-rendered markup already shows the final formatted value (no
 * hydration mismatch, correct value for no-JS / SSR snapshots). The mount
 * effect then imperatively rewrites the node's textContent frame-by-frame
 * via a ref, bypassing React state so the tween doesn't cause 60 re-renders
 * a second. `prefers-reduced-motion: reduce` skips straight to the final
 * value.
 */
export function AnimatedNumber({
  value,
  format = defaultFormat,
  durationMs = 700,
  className,
}: {
  value: number
  format?: (n: number) => string
  durationMs?: number
  className?: string
}) {
  const ref = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduceMotion || value === 0) {
      el.textContent = format(value)
      return
    }

    let raf = 0
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs)
      const eased = 1 - Math.pow(1 - t, 3) // ease-out cubic
      el.textContent = format(value * eased)
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value, durationMs, format])

  return (
    <span ref={ref} className={className} suppressHydrationWarning>
      {format(value)}
    </span>
  )
}
