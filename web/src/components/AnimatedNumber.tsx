function defaultFormat(n: number): string {
  return Math.round(n).toLocaleString('en-US')
}

/**
 * A formatted numeral.
 *
 * THIS IS DELIBERATELY A SERVER COMPONENT. It previously carried a
 * `'use client'` directive and counted up from 0 on mount, which crashed every
 * page that used it:
 *
 *   Error: Functions cannot be passed directly to Client Components unless you
 *   explicitly expose it by marking it with "use server".
 *     {value: 39311, format: function F, className: ...}
 *
 * The `format` prop is a function, and functions cannot be serialized across
 * the server -> client boundary. The build succeeded because the failure only
 * happens at render time, so it reached production before anyone saw it.
 *
 * Keeping this a Server Component means `format` never has to cross that
 * boundary, so callers can pass any formatter they like. The count-up tween is
 * gone; the page-level `content-fade` animation still provides entrance motion
 * without needing client JS. Do not re-add `'use client'` here without also
 * changing every call site to pass a serializable format token instead of a
 * function.
 */
export function AnimatedNumber({
  value,
  format = defaultFormat,
  className,
}: {
  value: number
  format?: (n: number) => string
  /** Accepted and ignored; retained so existing call sites keep type-checking. */
  durationMs?: number
  className?: string
}) {
  return <span className={className}>{format(value)}</span>
}
