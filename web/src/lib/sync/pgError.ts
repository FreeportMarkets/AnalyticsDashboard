/**
 * Classifies a thrown value as a Postgres "data error" -- SQLSTATE class 22
 * (data exception, e.g. `date/time field value out of range`) or class 23
 * (integrity constraint violation, e.g. a NOT NULL or unique violation).
 * These are errors ABOUT THE ROW being inserted, not about the database's
 * availability.
 *
 * The neon driver surfaces the raw Postgres SQLSTATE on `err.code`, matching
 * node-postgres. Anything without a string `code` -- a network failure, an
 * auth failure, a syntax error, a thrown JS TypeError -- is NOT a data error
 * and must be treated as fail-safe: NOT a data error, so it propagates.
 * Bisecting on a connection failure would relabel an outage as "every row in
 * this batch is corrupt" and silently quarantine a healthy batch instead of
 * failing the sync tick loudly.
 */
export function isDataError(err: unknown): err is { code: string; message?: unknown } {
  if (typeof err !== 'object' || err === null) return false
  const code = (err as { code?: unknown }).code
  if (typeof code !== 'string') return false
  return code.startsWith('22') || code.startsWith('23')
}

/** Best-effort human-readable message for a caught value. */
export function pgErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'object' && err !== null) {
    const m = (err as { message?: unknown }).message
    if (typeof m === 'string') return m
  }
  return String(err)
}
