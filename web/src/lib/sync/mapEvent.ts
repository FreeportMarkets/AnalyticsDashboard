import type { EventRow, MapResult } from './types'

const REQUIRED = ['date', 'sk', 'event', 'timestamp'] as const

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/

/**
 * Validate that `value` is a real calendar date in YYYY-MM-DD form.
 *
 * Format alone (the regex) is not enough: `2026-13-45` matches the regex but
 * is not a date. We parse it as UTC midnight and round-trip it back through
 * `toISOString` -- `Date` silently normalizes out-of-range components (e.g.
 * `2026-02-30` becomes March 2), so a round-trip mismatch is how we catch
 * that normalization instead of accepting it.
 */
function isValidCalendarDate(value: string): boolean {
  if (!DATE_RE.test(value)) return false
  const d = new Date(`${value}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return false
  return d.toISOString().slice(0, 10) === value
}

/**
 * Optional nullable-column fields: if present but the wrong type, we absorb
 * them as `null` rather than quarantining the whole row.
 *
 * This is deliberate. These map to nullable columns, and a type-confused
 * optional field (e.g. `wallet_address: 12345`) is not evidence the rest of
 * the event is corrupt -- quarantining an otherwise-valid row over one bad
 * optional field would lose more data than it saves. The tradeoff: a
 * type-confused optional field is silently coerced to `null` with no
 * quarantine trace, so this path can mask upstream bugs in that one field.
 */
function optString(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null
}

/**
 * Map a raw DynamoDB item from `freeport-analytics-events` to an EventRow.
 *
 * The source shape is written by Swap_Server/src/services/analytics.ts:70-92.
 * The denormalized `hour` and `day_of_week` columns are deliberately dropped:
 * they are UTC-derived, while every metric in this project buckets in
 * America/New_York. Carrying them forward would invite grouping on the wrong
 * value (spec risk #1).
 */
export function mapEvent(item: unknown): MapResult<EventRow> {
  if (typeof item !== 'object' || item === null) {
    return { ok: false, reason: 'item is not an object' }
  }
  const o = item as Record<string, unknown>

  for (const f of REQUIRED) {
    if (typeof o[f] !== 'string' || (o[f] as string).length === 0) {
      return { ok: false, reason: `missing or non-string required field: ${f}` }
    }
  }

  if (!isValidCalendarDate(o.date as string)) {
    return { ok: false, reason: `invalid date: ${String(o.date)}` }
  }

  const ts = new Date(o.timestamp as string)
  if (Number.isNaN(ts.getTime())) {
    return { ok: false, reason: `unparseable timestamp: ${String(o.timestamp)}` }
  }

  let metadata: Record<string, unknown> | null = null
  if (o.metadata !== undefined && o.metadata !== null) {
    if (
      typeof o.metadata !== 'object' ||
      Array.isArray(o.metadata) ||
      (Object.getPrototypeOf(o.metadata) !== Object.prototype &&
        Object.getPrototypeOf(o.metadata) !== null)
    ) {
      return { ok: false, reason: 'metadata is present but not a plain object' }
    }
    metadata = o.metadata as Record<string, unknown>
  }

  return {
    ok: true,
    value: {
      date: o.date as string,
      sk: o.sk as string,
      ts,
      event: o.event as string,
      screen: optString(o.screen),
      component: optString(o.component),
      wallet_address: optString(o.wallet_address),
      session_id: optString(o.session_id),
      platform: optString(o.platform),
      app_version: optString(o.app_version),
      metadata,
    },
  }
}
