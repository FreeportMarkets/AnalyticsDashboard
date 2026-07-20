import type { EventRow, MapResult } from './types'
import { isValidCalendarDate } from '../time'

const REQUIRED = ['date', 'sk', 'event', 'timestamp'] as const

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
 * Recursively checks that `value` is JSON-safe -- something that will
 * round-trip through `JSON.stringify`/the `jsonb` column without silent
 * data loss. Returns `null` if safe, or a human-readable reason if not.
 *
 * A shallow top-level check is not enough: `DynamoDBDocumentClient`
 * unmarshals DynamoDB `SS`/`NS`/`BS` set types to native `Set`, so
 * `{ tags: new Set(['a', 'b']) }` is a real shape this pipeline sees, not a
 * hypothetical. `JSON.stringify` turns a `Set` (and a `Map`, and a `Date`
 * inside an object) into `{}` with no error, so the data vanishes into the
 * `jsonb` column silently. Non-finite numbers are the same corruption
 * class: `JSON.stringify(NaN)` and `JSON.stringify(Infinity)` both produce
 * `null`.
 *
 * `seen` tracks the objects/arrays currently on the recursion path (not
 * every object visited) so that a shared-but-non-cyclic reference -- the
 * same sub-object reachable from two different keys -- is not mistaken for
 * a cycle. It is removed on the way back out (backtracking), which is what
 * makes this a path check rather than a global "already visited" check. A
 * genuinely self-referencing object would otherwise recurse forever and
 * either loop or blow the stack; `mapEvent` must never throw, so this is
 * checked explicitly rather than left to surface as a crash.
 */
function jsonSafeViolation(value: unknown, seen: Set<object>): string | null {
  if (value === null) return null
  const t = typeof value
  if (t === 'string' || t === 'boolean') return null
  if (t === 'number') {
    return Number.isFinite(value as number) ? null : 'contains a non-finite number'
  }
  if (Array.isArray(value)) {
    if (seen.has(value)) return 'contains a cyclic reference'
    seen.add(value)
    for (const el of value) {
      const violation = jsonSafeViolation(el, seen)
      if (violation) {
        seen.delete(value)
        return violation
      }
    }
    seen.delete(value)
    return null
  }
  if (t === 'object') {
    const proto = Object.getPrototypeOf(value)
    if (proto !== Object.prototype && proto !== null) {
      const ctorName = (value as { constructor?: { name?: string } })?.constructor?.name
      return `contains a non-plain-object value (${ctorName ?? 'unknown type'})`
    }
    const obj = value as object
    if (seen.has(obj)) return 'contains a cyclic reference'
    seen.add(obj)
    for (const v of Object.values(obj as Record<string, unknown>)) {
      const violation = jsonSafeViolation(v, seen)
      if (violation) {
        seen.delete(obj)
        return violation
      }
    }
    seen.delete(obj)
    return null
  }
  return `contains an unsupported value (${t})`
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
    const violation = jsonSafeViolation(o.metadata, new Set())
    if (violation) {
      return { ok: false, reason: `metadata is present but not JSON-safe: ${violation}` }
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
