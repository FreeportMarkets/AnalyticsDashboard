import type { EventRow, MapResult } from './types'

const REQUIRED = ['date', 'sk', 'event', 'timestamp'] as const

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

  const ts = new Date(o.timestamp as string)
  if (Number.isNaN(ts.getTime())) {
    return { ok: false, reason: `unparseable timestamp: ${String(o.timestamp)}` }
  }

  let metadata: Record<string, unknown> | null = null
  if (o.metadata !== undefined && o.metadata !== null) {
    if (typeof o.metadata !== 'object' || Array.isArray(o.metadata)) {
      return { ok: false, reason: 'metadata is present but not an object' }
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
