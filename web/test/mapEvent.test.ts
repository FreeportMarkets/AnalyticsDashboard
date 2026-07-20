import { describe, it, expect } from 'vitest'
import { mapEvent } from '@/lib/sync/mapEvent'

// Shape produced by Swap_Server/src/services/analytics.ts:70-92
const valid = {
  date: '2026-07-19',
  sk: '2026-07-19T14:32:11.482Z#0xabcdef#k3j4h5g6',
  event: 'trade_success',
  screen: 'trade',
  component: 'SwapModal',
  wallet_address: '0xABCDEF0123456789',
  session_id: 'm4k2j1-x9',
  timestamp: '2026-07-19T14:32:11.482Z',
  metadata: { asset: 'BTC', amount_usd: 250.5 },
  platform: 'ios',
  app_version: '1.4.2',
  hour: 14,
  day_of_week: 0,
}

describe('mapEvent', () => {
  it('maps a well-formed item', () => {
    const r = mapEvent(valid)
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.date).toBe('2026-07-19')
    expect(r.value.sk).toBe(valid.sk)
    expect(r.value.ts.toISOString()).toBe('2026-07-19T14:32:11.482Z')
    expect(r.value.event).toBe('trade_success')
    expect(r.value.metadata).toEqual({ asset: 'BTC', amount_usd: 250.5 })
  })

  it('preserves wallet_address case exactly', () => {
    // Downstream joins lowercase explicitly; the mirror must not pre-normalize
    // or it stops being a faithful copy of the source row.
    const r = mapEvent(valid)
    expect(r.ok && r.value.wallet_address).toBe('0xABCDEF0123456789')
  })

  it('drops the denormalized UTC hour and day_of_week', () => {
    // These are UTC-derived; the dashboard must recompute in America/New_York.
    // Carrying them forward invites a GROUP BY on the wrong value (spec risk #1).
    const r = mapEvent(valid)
    expect(r.ok && 'hour' in (r.value as object)).toBe(false)
    expect(r.ok && 'day_of_week' in (r.value as object)).toBe(false)
  })

  it('allows absent optional fields', () => {
    const r = mapEvent({
      date: '2026-07-19', sk: 'a', event: 'session_start',
      wallet_address: '0x1', session_id: 's', timestamp: '2026-07-19T00:00:00.000Z',
    })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.screen).toBeNull()
    expect(r.value.component).toBeNull()
    expect(r.value.wallet_address).toBe('0x1')
    expect(r.value.session_id).toBe('s')
    expect(r.value.platform).toBeNull()
    expect(r.value.app_version).toBeNull()
    expect(r.value.metadata).toBeNull()
  })

  it('nulls a present-but-wrong-typed optional field rather than quarantining the row', () => {
    // Documents the deliberate optString tradeoff: type-confused optional
    // fields are absorbed as null, not treated as row-level corruption.
    const r = mapEvent({ ...valid, wallet_address: 12345 })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.wallet_address).toBeNull()
  })

  it('quarantines a missing required field', () => {
    const { date, ...noDate } = valid
    const r = mapEvent(noDate)
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/date/)
  })

  it.each(['sk', 'event', 'timestamp'] as const)(
    'quarantines a missing required field: %s',
    (field) => {
      const clone = { ...valid } as Record<string, unknown>
      delete clone[field]
      const r = mapEvent(clone)
      expect(r.ok).toBe(false)
      if (r.ok) return
      expect(r.reason).toMatch(new RegExp(field))
    },
  )

  it('quarantines an unparseable timestamp', () => {
    const r = mapEvent({ ...valid, timestamp: 'not-a-date' })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/timestamp/)
  })

  it('quarantines a non-object', () => {
    expect(mapEvent(null).ok).toBe(false)
    expect(mapEvent('nope').ok).toBe(false)
    expect(mapEvent([1, 2, 3]).ok).toBe(false)
    expect(mapEvent(42).ok).toBe(false)
  })

  it('quarantines non-string metadata rather than coercing', () => {
    const r = mapEvent({ ...valid, metadata: 'a string' })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it.each(['not-a-date', '2026-13-45', '2026-02-30', '26-01-01'])(
    'quarantines an invalid date: %s',
    (badDate) => {
      const r = mapEvent({ ...valid, date: badDate })
      expect(r.ok).toBe(false)
      if (r.ok) return
      expect(r.reason).toMatch(/date/)
    },
  )

  it('quarantines metadata that is a Set (would silently vanish on JSON.stringify)', () => {
    const r = mapEvent({ ...valid, metadata: new Set([1, 2]) })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('quarantines metadata that is a Map', () => {
    const r = mapEvent({ ...valid, metadata: new Map([['a', 1]]) })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('quarantines metadata that is a Date', () => {
    const r = mapEvent({ ...valid, metadata: new Date() })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('allows plain-object metadata with nested objects and arrays', () => {
    const r = mapEvent({
      ...valid,
      metadata: { asset: 'BTC', tags: ['a', 'b'], nested: { x: 1, y: [1, 2, 3] } },
    })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.metadata).toEqual({
      asset: 'BTC',
      tags: ['a', 'b'],
      nested: { x: 1, y: [1, 2, 3] },
    })
  })

  // DynamoDBDocumentClient unmarshals DynamoDB SS/NS/BS set types to native
  // Set, so a nested Set is a real shape for this pipeline, not a
  // hypothetical. A shallow top-level check would miss it -- the object
  // itself is a plain object, only the nested value is unsafe -- and it
  // would silently JSON.stringify to `{}` in the jsonb column.
  it('quarantines metadata containing a nested Set', () => {
    const r = mapEvent({ ...valid, metadata: { tags: new Set(['a', 'b']) } })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('quarantines metadata containing a nested Map', () => {
    const r = mapEvent({ ...valid, metadata: { nested: { m: new Map([['k', 1]]) } } })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('quarantines metadata containing a nested Date', () => {
    const r = mapEvent({ ...valid, metadata: { createdAt: new Date() } })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('quarantines metadata containing NaN', () => {
    const r = mapEvent({ ...valid, metadata: { amount_usd: NaN } })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('quarantines metadata containing Infinity', () => {
    const r = mapEvent({ ...valid, metadata: { amount_usd: Infinity } })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('quarantines cyclic metadata without throwing or hanging', () => {
    const cyclic: Record<string, unknown> = { asset: 'BTC' }
    cyclic.self = cyclic
    expect(() => mapEvent({ ...valid, metadata: cyclic })).not.toThrow()
    const r = mapEvent({ ...valid, metadata: cyclic })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })

  it('allows deeply nested plain-object/array metadata', () => {
    const r = mapEvent({
      ...valid,
      metadata: {
        a: { b: { c: [1, 2, { d: ['e', 'f', { g: null, h: true, i: 3.5 }] }] } },
      },
    })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.metadata).toEqual({
      a: { b: { c: [1, 2, { d: ['e', 'f', { g: null, h: true, i: 3.5 }] }] } },
    })
  })

  it('allows metadata with a null-prototype top-level object', () => {
    const metadata = Object.create(null) as Record<string, unknown>
    metadata.asset = 'BTC'
    const r = mapEvent({ ...valid, metadata })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.metadata).toEqual({ asset: 'BTC' })
  })
})
