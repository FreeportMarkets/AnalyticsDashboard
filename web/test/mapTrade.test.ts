import { describe, it, expect } from 'vitest'
import { mapTrade } from '@/lib/sync/mapTrade'

// Perps shape: freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts:167-192
const perp = {
  wallet_address: '0xabc', timestamp: '2026-07-20T12:00:00.000Z', trade_date: '2026-07-20',
  id: '1753000000000-x9k2', type: 'perps', asset: 'BTC', display_symbol: 'BTC',
  side: 'long', size: 0.05, price: 61000, leverage: 5, amount_usd: 610,
  order_type: 'market', is_close: false, is_hip3: false, category: 'hl',
  status: 'success', source: 'perps', client: 'web', trace_id: 'tr-1',
}

// Swap shape: Swap_Server/src/services/dynamodb.ts:67-107
const swap = {
  wallet_address: '0xdef', timestamp: '2026-07-20T13:00:00.000Z', trade_date: '2026-07-20',
  id: 'sw-1', type: 'swap', amount_usd: 42.5, status: 'success', source: 'swap',
  from_token: 'USDC', to_token: 'SOL', amount_from_token: 42.5, amount_to_token: 0.3,
  tx_signature: 'sig123', request_id: 'req-1',
}

describe('mapTrade', () => {
  it('maps a perps row', () => {
    const r = mapTrade(perp)
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.type).toBe('perps')
    expect(r.value.amount_usd).toBe(610)
    expect(r.value.client).toBe('web')
    expect(r.value.leverage).toBe(5)
    expect(r.value.ts.toISOString()).toBe('2026-07-20T12:00:00.000Z')
  })

  it('maps a swap row, leaving perps fields null', () => {
    const r = mapTrade(swap)
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.type).toBe('swap')
    expect(r.value.asset).toBeNull()
    expect(r.value.leverage).toBeNull()
    expect(r.value.from_token).toBe('USDC')
  })

  it('retains the full raw item', () => {
    // amount_usd on perps rows is MARGIN, not notional (trade-logger.ts:39-46),
    // and volume math may need fields this schema does not name. Keep everything.
    const r = mapTrade(perp)
    expect(r.ok && r.value.raw).toEqual(perp)
  })

  it('defaults a missing client to unknown, matching the writer', () => {
    const { client, ...noClient } = perp
    const r = mapTrade(noClient)
    expect(r.ok && r.value.client).toBe('unknown')
  })

  it('preserves numeric precision as a string-safe value', () => {
    const r = mapTrade({ ...perp, amount_usd: 0.1 + 0.2 })
    expect(r.ok && r.value.amount_usd).toBeCloseTo(0.30000000000000004, 15)
  })

  it('quarantines a missing key field', () => {
    const { wallet_address, ...bad } = perp
    const r = mapTrade(bad)
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/wallet_address/)
  })

  it('quarantines an unparseable timestamp', () => {
    expect(mapTrade({ ...perp, timestamp: 'nope' }).ok).toBe(false)
  })
})
