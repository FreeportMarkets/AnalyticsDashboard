import { describe, it, expect, afterAll } from 'vitest'
import { readFileSync } from 'node:fs'
import { neon } from '@neondatabase/serverless'

const env = Object.fromEntries(
  (() => { try { return readFileSync('.env.local', 'utf8').split('\n') } catch { return [] } })()
    .filter(l => l.includes('='))
    .map(l => { const i = l.indexOf('='); return [l.slice(0, i), l.slice(i + 1).replace(/^"|"$/g, '')] })
) as Record<string, string>

const url = env.TEST_DATABASE_URL
const MARK = 'levproof_'
// Year 2078 is unused by other test files (2026, 2027, 2031, 2077, 2088 are taken).
const sql = url ? neon<boolean, boolean>(url) : null

// Same module-load-order trick as metrics.integration.test.ts: `@/lib/db`
// reads DATABASE_URL when first evaluated, so point it at the test DB before
// dynamically importing anything that pulls it in.
if (url) process.env.DATABASE_URL = url
const { recentTrades } = url
  ? await import('@/lib/metrics/trades')
  : { recentTrades: undefined as unknown as typeof import('@/lib/metrics/trades')['recentTrades'] }

/** Insert one minimal trades row. `raw` is NOT NULL but its content is unused here. */
async function seed(row: {
  timestamp: string
  type: string
  leverage: number | null
  is_close?: boolean
  display_symbol?: string
  side?: string
}) {
  await sql!(
    `INSERT INTO trades (wallet_address, timestamp, ts, type, amount_usd, display_symbol,
                         side, size, price, leverage, is_close, raw)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) ON CONFLICT DO NOTHING`,
    [`${MARK}w1`, row.timestamp, row.timestamp, row.type, 100, row.display_symbol ?? null,
     row.side ?? null, 0.5, 60000, row.leverage, row.is_close ?? null, JSON.stringify({})]
  )
}

describe.skipIf(!url)('recentTrades leverage inheritance (real DB)', () => {
  afterAll(async () => {
    if (!sql) return
    await sql('DELETE FROM trades WHERE wallet_address LIKE $1', [`${MARK}%`])
  })

  it('fills a leverage-less close from its position open; leaves orphans and swaps null', async () => {
    if (!sql) return

    // The writer regression this guards: closes stopped carrying `leverage`
    // around 2026-04-10..13 (see recentTrades's doc comment). All timestamps
    // fall on NY day 2078-03-16 (UTC-4, DST).
    await seed({ timestamp: '2078-03-16T15:00:00.000Z', type: 'perps', leverage: 12, is_close: false, display_symbol: 'BTC', side: 'long' })
    await seed({ timestamp: '2078-03-16T16:00:00.000Z', type: 'perps', leverage: null, is_close: true, display_symbol: 'BTC', side: 'long' })
    // Orphan: no prior open for this symbol -- must stay null, not borrow BTC's.
    await seed({ timestamp: '2078-03-16T17:00:00.000Z', type: 'perps', leverage: null, is_close: true, display_symbol: 'ETH', side: 'long' })
    await seed({ timestamp: '2078-03-16T18:00:00.000Z', type: 'swap', leverage: null })

    const rows = await recentTrades('2078-03-16', '2078-03-16')
    const mine = rows.filter(r => r.walletAddress === `${MARK}w1`)
    const byTs = Object.fromEntries(mine.map(r => [r.ts, r.leverage]))

    expect(byTs['2078-03-16T15:00:00.000Z']).toBe(12) // open: its own value
    expect(byTs['2078-03-16T16:00:00.000Z']).toBe(12) // close: inherited from the open
    expect(byTs['2078-03-16T17:00:00.000Z']).toBeNull() // orphan close: nothing to inherit
    expect(byTs['2078-03-16T18:00:00.000Z']).toBeNull() // swap: leverage is meaningless
  })
})
