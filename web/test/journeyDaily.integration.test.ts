import { randomUUID } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { neon } from '@neondatabase/serverless'
import { describe, expect, it, vi } from 'vitest'
import { firstAccountActions, buildDailyJourney } from '@/lib/metrics/journeyDaily'

vi.mock('next/cache', () => ({ unstable_cache: (fn: unknown) => fn }))
let url = process.env.TEST_DATABASE_URL
if (!url) {
  try { url = /^TEST_DATABASE_URL=["']?([^\r\n"']+)/m.exec(readFileSync('.env.local', 'utf8'))?.[1] } catch {}
}
const sql = url ? neon<boolean, boolean>(url) : null

describe.skipIf(!sql)('Daily first actions (real Postgres)', () => {
  it('deduplicates sources and wallets, excludes failed orders, and retains earlier firsts', async () => {
    if (!sql) return
    const prefix = `journey_test_${randomUUID()}`
    const wallets = [`${prefix}_a`, `${prefix}_b`, `${prefix}_old`, `${prefix}_failed`]
    const accounts = [
      { id: 'new', createdAt: '2091-09-08T04:00:00Z', wallets: wallets.slice(0, 2) },
      { id: 'old', createdAt: '2091-08-01T04:00:00Z', wallets: [wallets[2]!] },
      { id: 'failed', createdAt: '2091-09-08T04:00:00Z', wallets: [wallets[3]!] },
    ]
    try {
      for (const [index, wallet, event, ts] of [
        [0, wallets[0], 'trade_succeeded', '2091-09-08T10:00:00Z'],
        [1, wallets[1], 'trade_success', '2091-09-08T11:00:00Z'],
        [2, wallets[0], 'deposit_completed', '2091-09-08T12:00:00Z'],
        [3, wallets[1], 'deposit_funds_arrived', '2091-09-09T04:01:00Z'],
        [4, wallets[2], 'trade_success', '2091-08-15T12:00:00Z'],
        [5, wallets[2], 'trade_success', '2091-09-08T13:00:00Z'],
        [6, wallets[3], 'trade_failed', '2091-09-08T13:00:00Z'],
      ] as const) {
        await sql('INSERT INTO events (date, sk, ts, event, wallet_address) VALUES ($1, $2, $3, $4, $5)', [ts.slice(0, 10), `${prefix}_${index}`, ts, event, wallet])
      }
      for (const [wallet, status] of [[wallets[0], 'success'], [wallets[3], 'pending'], [wallets[3], 'failed']] as const) {
        await sql('INSERT INTO trades (wallet_address, timestamp, ts, type, status, raw) VALUES ($1, $2, $3, $4, $5, $6)', [wallet, `${prefix}_${status}`, '2091-09-08T10:00:00Z', 'swap', status, '{}'])
      }
      const actions = await firstAccountActions(sql, accounts, '2091-09-09')
      expect(actions).toHaveLength(2)
      const daily = buildDailyJourney(accounts, actions, '2091-09-08', '2091-09-09')
      expect(daily[0]).toMatchObject({ signups: 2, firstTrades: 1, firstDeposits: 1, signupDayTrades: 1, signupDayDeposits: 1 })
      expect(daily[1]).toMatchObject({ firstDeposits: 0, firstTrades: 0 })
    } finally {
      await sql('DELETE FROM events WHERE wallet_address = ANY($1::text[])', [wallets])
      await sql('DELETE FROM trades WHERE wallet_address = ANY($1::text[])', [wallets])
    }
  })
})
