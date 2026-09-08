import { describe, expect, it, vi } from 'vitest'
import { orderMilestones, tradeOutcome } from '@/lib/journey'
import type { FunnelStep } from '@/lib/funnelApi'
import { accountWallets, buildDailyJourney } from '@/lib/metrics/journeyDaily'
import type { JourneyAccount } from '@/lib/privyAccounts'

vi.mock('next/cache', () => ({ unstable_cache: (fn: unknown) => fn }))

const step = (key: string, reached: number, index: number, size = 100): FunnelStep => ({ step_key: key, step_index: index, cohort_size: size, users_reached: reached, conversion: reached / size, p50_ms: null, p90_ms: null })
describe('Journey milestones', () => {
  const steps = [step('intro_started', 100, 0), step('deposit_completed', 10, 1), step('trade_succeeded', 30, 2)]
  it('sorts independent counts descending without changing or clamping them', () => {
    expect(orderMilestones(steps, 'reach').map(s => s.users_reached)).toEqual([100, 30, 10])
    expect(steps.map(s => s.users_reached)).toEqual([100, 10, 30])
    expect(orderMilestones([...steps].reverse(), 'journey')).toEqual(steps)
  })
  it('derives conversion only from the cohort and trade outcome', () => {
    expect(tradeOutcome([...steps].reverse())).toMatchObject({ cohortSize: 100, traded: 30, withoutTrade: 70, conversion: 0.3 })
    expect(tradeOutcome(steps.slice(0, 2)).traded).toBeNull()
    expect(tradeOutcome([step('intro_started', 0, 0, 0), step('trade_succeeded', 0, 1, 0)]).conversion).toBeNull()
    expect(tradeOutcome([step('intro_started', 100, 0), step('trade_succeeded', 0, 1)])).toMatchObject({ traded: 0, withoutTrade: 100 })
  })
  it('withholds inferred metrics for inconsistent counts', () => {
    expect(tradeOutcome([...steps, step('deposit_opened', 101, 3)])).toMatchObject({ valid: false, conversion: null })
    expect(tradeOutcome([steps[0]!, step('trade_succeeded', 20, 1, 50)]).valid).toBe(false)
  })
})

describe('Daily journey', () => {
  const accounts: JourneyAccount[] = [
    { id: 'old', createdAt: '2026-08-01T12:00:00Z', wallets: ['0xAB'] },
    { id: 'new', createdAt: '2026-09-08T04:01:00Z', wallets: ['0xcd', 'SolABC'] },
    { id: 'walletless', createdAt: '2026-09-08T08:00:00Z', wallets: [] },
  ]
  it('separates action-day firsts from same-day signup conversion and zero fills', () => {
    const rows = buildDailyJourney(accounts, [
      { account_id: 'old', first_trade: '2026-09-08T10:00:00Z', first_deposit: '2026-08-02T10:00:00Z' },
      { account_id: 'new', first_trade: '2026-09-09T03:59:59Z', first_deposit: '2026-09-08T11:00:00Z' },
    ], '2026-09-07', '2026-09-09')
    expect(rows).toEqual([
      { day: '2026-09-07', signups: 0, firstTrades: 0, firstDeposits: 0, signupDayTrades: 0, signupDayDeposits: 0 },
      { day: '2026-09-08', signups: 2, firstTrades: 2, firstDeposits: 1, signupDayTrades: 1, signupDayDeposits: 1 },
      { day: '2026-09-09', signups: 0, firstTrades: 0, firstDeposits: 0, signupDayTrades: 0, signupDayDeposits: 0 },
    ])
  })
  it('does not reset earlier actions when the visible range changes', () => {
    const actions = [{ account_id: 'old', first_trade: '2026-08-03T10:00:00Z', first_deposit: null }]
    expect(buildDailyJourney(accounts, actions, '2026-09-08', '2026-09-08')[0]?.firstTrades).toBe(0)
  })
  it('counts duplicate signup IDs once and excludes actions predating account creation', () => {
    const rows = buildDailyJourney([...accounts, accounts[1]!], [{ account_id: 'new', first_trade: '2026-09-08T04:00:00Z', first_deposit: null }], '2026-09-08', '2026-09-08')
    expect(rows[0]).toMatchObject({ signups: 2, firstTrades: 0 })
  })
  it('handles both repeated hours at the autumn DST boundary', () => {
    const rows = buildDailyJourney([{ id: 'dst', createdAt: '2026-11-01T05:30:00Z', wallets: [] }], [{ account_id: 'dst', first_trade: '2026-11-01T06:30:00Z', first_deposit: '2026-11-02T04:59:59Z' }], '2026-11-01', '2026-11-02')
    expect(rows[0]).toMatchObject({ signups: 1, signupDayTrades: 1, signupDayDeposits: 1 })
    expect(rows[1]?.firstDeposits).toBe(0)
  })
  it('merges EVM casing, preserves Solana casing, and excludes ambiguous wallet links', () => {
    expect(accountWallets([...accounts, { id: 'conflict', createdAt: accounts[0]!.createdAt, wallets: ['0xab', 'solABC'] }])).toEqual([
      { wallet: '0xcd', id: 'new' }, { wallet: 'SolABC', id: 'new' }, { wallet: 'solABC', id: 'conflict' },
    ])
  })
})
