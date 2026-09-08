import type { neon } from '@neondatabase/serverless'
import type { JourneyAccount } from '@/lib/privyAccounts'
import { normalizeWallet } from '@/lib/privyAccounts'
import { nyDateOf } from '@/lib/time'
import { nyRangeToUtc } from './nyRange'
import { addDays } from '@/lib/ranges'

export interface FirstAction {
  account_id: string
  first_trade: string | null
  first_deposit: string | null
}
export interface DailyJourneyPoint {
  day: string
  signups: number
  firstTrades: number
  firstDeposits: number
  signupDayTrades: number
  signupDayDeposits: number
}

/** Refuse ambiguous wallet ownership rather than duplicate actions. */
export function accountWallets(accounts: JourneyAccount[]) {
  const owners = new Map<string, string | null>()
  for (const account of accounts) for (const raw of account.wallets) {
    const wallet = normalizeWallet(raw)
    if (['', 'server', 'unknown', 'system'].includes(wallet)) continue
    if (owners.has(wallet) && owners.get(wallet) !== account.id) owners.set(wallet, null)
    else owners.set(wallet, account.id)
  }
  return [...owners].flatMap(([wallet, id]) => id ? [{ wallet, id }] : [])
}

/** MIN is over ALL stored history, before the display range is applied. */
export async function firstAccountActions(sql: ReturnType<typeof neon>, accounts: JourneyAccount[], endDate: string): Promise<FirstAction[]> {
  const mappings = accountWallets(accounts)
  if (!mappings.length) return []
  const { toUtc } = nyRangeToUtc(endDate, endDate)
  return await sql(
    `WITH account_wallets AS (
       SELECT * FROM unnest($1::text[], $2::text[]) AS m(wallet, account_id)
     ), successes AS (
       SELECT wallet_address, ts,
              CASE WHEN event IN ('trade_success', 'trade_succeeded') THEN 'trade' ELSE 'deposit' END AS kind
         FROM events
        WHERE ts < $3 AND (platform IS NULL OR platform <> 'server')
          AND event IN ('trade_success', 'trade_succeeded', 'deposit_success', 'deposit_completed', 'deposit_funds_arrived')
       UNION ALL
       SELECT wallet_address, ts, 'trade' AS kind
         FROM trades
        WHERE ts < $3 AND type IN ('swap', 'perps') AND lower(status) IN ('success', 'filled')
     )
     SELECT m.account_id,
            min(s.ts) FILTER (WHERE s.kind = 'trade') AS first_trade,
            min(s.ts) FILTER (WHERE s.kind = 'deposit') AS first_deposit
       FROM successes s
       JOIN account_wallets m ON m.wallet = CASE WHEN s.wallet_address ~* '^0x[0-9a-f]+$' THEN lower(s.wallet_address) ELSE s.wallet_address END
      GROUP BY m.account_id`,
    [mappings.map(m => m.wallet), mappings.map(m => m.id), toUtc.toISOString()],
  ) as FirstAction[]
}

export function buildDailyJourney(accounts: JourneyAccount[], actions: FirstAction[], startDate: string, endDate: string): DailyJourneyPoint[] {
  nyRangeToUtc(startDate, endDate) // validate before looping
  const days = new Map<string, DailyJourneyPoint>()
  for (let day = startDate; day <= endDate; day = addDays(day, 1)) {
    days.set(day, { day, signups: 0, firstTrades: 0, firstDeposits: 0, signupDayTrades: 0, signupDayDeposits: 0 })
  }
  const byAccount = new Map(accounts.map(a => [a.id, a]))
  for (const account of byAccount.values()) {
    const row = days.get(nyDateOf(new Date(account.createdAt)))
    if (row) row.signups++
  }
  for (const action of actions) {
    const account = byAccount.get(action.account_id)
    if (!account) continue
    const created = new Date(account.createdAt)
    const signupDay = nyDateOf(created)
    for (const [timestamp, total, sameDay] of [
      [action.first_trade, 'firstTrades', 'signupDayTrades'],
      [action.first_deposit, 'firstDeposits', 'signupDayDeposits'],
    ] as const) {
      if (!timestamp) continue
      const at = new Date(timestamp)
      // A linked wallet may have activity predating this account. That is not
      // a new action for this signup; neither invent nor move its first date.
      if (at < created) continue
      const day = nyDateOf(at)
      const row = days.get(day)
      if (row) {
        row[total]++
        if (day === signupDay) row[sameDay]++
      }
    }
  }
  return [...days.values()]
}
