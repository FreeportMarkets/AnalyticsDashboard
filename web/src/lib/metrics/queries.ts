import { sql } from '@/lib/db'
import { nyDateExpr } from '@/lib/time'
import { nyRangeToUtc } from './nyRange'

/**
 * Every metric in this file follows the same two rules:
 *   1. Filter on `ts` using UTC instant bounds derived from the NY calendar range.
 *   2. Group with nyDateExpr('ts'), never with the `date` column.
 * The `date` column is the DynamoDB partition key, derived in UTC. Grouping on it
 * shifts daily figures by up to 5 hours and the result still looks plausible.
 * scripts/lint-no-date-grouping.ts enforces this mechanically.
 */

const SYSTEM_WALLETS = ['server', 'unknown', 'system', '']

export async function dailyEventCounts(
  startDate: string, endDate: string
): Promise<Array<{ day: string; count: number }>> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day, count(*)::int AS count
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ day: string | Date; count: number }>
  return rows.map(r => ({
    day: typeof r.day === 'string' ? r.day : r.day.toISOString().slice(0, 10),
    count: r.count,
  }))
}

export async function dailyActiveUsers(
  startDate: string, endDate: string
): Promise<Array<{ day: string; users: number }>> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day, count(DISTINCT wallet_address)::int AS users
       FROM events
      WHERE ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL
        AND wallet_address <> ALL($3::text[])
        AND (platform IS NULL OR platform <> 'server')
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; users: number }>
  return rows.map(r => ({
    day: typeof r.day === 'string' ? r.day : r.day.toISOString().slice(0, 10),
    users: r.users,
  }))
}
