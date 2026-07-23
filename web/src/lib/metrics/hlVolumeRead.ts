import { sql } from '@/lib/db'

/**
 * Dashboard reads for HL builder-fee-authoritative volume. These hit the
 * pre-aggregated `wallet_volume_daily` table (populated by the cron), never
 * HL — so page loads stay fast and never risk HL rate limits.
 *
 * `day` in wallet_volume_daily is already an NY calendar date, so range
 * filters here are plain date comparisons (inclusive), NOT the ts/nyRange
 * dance the raw event/trade tables need.
 */

export interface VolumeTotal {
  notionalUsd: number
  builderFeeUsd: number
  fillCount: number
}

/** Total authoritative volume over an inclusive NY-day range. */
export async function hlVolumeTotal(startDay: string, endDay: string): Promise<VolumeTotal> {
  const rows = (await sql(
    `SELECT
        coalesce(sum(notional_usd), 0)::float8    AS notional,
        coalesce(sum(builder_fee_usd), 0)::float8 AS fee,
        coalesce(sum(fill_count), 0)::int         AS fills
       FROM wallet_volume_daily
      WHERE day >= $1 AND day <= $2`,
    [startDay, endDay]
  )) as Array<{ notional: number; fee: number; fills: number }>
  const r = rows[0]!
  return { notionalUsd: r.notional, builderFeeUsd: r.fee, fillCount: r.fills }
}

/** Per-NY-day volume series over an inclusive range (missing days omitted). */
export async function hlVolumeDaily(
  startDay: string,
  endDay: string
): Promise<Array<{ day: string; notionalUsd: number; fillCount: number }>> {
  const rows = (await sql(
    `SELECT day::text AS day,
            sum(notional_usd)::float8 AS notional,
            sum(fill_count)::int      AS fills
       FROM wallet_volume_daily
      WHERE day >= $1 AND day <= $2
      GROUP BY day
      ORDER BY day`,
    [startDay, endDay]
  )) as Array<{ day: string; notional: number; fills: number }>
  return rows.map(r => ({ day: r.day, notionalUsd: r.notional, fillCount: r.fills }))
}

/**
 * Authoritative volume for a range and its prior period, shaped like the KPI
 * tiles (`{ current, previous }`), plus `hasData` so the page can fall back to
 * the DB reconstruction until the backfill has populated the table.
 *
 * `hasData` is false only when BOTH periods are empty -- a genuinely
 * un-backfilled range -- so a real zero-volume period still reads as
 * authoritative rather than silently reverting to the estimate.
 */
export async function hlVolumeKpi(
  start: string,
  end: string,
  prevStart: string,
  prevEnd: string
): Promise<{ current: number; previous: number; builderFeeUsd: number; hasData: boolean }> {
  try {
    const [cur, prev] = await Promise.all([
      hlVolumeTotal(start, end),
      hlVolumeTotal(prevStart, prevEnd),
    ])
    return {
      current: cur.notionalUsd,
      previous: prev.notionalUsd,
      builderFeeUsd: cur.builderFeeUsd,
      hasData: cur.fillCount > 0 || prev.fillCount > 0,
    }
  } catch {
    // Resilient to the table not existing yet (migration not run) so the
    // Overview page never breaks on the new dependency -- it just falls back
    // to the DB reconstruction until the backfill lands.
    return { current: 0, previous: 0, builderFeeUsd: 0, hasData: false }
  }
}

/** Lifetime authoritative total, and the latest reconciliation ratio. */
export async function hlVolumeLifetime(): Promise<{
  notionalUsd: number
  builderFeeUsd: number
  earliestDay: string | null
  latestDay: string | null
  reconciliationRatio: number | null
}> {
  const totalRows = (await sql`
    SELECT coalesce(sum(notional_usd), 0)::float8    AS notional,
           coalesce(sum(builder_fee_usd), 0)::float8 AS fee,
           min(day)::text                            AS earliest,
           max(day)::text                            AS latest
      FROM wallet_volume_daily
  `) as Array<{ notional: number; fee: number; earliest: string | null; latest: string | null }>
  const t = totalRows[0]!

  const reconRows = (await sql`
    SELECT ratio::float8 AS ratio FROM volume_reconciliation ORDER BY run_at DESC LIMIT 1
  `) as Array<{ ratio: number }>

  return {
    notionalUsd: t.notional,
    builderFeeUsd: t.fee,
    earliestDay: t.earliest,
    latestDay: t.latest,
    reconciliationRatio: reconRows[0]?.ratio ?? null,
  }
}
