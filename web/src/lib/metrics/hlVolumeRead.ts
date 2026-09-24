import { sql } from '@/lib/db'
import { fetchHlLedgerMetrics } from '@/lib/hlLedgerApi'

/**
 * Dashboard reads for recorded HL fill volume. The backend flag reads its
 * shared ledger; the default reads pre-aggregated `wallet_volume_daily`.
 * Neither path calls Hyperliquid during a page load.
 *
 * `day` in wallet_volume_daily is already an NY calendar date, so range
 * filters here are plain date comparisons (inclusive), NOT the ts/nyRange
 * dance the raw event/trade tables need.
 */

export interface VolumeTotal {
  notionalUsd: number
  builderFeeUsd: number
  fillCount: number
  source?: 'backend'
  unresolvedBuilderFeeUsd?: number
}

const backendLedgerEnabled = () => process.env.HL_VOLUME_SOURCE === 'backend'

/** Total authoritative volume over an inclusive NY-day range. */
export async function hlVolumeTotal(startDay: string, endDay: string): Promise<VolumeTotal> {
  if (backendLedgerEnabled()) {
    const report = await fetchHlLedgerMetrics(startDay, endDay)
    return {
      notionalUsd: report.days.reduce((sum, day) => sum + Number(day.notionalUsd), 0),
      builderFeeUsd: report.days.reduce((sum, day) => sum + Number(day.confirmedFeeUsd), 0),
      unresolvedBuilderFeeUsd: report.days.reduce((sum, day) => sum + Number(day.unresolvedBuilderFeeUsd), 0),
      fillCount: report.days.reduce((sum, day) => sum + day.fillCount, 0),
      source: 'backend',
    }
  }
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
  if (backendLedgerEnabled()) {
    const report = await fetchHlLedgerMetrics(startDay, endDay)
    return report.days.map(day => ({ day: day.day, notionalUsd: Number(day.notionalUsd), fillCount: day.fillCount }))
  }
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
 * An empty report cannot distinguish a real zero-volume period from missing
 * ingestion, so both empty periods use the labeled estimate path.
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
      // An empty backend response does not prove complete coverage. Until the
      // ledger API exposes a coverage watermark, keep the estimate fallback.
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
