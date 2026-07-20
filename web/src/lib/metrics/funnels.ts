import { sql } from '@/lib/db'
import { nyRangeToUtc } from './nyRange'

/**
 * Composable funnels: any ordered sequence of event names, with a
 * conversion window between consecutive steps and an optional breakdown
 * dimension. Replaces the Streamlit original's three hardcoded funnels
 * (Notification -> Trade, Trade Funnel, Deposit Funnel — app.py lines
 * ~1959-2022) with one general primitive; those three are ported as
 * preset step lists in the page, not baked in here.
 *
 * Same two rules as every other metrics module: filter `ts` with
 * nyRangeToUtc bounds, never range-filter or group on the `date` column.
 * scripts/lint-no-date-grouping.ts enforces this.
 *
 * Funnel semantics: a wallet counts at step N only if it performed steps
 * 1..N in order, each within `windowSeconds` of the previous step's
 * qualifying event. Users are counted as DISTINCT wallets per step, never
 * raw event rows. Everything is computed in one SQL statement per query
 * (a chain of CTEs, each an aggregate over the previous step's wallets) --
 * with ~1.5M events, fetching rows and reducing in JS is not an option.
 */

const SYSTEM_WALLETS = ['server', 'unknown', 'system', '']

const MIN_STEPS = 2
const MAX_STEPS = 6

export interface EventCount {
  event: string
  count: number
}

/** Distinct event names present in the range, for populating a step picker. */
export async function availableEvents(startDate: string, endDate: string): Promise<EventCount[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT event, count(*)::int AS count
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY count DESC, event ASC`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as EventCount[]
  return rows
}

export type BreakdownDim = 'platform' | 'app_version'

export interface FunnelStep {
  event: string
  users: number
  /** Fraction (0..1) converted from the immediately preceding step. 1 for step 0. */
  convFromPrev: number
  /** Fraction (0..1) converted from step 0. */
  convFromStart: number
}

export interface FunnelBreakdownRow {
  event: string
  dimValue: string
  users: number
}

export interface FunnelResult {
  steps: FunnelStep[]
  breakdown: FunnelBreakdownRow[] | null
}

/**
 * Compute an ordered funnel over `steps` (event names, in required order).
 *
 * Builds a chain of CTEs: step0 finds each wallet's earliest occurrence of
 * steps[0] in range; step_i finds each wallet's earliest occurrence of
 * steps[i] that comes after step_{i-1}'s qualifying event and within
 * `windowSeconds` of it, restricted to wallets that already qualified for
 * step_{i-1}. A wallet surviving to step_i's CTE has, by construction,
 * performed steps 0..i in order within the window at every hop.
 *
 * `steps` must already be validated against real event names present in
 * the range (see availableEvents) -- this function still only ever
 * interpolates step names as bound parameters, never string-concatenated,
 * but callers should reject unknown event names before calling so results
 * aren't silently all-zero from a typo.
 */
export async function computeFunnel(
  startDate: string,
  endDate: string,
  steps: string[],
  windowSeconds: number,
  breakdownBy?: BreakdownDim
): Promise<FunnelResult> {
  if (steps.length < MIN_STEPS || steps.length > MAX_STEPS) {
    throw new Error(`computeFunnel: steps must have between ${MIN_STEPS} and ${MAX_STEPS} entries, got ${steps.length}`)
  }
  if (!Number.isFinite(windowSeconds) || windowSeconds <= 0) {
    throw new Error(`computeFunnel: windowSeconds must be a positive number, got ${windowSeconds}`)
  }
  // breakdownBy is only ever read from a literal union type by callers (see
  // isBreakdownKey in the page), but guard here too since it is interpolated
  // as a bare column name below rather than a bound parameter.
  if (breakdownBy !== undefined && breakdownBy !== 'platform' && breakdownBy !== 'app_version') {
    throw new Error(`computeFunnel: invalid breakdownBy: ${String(breakdownBy)}`)
  }

  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  // Param layout: $1 fromUtc, $2 toUtc, $3 system wallets, $4 window
  // seconds, $5.. one per step event name in order. Step names are always
  // passed as bound parameters -- never string-concatenated into the SQL.
  const params: Array<string | string[] | number> = [
    fromUtc.toISOString(),
    toUtc.toISOString(),
    SYSTEM_WALLETS,
    windowSeconds,
  ]
  const eventParamIdx = steps.map(event => {
    params.push(event)
    return params.length
  })

  const dimSelect = breakdownBy ? `(array_agg(${breakdownBy} ORDER BY ts))[1]` : 'NULL::text'

  const cteParts: string[] = [
    `step0 AS (
      SELECT wallet_address,
             min(ts) AS ts0,
             ${dimSelect} AS dim
        FROM events
       WHERE event = $${eventParamIdx[0]}
         AND ts >= $1 AND ts < $2
         AND wallet_address IS NOT NULL
         AND wallet_address <> ALL($3::text[])
         AND (platform IS NULL OR platform <> 'server')
       GROUP BY wallet_address
    )`,
  ]

  for (let i = 1; i < steps.length; i++) {
    cteParts.push(
      `step${i} AS (
        SELECT e.wallet_address,
               min(e.ts) AS ts${i},
               prev.dim AS dim
          FROM events e
          JOIN step${i - 1} prev ON prev.wallet_address = e.wallet_address
         WHERE e.event = $${eventParamIdx[i]}
           AND e.ts > prev.ts${i - 1}
           AND e.ts <= prev.ts${i - 1} + ($4::numeric * INTERVAL '1 second')
           AND e.ts >= $1 AND e.ts < $2
           AND e.wallet_address <> ALL($3::text[])
           AND (e.platform IS NULL OR e.platform <> 'server')
         GROUP BY e.wallet_address, prev.dim
      )`
    )
  }

  const withClause = `WITH ${cteParts.join(',\n')}`
  const unionSql = steps
    .map((_, i) => `SELECT ${i} AS step_idx, wallet_address, dim FROM step${i}`)
    .join('\n UNION ALL \n')

  const totalsRows = (await sql(
    `${withClause}
     SELECT step_idx, count(*)::int AS users
       FROM (${unionSql}) t
      GROUP BY step_idx
      ORDER BY step_idx`,
    params
  )) as Array<{ step_idx: number; users: number }>

  const usersByIdx = new Map(totalsRows.map(r => [r.step_idx, r.users]))
  const startUsers = usersByIdx.get(0) ?? 0

  const stepResults: FunnelStep[] = steps.map((event, i) => {
    const users = usersByIdx.get(i) ?? 0
    const prevUsers = i === 0 ? users : usersByIdx.get(i - 1) ?? 0
    return {
      event,
      users,
      convFromPrev: i === 0 ? 1 : prevUsers > 0 ? users / prevUsers : 0,
      convFromStart: startUsers > 0 ? users / startUsers : 0,
    }
  })

  let breakdown: FunnelBreakdownRow[] | null = null
  if (breakdownBy) {
    const breakdownRows = (await sql(
      `${withClause}
       SELECT step_idx, coalesce(dim, 'unknown') AS dim_value, count(*)::int AS users
         FROM (${unionSql}) t
        GROUP BY step_idx, dim_value
        ORDER BY step_idx, users DESC`,
      params
    )) as Array<{ step_idx: number; dim_value: string; users: number }>
    breakdown = breakdownRows
      .filter(r => r.step_idx >= 0 && r.step_idx < steps.length)
      .map(r => ({ event: steps[r.step_idx] as string, dimValue: r.dim_value, users: r.users }))
  }

  return { steps: stepResults, breakdown }
}

export interface FeatureEngagementRow {
  event: string
  total: number
  users: number
}

/**
 * Event usage by distinct users, across every real event in the range --
 * generalizes the Streamlit original's hardcoded ~10-event allowlist
 * (app.py ~2028-2032, several of which -- search_query, category_select,
 * bridge_transfer -- aren't in this dataset's actual event names) to
 * whatever events actually occurred.
 */
export async function featureEngagement(startDate: string, endDate: string): Promise<FeatureEngagementRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT event,
            count(*)::int AS total,
            count(DISTINCT wallet_address) FILTER (
              WHERE wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
                AND (platform IS NULL OR platform <> 'server')
            )::int AS users
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY total DESC`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as FeatureEngagementRow[]
  return rows
}
