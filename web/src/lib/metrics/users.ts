import { sql } from '@/lib/db'
import { nyDateExpr } from '@/lib/time'
import { nyRangeToUtc } from './nyRange'
import { VOLUME_USD_EXPR } from './trades'

/**
 * Ported from app.py `with tab_users:` (~1695-1863) and `with tab_retention:`
 * (~1864-1950), plus the `extract_session_durations` helper (~1056-1090).
 *
 * Same two rules as queries.ts/overview.ts: bucket with nyDateExpr('ts'),
 * filter `ts` with nyRangeToUtc bounds. Never GROUP BY or range-filter on
 * `date` -- scripts/lint-no-date-grouping.ts enforces this.
 *
 * System-wallet exclusion mirrors app.py's `user_df` filter (SYSTEM_WALLETS
 * set + platform != 'server', app.py:1126-1134) and is applied everywhere
 * here -- every function in this file operates on the same "real user"
 * population app.py's tab_users/tab_retention did, including the activity
 * heatmap (app.py's heatmap reads from user_df, which is already filtered).
 *
 * "First-ever event" for cohorts/new-vs-returning is scoped to the queried
 * [startDate, endDate] window, not lifetime history -- this matches app.py:
 * `load_events_range(start_str, end_str)` only ever loads that window, so
 * `user_df["date"].min()` (app.py:1868, :1465) is a first-seen-in-window,
 * not a true lifetime first-seen. Porting anything else would silently
 * change what the numbers mean.
 */
const SYSTEM_WALLETS = ['server', 'unknown', 'system', '']
const RETENTION_DAYS = [0, 1, 3, 7, 14, 30] as const
const RETENTION_OFFSETS_SQL = '(VALUES (0),(1),(3),(7),(14),(30)) AS o(offset_day)'

function addDays(date: string, days: number): string {
  const d = new Date(`${date}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

/** Every NY calendar date from startDate to endDate, inclusive. */
function dateRange(startDate: string, endDate: string): string[] {
  const out: string[] = []
  let cur = startDate
  while (cur <= endDate) {
    out.push(cur)
    cur = addDays(cur, 1)
  }
  return out
}

function dayKey(d: string | Date): string {
  return typeof d === 'string' ? d : d.toISOString().slice(0, 10)
}

// --- Active users (DAU/WAU/MAU) ---

export interface ActiveUsersResult {
  daily: Array<{ day: string; users: number }>
  /** Users active on the last day of [startDate, endDate]. */
  dau: number
  /** Users active in the trailing 7 NY days ending endDate (independent of startDate). */
  wau: number
  /** Users active in the trailing 30 NY days ending endDate (independent of startDate). */
  mau: number
}

export async function activeUsers(startDate: string, endDate: string): Promise<ActiveUsersResult> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const dailyRows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day,
            count(DISTINCT wallet_address) FILTER (
              WHERE wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
                AND (platform IS NULL OR platform <> 'server')
            )::int AS users
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; users: number }>

  const byDay = new Map(dailyRows.map(r => [dayKey(r.day), r.users]))
  const daily = dateRange(startDate, endDate).map(day => ({ day, users: byDay.get(day) ?? 0 }))

  const { fromUtc: mauFrom } = nyRangeToUtc(addDays(endDate, -29), endDate)
  const { fromUtc: wauFrom } = nyRangeToUtc(addDays(endDate, -6), endDate)

  const windowRows = (await sql(
    `SELECT
        count(DISTINCT wallet_address) FILTER (WHERE ts >= $2)::int AS wau,
        count(DISTINCT wallet_address)::int AS mau
       FROM events
      WHERE ts >= $1 AND ts < $3
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($4::text[])
        AND (platform IS NULL OR platform <> 'server')`,
    [mauFrom.toISOString(), wauFrom.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ wau: number; mau: number }>

  const w = windowRows[0]
  if (!w) throw new Error('activeUsers: window query returned no rows')

  return {
    daily,
    dau: daily.length > 0 ? daily[daily.length - 1]!.users : 0,
    wau: w.wau,
    mau: w.mau,
  }
}

// --- Session stats ---

export interface SessionStats {
  sessionCount: number
  distinctUsers: number
  avgSessionsPerUser: number
  medianDurationMin: number
  p90DurationMin: number
  daily: Array<{ day: string; totalMinutes: number }>
}

/**
 * Session duration comes from `session_end` events carrying
 * `metadata->>'duration_ms'` (app.py `extract_session_durations`). Only
 * positive numeric values count -- `jsonb_typeof(...) = 'number'` mirrors
 * app.py's `isinstance(d, (int, float))` guard, and `> 0` mirrors `d > 0`.
 */
export async function sessionStats(startDate: string, endDate: string): Promise<SessionStats> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)

  const aggRows = (await sql(
    `WITH s AS (
        SELECT wallet_address, session_id, (metadata->>'duration_ms')::numeric AS duration_ms
          FROM events
         WHERE event = 'session_end'
           AND ts >= $1 AND ts < $2
           AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
           AND (platform IS NULL OR platform <> 'server')
           AND jsonb_typeof(metadata -> 'duration_ms') = 'number'
           AND (metadata->>'duration_ms')::numeric > 0
     )
     SELECT
        count(*)::int AS session_count,
        count(DISTINCT wallet_address)::int AS distinct_users,
        coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms), 0)::float8 AS median_duration_ms,
        coalesce(percentile_cont(0.9) WITHIN GROUP (ORDER BY duration_ms), 0)::float8 AS p90_duration_ms
       FROM s`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{
    session_count: number
    distinct_users: number
    median_duration_ms: number
    p90_duration_ms: number
  }>

  const dailyRows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day,
            coalesce(sum((metadata->>'duration_ms')::numeric), 0)::float8 AS total_duration_ms
       FROM events
      WHERE event = 'session_end'
        AND ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
        AND (platform IS NULL OR platform <> 'server')
        AND jsonb_typeof(metadata -> 'duration_ms') = 'number'
        AND (metadata->>'duration_ms')::numeric > 0
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; total_duration_ms: number }>

  const byDay = new Map(dailyRows.map(r => [dayKey(r.day), r.total_duration_ms]))
  const daily = dateRange(startDate, endDate).map(day => ({
    day,
    totalMinutes: Math.round(((byDay.get(day) ?? 0) / 60000) * 10) / 10,
  }))

  const a = aggRows[0]
  if (!a) throw new Error('sessionStats: aggregate query returned no rows')

  return {
    sessionCount: a.session_count,
    distinctUsers: a.distinct_users,
    avgSessionsPerUser: a.distinct_users > 0 ? a.session_count / a.distinct_users : 0,
    medianDurationMin: a.median_duration_ms / 60000,
    p90DurationMin: a.p90_duration_ms / 60000,
    daily,
  }
}

// --- Top users by activity ---

export interface TopUser {
  wallet: string
  events: number
  sessions: number
  lastSeen: string
}

export async function topUsersByActivity(
  startDate: string,
  endDate: string,
  limit = 20
): Promise<TopUser[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT wallet_address,
            count(*)::int AS events,
            count(DISTINCT session_id)::int AS sessions,
            max(ts) AS last_seen
       FROM events
      WHERE ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
        AND (platform IS NULL OR platform <> 'server')
      GROUP BY 1
      ORDER BY events DESC
      LIMIT $4`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS, limit]
  )) as Array<{ wallet_address: string; events: number; sessions: number; last_seen: string | Date }>

  return rows.map(r => ({
    wallet: r.wallet_address,
    events: r.events,
    sessions: r.sessions,
    lastSeen: typeof r.last_seen === 'string' ? r.last_seen : r.last_seen.toISOString(),
  }))
}

// --- Top traders ---

export interface TopTrader {
  wallet: string
  trades: number
  volumeUsd: number
}

/**
 * Mirrors app.py's "Top Traders" section, which excludes type == 'deposit'
 * (app.py:1760). Volume sums `VOLUME_USD_EXPR` (imported from trades.ts),
 * never raw `amount_usd` -- `amount_usd` on a perps row is MARGIN, not
 * notional (same trap documented in trades.ts). Scoped to
 * `type IN ('swap', 'perps')`, matching trades.ts/overview.ts exactly,
 * rather than the looser `type <> 'deposit'` (equivalent today, but explicit
 * about the intended trade-type set).
 */
export async function topTraders(
  startDate: string,
  endDate: string,
  limit = 20
): Promise<TopTrader[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT wallet_address,
            count(*)::int AS trades,
            coalesce(sum(${VOLUME_USD_EXPR}), 0)::float8 AS volume_usd
       FROM trades
      WHERE ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
        AND type IN ('swap', 'perps')
      GROUP BY 1
      ORDER BY volume_usd DESC
      LIMIT $4`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS, limit]
  )) as Array<{ wallet_address: string; trades: number; volume_usd: number }>

  return rows.map(r => ({ wallet: r.wallet_address, trades: r.trades, volumeUsd: r.volume_usd }))
}

// --- Activity heatmap ---

export interface HeatmapCell {
  /** 0 = Sunday, matching app.py's dow_labels ["Sun", "Mon", ...] (app.py:1791). */
  dayOfWeek: number
  hour: number
  count: number
}

export async function activityHeatmap(startDate: string, endDate: string): Promise<HeatmapCell[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT extract(dow FROM (ts AT TIME ZONE 'America/New_York'))::int AS day_of_week,
            extract(hour FROM (ts AT TIME ZONE 'America/New_York'))::int AS hour,
            count(*)::int AS count
       FROM events
      WHERE ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
        AND (platform IS NULL OR platform <> 'server')
      GROUP BY 1, 2`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day_of_week: number; hour: number; count: number }>

  const byKey = new Map(rows.map(r => [`${r.day_of_week}-${r.hour}`, r.count]))
  const cells: HeatmapCell[] = []
  for (let dow = 0; dow < 7; dow++) {
    for (let hour = 0; hour < 24; hour++) {
      cells.push({ dayOfWeek: dow, hour, count: byKey.get(`${dow}-${hour}`) ?? 0 })
    }
  }
  return cells
}

// --- New vs returning ---

export interface NewVsReturningRow {
  day: string
  newUsers: number
  returningUsers: number
}

/**
 * Daily split of users whose first-ever event (within [startDate, endDate])
 * falls on that day (new) vs. an earlier day in the same window (returning).
 */
export async function newVsReturning(startDate: string, endDate: string): Promise<NewVsReturningRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `WITH range_events AS (
        SELECT wallet_address, ${nyDateExpr('ts')} AS day
          FROM events
         WHERE ts >= $1 AND ts < $2
           AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
           AND (platform IS NULL OR platform <> 'server')
     ),
     first_seen AS (
        SELECT wallet_address, min(day) AS first_day FROM range_events GROUP BY 1
     ),
     daily_active AS (
        SELECT DISTINCT wallet_address, day FROM range_events
     )
     SELECT da.day,
            count(*) FILTER (WHERE fs.first_day = da.day)::int AS new_users,
            count(*) FILTER (WHERE fs.first_day < da.day)::int AS returning_users
       FROM daily_active da
       JOIN first_seen fs ON fs.wallet_address = da.wallet_address
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; new_users: number; returning_users: number }>

  const byDay = new Map(rows.map(r => [dayKey(r.day), r]))
  return dateRange(startDate, endDate).map(day => ({
    day,
    newUsers: byDay.get(day)?.new_users ?? 0,
    returningUsers: byDay.get(day)?.returning_users ?? 0,
  }))
}

// --- Retention ---

export interface RetentionCurvePoint {
  offsetDay: number
  label: string
  avgPct: number
}

/**
 * Average, unweighted-by-cohort-size retention curve across all cohorts in
 * range, matching app.py's `avg_ret[d] = sum(vals) / len(vals)` (app.py:1902-1905)
 * -- each cohort's D-day retention percentage counts equally, not weighted
 * by cohort size.
 */
export async function retentionCurve(startDate: string, endDate: string): Promise<RetentionCurvePoint[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `WITH range_events AS (
        SELECT wallet_address, ${nyDateExpr('ts')} AS day
          FROM events
         WHERE ts >= $1 AND ts < $2
           AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
           AND (platform IS NULL OR platform <> 'server')
     ),
     cohorts AS (
        SELECT wallet_address, min(day) AS cohort_day FROM range_events GROUP BY 1
     ),
     cohort_sizes AS (
        SELECT cohort_day, count(*)::int AS size FROM cohorts GROUP BY 1
     ),
     active_days AS (
        SELECT DISTINCT wallet_address, day FROM range_events
     ),
     per_cohort AS (
        SELECT c.cohort_day, cs.size AS cohort_size, o.offset_day,
               count(DISTINCT a.wallet_address)::int AS retained
          FROM cohorts c
          JOIN cohort_sizes cs ON cs.cohort_day = c.cohort_day
          CROSS JOIN ${RETENTION_OFFSETS_SQL}
          LEFT JOIN active_days a
            ON a.wallet_address = c.wallet_address
           AND a.day = c.cohort_day + o.offset_day
         GROUP BY 1, 2, 3
     )
     SELECT offset_day,
            avg(retained::float8 / NULLIF(cohort_size, 0) * 100)::float8 AS avg_pct
       FROM per_cohort
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ offset_day: number; avg_pct: number | null }>

  const byOffset = new Map(rows.map(r => [r.offset_day, r.avg_pct ?? 0]))
  return RETENTION_DAYS.map(d => ({
    offsetDay: d,
    label: `D${d}`,
    avgPct: byOffset.get(d) ?? 0,
  }))
}

export interface CohortRetentionCell {
  offsetDay: number
  retained: number
  pct: number
}

export interface CohortRetentionRow {
  cohortDate: string
  cohortSize: number
  cells: CohortRetentionCell[]
}

/** Cohort (first-seen NY day, within range) x day-offset grid of retention percentages. */
export async function cohortRetention(startDate: string, endDate: string): Promise<CohortRetentionRow[]> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `WITH range_events AS (
        SELECT wallet_address, ${nyDateExpr('ts')} AS day
          FROM events
         WHERE ts >= $1 AND ts < $2
           AND wallet_address IS NOT NULL AND wallet_address <> ALL($3::text[])
           AND (platform IS NULL OR platform <> 'server')
     ),
     cohorts AS (
        SELECT wallet_address, min(day) AS cohort_day FROM range_events GROUP BY 1
     ),
     cohort_sizes AS (
        SELECT cohort_day, count(*)::int AS size FROM cohorts GROUP BY 1
     ),
     active_days AS (
        SELECT DISTINCT wallet_address, day FROM range_events
     )
     SELECT c.cohort_day::text AS cohort_day,
            cs.size AS cohort_size,
            o.offset_day,
            count(DISTINCT a.wallet_address)::int AS retained
       FROM cohorts c
       JOIN cohort_sizes cs ON cs.cohort_day = c.cohort_day
       CROSS JOIN ${RETENTION_OFFSETS_SQL}
       LEFT JOIN active_days a
         ON a.wallet_address = c.wallet_address
        AND a.day = c.cohort_day + o.offset_day
      GROUP BY 1, 2, 3
      ORDER BY 1, 3`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ cohort_day: string; cohort_size: number; offset_day: number; retained: number }>

  const byCohort = new Map<string, CohortRetentionRow>()
  for (const r of rows) {
    let entry = byCohort.get(r.cohort_day)
    if (!entry) {
      entry = { cohortDate: r.cohort_day, cohortSize: r.cohort_size, cells: [] }
      byCohort.set(r.cohort_day, entry)
    }
    entry.cells.push({
      offsetDay: r.offset_day,
      retained: r.retained,
      pct: r.cohort_size > 0 ? (r.retained / r.cohort_size) * 100 : 0,
    })
  }
  return Array.from(byCohort.values()).sort((a, b) => a.cohortDate.localeCompare(b.cohortDate))
}
