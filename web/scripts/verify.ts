/**
 * Verify the Postgres mirror against DynamoDB, per calendar date, for both
 * sources (`events` keyed on the analytics table's `date` partition key,
 * `trades` keyed on the trades table's `trade_date-timestamp-index` GSI --
 * the same GSI `fetchTrades`/`backfill.ts` read through, so a mismatch here
 * means the mirror, not the counting method, disagrees with the source of
 * truth).
 *
 * This is the evidence gate for Phase 1, not a status report: it exits
 * non-zero if ANY date mismatches for either source, or if `events_default`
 * (the partition backstop -- see db/migrations/0001_init.sql) is non-empty.
 * A non-empty default partition means a row landed outside every declared
 * monthly range, which permanently blocks creating that month's partition
 * ("updated partition constraint ... would be violated") -- see
 * scripts/backfill.ts's identical rationale. Printing a warning for that
 * condition would not be a gate, so it is a hard failure here.
 *
 * DynamoDB's `Select: COUNT` response paginates exactly like an `Items`
 * response -- `Count` on any one page is only that page's count, not the
 * partition total. Accumulating across `LastEvaluatedKey` pages is required;
 * a partial count that happens to equal the Postgres count would be a false
 * pass, which is exactly the silent-failure class this script exists to
 * rule out.
 *
 * Usage: tsx scripts/verify.ts <start YYYY-MM-DD> <end YYYY-MM-DD>
 */
import { QueryCommand } from '@aws-sdk/lib-dynamodb'
import { sql } from '../src/lib/db'
import { ddb, ANALYTICS_TABLE, TRADES_TABLE } from '../src/lib/ddb'
import { isValidCalendarDate } from '../src/lib/time'

const TRADES_DATE_INDEX = 'trade_date-timestamp-index'

interface DdbCountParams {
  table: string
  keyName: string
  value: string
  indexName?: string
}

/** Accumulate `Count` across every page of a `Select: COUNT` query. */
async function ddbCount(params: DdbCountParams): Promise<number> {
  let total = 0
  let last: Record<string, unknown> | undefined
  do {
    const resp = await ddb.send(
      new QueryCommand({
        TableName: params.table,
        IndexName: params.indexName,
        KeyConditionExpression: '#k = :v',
        ExpressionAttributeNames: { '#k': params.keyName },
        ExpressionAttributeValues: { ':v': params.value },
        Select: 'COUNT',
        ExclusiveStartKey: last,
      })
    )
    total += resp.Count ?? 0
    last = resp.LastEvaluatedKey
  } while (last)
  return total
}

function eachDate(start: string, end: string): string[] {
  const out: string[] = []
  const cur = new Date(`${start}T00:00:00Z`)
  const stop = new Date(`${end}T00:00:00Z`)
  while (cur <= stop) {
    out.push(cur.toISOString().slice(0, 10))
    cur.setUTCDate(cur.getUTCDate() + 1)
  }
  return out
}

interface DateCheck {
  date: string
  ddb: number
  pg: number
  ok: boolean
}

async function checkEventsDate(date: string): Promise<DateCheck> {
  const ddbN = await ddbCount({ table: ANALYTICS_TABLE, keyName: 'date', value: date })
  const [row] = (await sql`
    SELECT count(*)::int AS n FROM events WHERE date = ${date}
  `) as Array<{ n: number }>
  const pgN = row?.n ?? 0
  return { date, ddb: ddbN, pg: pgN, ok: ddbN === pgN }
}

/**
 * Rows with no `trade_date` (absent on the DynamoDB item, or -- pre-mapping
 * -- rejected as malformed by `mapTrade`) are invisible to BOTH sides of
 * this comparison: they never appear in the GSI query (the GSI is only
 * populated for items that carry the key attribute) and they land with
 * `trade_date = NULL` in Postgres, which `WHERE trade_date = date` never
 * matches. That symmetry is what makes per-date trade counts meaningful
 * despite `trade_date` being nullable, not a gap this check misses.
 */
async function checkTradesDate(date: string): Promise<DateCheck> {
  const ddbN = await ddbCount({
    table: TRADES_TABLE,
    indexName: TRADES_DATE_INDEX,
    keyName: 'trade_date',
    value: date,
  })
  const [row] = (await sql`
    SELECT count(*)::int AS n FROM trades WHERE trade_date = ${date}
  `) as Array<{ n: number }>
  const pgN = row?.n ?? 0
  return { date, ddb: ddbN, pg: pgN, ok: ddbN === pgN }
}

function fmt(source: string, c: DateCheck): string {
  return `${source} ddb=${c.ddb} pg=${c.pg} ${c.ok ? 'OK' : 'MISMATCH'}`
}

async function main() {
  const [start, end] = process.argv.slice(2)
  if (!start || !end) throw new Error('usage: verify.ts <start YYYY-MM-DD> <end YYYY-MM-DD>')
  if (!isValidCalendarDate(start) || !isValidCalendarDate(end)) {
    throw new Error(`invalid date range: ${start}..${end}`)
  }

  const dates = eachDate(start, end)
  let failures = 0

  for (const date of dates) {
    const events = await checkEventsDate(date)
    const trades = await checkTradesDate(date)
    if (!events.ok) failures++
    if (!trades.ok) failures++
    console.log(`${date}  ${fmt('events', events)}  ${fmt('trades', trades)}`)
  }

  const [q] = (await sql`SELECT count(*)::int AS n FROM quarantine`) as Array<{ n: number }>
  const [d] = (await sql`SELECT count(*)::int AS n FROM events_default`) as Array<{ n: number }>
  const quarantineN = q?.n ?? 0
  const defaultN = d?.n ?? 0

  console.log(`quarantine rows: ${quarantineN}`)
  console.log(`events_default rows: ${defaultN} (must be 0)`)

  if (failures > 0 || defaultN > 0) {
    console.error(
      `FAILED: ${failures} date/source mismatch(es)` +
        (defaultN > 0 ? `, events_default has ${defaultN} row(s)` : '')
    )
    process.exit(1)
  }

  console.log('verification passed')
}

main().catch(e => {
  console.error(e)
  process.exit(1)
})
