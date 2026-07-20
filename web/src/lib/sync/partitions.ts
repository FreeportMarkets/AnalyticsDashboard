const DATE_RE = /^\d{4}-\d{2}-\d{2}$/

function assertDate(date: string): void {
  // Partition names and bounds go into DDL, which cannot be parameterized.
  // Validate strictly so nothing but a literal date can ever reach the string.
  if (!DATE_RE.test(date)) throw new Error(`invalid date: ${date}`)
  const d = new Date(`${date}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) throw new Error(`invalid date: ${date}`)
}

export function partitionNameFor(date: string): string {
  assertDate(date)
  return `events_${date.slice(0, 4)}_${date.slice(5, 7)}`
}

export function partitionBoundsFor(date: string): { from: string; to: string } {
  assertDate(date)
  const year = Number(date.slice(0, 4))
  const month = Number(date.slice(5, 7))
  const from = `${year}-${String(month).padStart(2, '0')}-01`
  const nextYear = month === 12 ? year + 1 : year
  const nextMonth = month === 12 ? 1 : month + 1
  const to = `${nextYear}-${String(nextMonth).padStart(2, '0')}-01`
  return { from, to }
}

import type { neon } from '@neondatabase/serverless'

type SqlTag = ReturnType<typeof neon>

/**
 * Create any missing monthly partitions plus their indexes.
 * Idempotent -- safe to call on every sync tick.
 *
 * NOTE ON THE NEON API: `@neondatabase/serverless` has NO `sql.query()` method.
 * The function returned by `neon()` is called either as a tagged template
 * (sql`...`) or directly as sql(text, params, opts). Each call sends exactly ONE
 * statement over HTTP -- multi-statement strings are rejected. Every raw query in
 * this codebase therefore uses the sql(text, params) form, one statement per call.
 */
export async function ensurePartitions(sql: SqlTag, dates: string[]): Promise<string[]> {
  const names = new Set<string>()
  for (const d of dates) names.add(partitionNameFor(d))

  const created: string[] = []
  for (const name of names) {
    const date = `${name.slice(7, 11)}-${name.slice(12, 14)}-01`
    const { from, to } = partitionBoundsFor(date)
    // DDL cannot be parameterized; `name`, `from`, and `to` are all derived from
    // a string already validated against DATE_RE by assertDate().
    await sql(
      `CREATE TABLE IF NOT EXISTS ${name} PARTITION OF events
       FOR VALUES FROM ('${from}') TO ('${to}')`
    )
    await sql(`CREATE INDEX IF NOT EXISTS ${name}_event_ts_idx   ON ${name} (event, ts)`)
    await sql(`CREATE INDEX IF NOT EXISTS ${name}_wallet_ts_idx  ON ${name} (wallet_address, ts)`)
    await sql(`CREATE INDEX IF NOT EXISTS ${name}_session_ts_idx ON ${name} (session_id, ts)`)
    created.push(name)
  }
  return created
}
