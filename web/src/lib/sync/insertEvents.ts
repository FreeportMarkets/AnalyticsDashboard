import type { neon } from '@neondatabase/serverless'
import type { EventRow } from './types'
import { isDataError, pgErrorMessage } from './pgError'

type SqlTag = ReturnType<typeof neon>

const CHUNK = 500

async function runInsert(sql: SqlTag, chunk: EventRow[]) {
  // `fullResults: true` is REQUIRED. Without it neon resolves to a bare rows
  // array and rowCount is undefined -- the function would then report the
  // ATTEMPTED count rather than the count actually inserted, silently hiding
  // how many rows ON CONFLICT skipped. On a project whose entire premise is
  // data correctness, the sync must not overstate what it wrote.
  //
  // No cast here: the neon driver's own overload narrows the return type to
  // `FullQueryResults` (which has a typed `rowCount: number`) purely from the
  // literal `{ fullResults: true }` passed below. If a future edit drops that
  // option, `result.rowCount` stops type-checking instead of silently
  // becoming `undefined` -> NaN.
  return sql(
    `INSERT INTO events (date, sk, ts, event, screen, component,
                         wallet_address, session_id, platform, app_version, metadata)
     SELECT * FROM UNNEST(
       $1::date[], $2::text[], $3::timestamptz[], $4::text[], $5::text[], $6::text[],
       $7::text[], $8::text[], $9::text[], $10::text[], $11::jsonb[]
     )
     ON CONFLICT (date, sk) DO NOTHING`,
    [
      chunk.map(r => r.date),
      chunk.map(r => r.sk),
      chunk.map(r => r.ts.toISOString()),
      chunk.map(r => r.event),
      chunk.map(r => r.screen),
      chunk.map(r => r.component),
      chunk.map(r => r.wallet_address),
      chunk.map(r => r.session_id),
      chunk.map(r => r.platform),
      chunk.map(r => r.app_version),
      chunk.map(r => (r.metadata === null ? null : JSON.stringify(r.metadata))),
    ],
    { fullResults: true }
  )
}

/**
 * Insert one chunk, bisecting on failure so one poisoned row cannot sink its
 * neighbors.
 *
 * A single `UNNEST` insert is one Postgres statement: if any row in it
 * violates a column constraint (e.g. an out-of-range `date`), the ENTIRE
 * statement rolls back and zero rows are inserted, even the valid ones. A
 * sync loop that doesn't advance its watermark on failure will then re-read
 * the same window forever, hitting the same poisoned row every tick --
 * permanently wedged, from a single crafted event on an unauthenticated
 * ingest endpoint.
 *
 * The fix is binary search over the chunk: split, retry each half, and
 * recurse until the failure narrows to exactly one row, which is quarantined
 * via `onRowFailure` rather than retried again.
 *
 * Only errors classified as Postgres "data errors" (SQLSTATE class 22/23) are
 * bisected. Anything else -- a connection failure, an auth failure, a syntax
 * error -- is rethrown immediately, unmodified, with no bisection and no
 * quarantining: an outage must fail the tick loudly, not be silently
 * absorbed as "every row in the batch was bad".
 */
async function insertChunk(
  sql: SqlTag,
  chunk: EventRow[],
  onRowFailure?: (row: EventRow, reason: string) => Promise<void>
): Promise<number> {
  if (chunk.length === 0) return 0
  try {
    const result = await runInsert(sql, chunk)
    return result.rowCount
  } catch (err) {
    if (!isDataError(err)) throw err

    if (chunk.length === 1) {
      const reason = pgErrorMessage(err)
      if (onRowFailure) await onRowFailure(chunk[0]!, reason)
      return 0
    }

    const mid = Math.ceil(chunk.length / 2)
    const left = await insertChunk(sql, chunk.slice(0, mid), onRowFailure)
    const right = await insertChunk(sql, chunk.slice(mid), onRowFailure)
    return left + right
  }
}

/**
 * Bulk upsert events. ON CONFLICT DO NOTHING -- events are append-only, and a
 * re-read of an already-synced window must be a no-op (idempotency requirement).
 *
 * `onRowFailure`, if given, is invoked once per row that a chunk insert
 * rejects for a data reason (see `insertChunk`), after bisection has isolated
 * it. The returned count is always the number of rows ACTUALLY inserted.
 */
export async function insertEvents(
  sql: SqlTag,
  rows: EventRow[],
  onRowFailure?: (row: EventRow, reason: string) => Promise<void>
): Promise<number> {
  let total = 0
  for (let i = 0; i < rows.length; i += CHUNK) {
    total += await insertChunk(sql, rows.slice(i, i + CHUNK), onRowFailure)
  }
  return total
}
