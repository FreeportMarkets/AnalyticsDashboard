import type { neon } from '@neondatabase/serverless'
import type { EventRow } from './types'

type SqlTag = ReturnType<typeof neon>

const CHUNK = 500

/**
 * Bulk upsert events. ON CONFLICT DO NOTHING -- events are append-only, and a
 * re-read of an already-synced window must be a no-op (idempotency requirement).
 */
export async function insertEvents(sql: SqlTag, rows: EventRow[]): Promise<number> {
  let total = 0
  for (let i = 0; i < rows.length; i += CHUNK) {
    const chunk = rows.slice(i, i + CHUNK)
    // `fullResults: true` is REQUIRED. Without it neon resolves to a bare rows
    // array and rowCount is undefined -- the function would then report the
    // ATTEMPTED count rather than the count actually inserted, silently hiding
    // how many rows ON CONFLICT skipped. On a project whose entire premise is
    // data correctness, the sync must not overstate what it wrote.
    const result = (await sql(
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
    )) as unknown as { rowCount: number }
    total += result.rowCount
  }
  return total
}
