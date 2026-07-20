import type { neon } from '@neondatabase/serverless'
import { createHash } from 'node:crypto'

type SqlTag = ReturnType<typeof neon>

/**
 * Quarantine one rejected row.
 *
 * `item_hash` dedups on (source, item_hash) so a permanently-wedged window
 * doesn't grow the quarantine table without bound: without this, the same
 * poisoned row would be re-inserted every sync tick (every 60s) forever,
 * which also violates the project's idempotency requirement -- re-running a
 * sync over any range must not change the resulting data.
 */
export async function quarantineRow(
  sql: SqlTag, source: string, raw: unknown, reason: string
): Promise<void> {
  const rawJson = JSON.stringify(raw)
  const itemHash = createHash('sha256').update(`${rawJson}|${reason}`).digest('hex')
  await sql`
    INSERT INTO quarantine (source, raw, reason, item_hash)
    VALUES (${source}, ${rawJson}::jsonb, ${reason}, ${itemHash})
    ON CONFLICT (source, item_hash) DO NOTHING
  `
}
