import type { neon } from '@neondatabase/serverless'

type SqlTag = ReturnType<typeof neon>

export async function quarantineRow(
  sql: SqlTag, source: string, raw: unknown, reason: string
): Promise<void> {
  await sql`
    INSERT INTO quarantine (source, raw, reason)
    VALUES (${source}, ${JSON.stringify(raw)}::jsonb, ${reason})
  `
}
