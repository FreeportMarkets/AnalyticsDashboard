import { neon } from '@neondatabase/serverless'
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')
const sql = neon(url)

/**
 * Split a migration file into individual statements.
 *
 * Neon's HTTP driver executes exactly ONE statement per call and rejects
 * multi-statement strings, so a migration file cannot be sent as a single query.
 *
 * This splitter is deliberately simple: it strips `--` line comments and splits
 * on semicolons. That is sufficient for plain DDL and is all this project's
 * migrations contain. If a future migration introduces a function body, a
 * dollar-quoted string, a semicolon inside a string literal, or a `--` sequence
 * inside a string literal, this splitter MUST be replaced with a real parser --
 * it will silently split such a file in the wrong place.
 */
export function splitStatements(sqlText: string): string[] {
  return sqlText
    .split('\n')
    .map(line => line.replace(/--.*$/, ''))
    .join('\n')
    .split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0)
}

async function main() {
  await sql`CREATE TABLE IF NOT EXISTS _migrations (
    name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now()
  )`
  const applied = new Set(
    (await sql`SELECT name FROM _migrations`).map((r: any) => r.name as string)
  )
  const dir = path.join(import.meta.dirname, 'migrations')
  for (const file of readdirSync(dir).filter(f => f.endsWith('.sql')).sort()) {
    if (applied.has(file)) {
      console.log(`skip ${file}`)
      continue
    }
    console.log(`apply ${file}`)
    // Design tradeoff: each statement is committed independently via separate HTTP calls.
    // Migration files are therefore not atomic — if a statement fails midway, previous
    // statements remain applied and the file is never recorded in _migrations. This is
    // acceptable only while every statement in the migration is idempotent (IF NOT EXISTS).
    // A future migration containing a non-idempotent statement such as ALTER TABLE or a
    // data backfill must switch this loop to sql.transaction() for atomicity.
    for (const stmt of splitStatements(readFileSync(path.join(dir, file), 'utf8'))) {
      await sql(stmt)
    }
    await sql`INSERT INTO _migrations (name) VALUES (${file})`
  }
  console.log('migrations complete')
}

main().catch(e => { console.error(e); process.exit(1) })
