import { neon } from '@neondatabase/serverless'
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')
const sql = neon(url)

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
    await sql(readFileSync(path.join(dir, file), 'utf8'))
    await sql`INSERT INTO _migrations (name) VALUES (${file})`
  }
  console.log('migrations complete')
}

main().catch(e => { console.error(e); process.exit(1) })
