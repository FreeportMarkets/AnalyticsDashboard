/**
 * Fails if any metric source groups or range-filters on the `date` column.
 *
 * `date` is the DynamoDB partition key, derived in UTC. Every metric must bucket
 * with (ts AT TIME ZONE 'America/New_York'). Grouping on `date` shifts daily
 * figures by up to 5 hours and the wrong numbers look entirely plausible, which
 * is why this is enforced mechanically rather than by review.
 *
 * Scope: src/lib/metrics/** only. Sync, backfill, and verify legitimately use
 * `date` as a storage/partition key.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'

const ROOT = path.join(import.meta.dirname, '..', 'src', 'lib', 'metrics')

const BANNED: Array<{ re: RegExp; why: string }> = [
  { re: /GROUP\s+BY\s+[^\n]*\bdate\b/i, why: 'GROUP BY on the `date` column' },
  { re: /WHERE[^\n]*\bdate\b\s*(>=|<=|<|>|BETWEEN)/i, why: 'range filter on the `date` column' },
]

function walk(dir: string): string[] {
  const out: string[] = []
  for (const e of readdirSync(dir)) {
    const p = path.join(dir, e)
    if (statSync(p).isDirectory()) out.push(...walk(p))
    else if (p.endsWith('.ts')) out.push(p)
  }
  return out
}

let failures = 0
for (const file of walk(ROOT)) {
  const lines = readFileSync(file, 'utf8').split('\n')
  lines.forEach((line, i) => {
    if (line.trimStart().startsWith('//') || line.trimStart().startsWith('*')) return
    for (const { re, why } of BANNED) {
      if (re.test(line)) {
        console.error(`${path.relative(process.cwd(), file)}:${i + 1}: ${why}`)
        console.error(`  ${line.trim()}`)
        failures++
      }
    }
  })
}

if (failures > 0) {
  console.error(`\n${failures} violation(s). Metrics must bucket with (ts AT TIME ZONE 'America/New_York').`)
  process.exit(1)
}
console.log('lint:dates passed — no metric groups or range-filters on `date`')
