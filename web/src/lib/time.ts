export const NY_TZ = 'America/New_York'

/**
 * SQL expression that buckets a timestamptz column into a New York calendar date.
 *
 * Use this for EVERY metric. The mirrored `date` column is the DynamoDB partition
 * key -- UTC-derived, storage only. Grouping on it shifts daily figures by up to
 * 5 hours in a way that looks entirely plausible (spec risk #1).
 */
// Matches a plain SQL identifier, optionally qualified as `table.column`
// (e.g. `ts`, `e.ts`). The caller is responsible for passing a column
// identifier here, never user input -- this is not a general SQL sanitizer,
// it only rules out the identifier being anything other than an identifier.
const SQL_IDENTIFIER = /^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?$/i

export function nyDateExpr(tsColumn: string): string {
  if (!SQL_IDENTIFIER.test(tsColumn)) {
    throw new Error(`nyDateExpr: invalid SQL identifier: ${JSON.stringify(tsColumn)}`)
  }
  return `(${tsColumn} AT TIME ZONE '${NY_TZ}')::date`
}

export function utcDateOf(ts: Date): string {
  return ts.toISOString().slice(0, 10)
}

export function nyDateOf(ts: Date): string {
  // Built from formatToParts rather than format() because locale fallback is
  // silent: en-CA is expected to yield YYYY-MM-DD, but on a small-ICU Node
  // build (or NODE_ICU_DATA pointing at incomplete data) ECMA-402 locale
  // negotiation can silently fall back to the runtime default locale instead
  // of throwing -- 'en-CA' could resolve to 'en-US', producing MM/DD/YYYY
  // with no exception. Assembling the string from named parts removes the
  // dependency on any locale's field ordering or separator. `timeZone`
  // resolution still throws loudly on a genuinely broken ICU build, which is
  // the correct behavior.
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: NY_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(ts)
  const get = (t: string) => {
    const p = parts.find(part => part.type === t)
    if (!p) throw new Error(`Intl.DateTimeFormat did not return a ${t} part`)
    return p.value
  }
  return `${get('year')}-${get('month')}-${get('day')}`
}
