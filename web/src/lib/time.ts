export const NY_TZ = 'America/New_York'

/**
 * SQL expression that buckets a timestamptz column into a New York calendar date.
 *
 * Use this for EVERY metric. The mirrored `date` column is the DynamoDB partition
 * key -- UTC-derived, storage only. Grouping on it shifts daily figures by up to
 * 5 hours in a way that looks entirely plausible (spec risk #1).
 */
export function nyDateExpr(tsColumn: string): string {
  return `(${tsColumn} AT TIME ZONE '${NY_TZ}')::date`
}

export function utcDateOf(ts: Date): string {
  return ts.toISOString().slice(0, 10)
}

export function nyDateOf(ts: Date): string {
  // en-CA yields YYYY-MM-DD.
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: NY_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(ts)
}
