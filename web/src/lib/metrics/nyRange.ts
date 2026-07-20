import { NY_TZ, isValidCalendarDate } from '@/lib/time'

/** UTC offset in minutes for a given instant in New York. */
function nyOffsetMinutes(at: Date): number {
  // 'shortOffset' yields e.g. "GMT-4". Parse the hour offset from it.
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: NY_TZ, timeZoneName: 'shortOffset',
  }).formatToParts(at)
  const tz = parts.find(p => p.type === 'timeZoneName')?.value ?? 'GMT+0'
  const m = /GMT([+-])(\d{1,2})(?::(\d{2}))?/.exec(tz)
  if (!m) throw new Error(`could not determine ${NY_TZ} offset from "${tz}"`)
  const sign = m[1] === '-' ? -1 : 1
  return sign * (Number(m[2]) * 60 + Number(m[3] ?? 0))
}

/** The UTC instant of local midnight in New York on the given calendar date. */
function nyMidnightUtc(date: string): Date {
  const naive = new Date(`${date}T00:00:00Z`).getTime()
  // Two passes: the offset itself depends on the instant (DST).
  let guess = new Date(naive - nyOffsetMinutes(new Date(naive)) * 60_000)
  guess = new Date(naive - nyOffsetMinutes(guess) * 60_000)
  return guess
}

/**
 * Convert an INCLUSIVE New York calendar date range into half-open UTC instant
 * bounds [fromUtc, toUtc).
 *
 * Metrics filter `ts` (timestamptz) with these bounds. They must NEVER filter or
 * group on the `date` column: that is the DynamoDB partition key, derived in UTC,
 * and using it shifts every daily figure by up to 5 hours in a way that still
 * looks plausible. See the project's Global Constraints.
 */
export function nyRangeToUtc(startDate: string, endDate: string): { fromUtc: Date; toUtc: Date } {
  if (!isValidCalendarDate(startDate)) throw new Error(`invalid date: ${startDate}`)
  if (!isValidCalendarDate(endDate)) throw new Error(`invalid date: ${endDate}`)
  if (startDate > endDate) throw new Error(`inverted range: ${startDate} > ${endDate}`)

  const fromUtc = nyMidnightUtc(startDate)
  const dayAfterEnd = new Date(`${endDate}T00:00:00Z`)
  dayAfterEnd.setUTCDate(dayAfterEnd.getUTCDate() + 1)
  const toUtc = nyMidnightUtc(dayAfterEnd.toISOString().slice(0, 10))
  return { fromUtc, toUtc }
}
