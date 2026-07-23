import { NY_TZ } from './time'

/**
 * The date-range presets, defined ONCE.
 *
 * This constant was previously copy-pasted into all five page files. That is
 * exactly the wrong place for it: the numbers on this dashboard are compared
 * against the Streamlit dashboard, and a preset that drifts on one page (or
 * against Streamlit) produces a figure that is wrong in a way nobody can see
 * -- it just quietly disagrees.
 *
 * `lookbackDays` is a LOOKBACK, not a span. The range it produces is
 * `today - lookbackDays` through `today`, INCLUSIVE, which is
 * `lookbackDays + 1` calendar dates.
 *
 * That off-by-one is deliberate and it is what Streamlit does:
 *
 *     # app.py
 *     value=(today - timedelta(days=7), today)
 *
 * Today is always a PARTIAL day -- the range ends at whatever time you load
 * the page, not at midnight. So `today - 7 .. today` is seven complete days
 * plus today-so-far. The previous implementation here used
 * `addDays(end, -(days - 1))`, i.e. `today - 6 .. today`, which is only six
 * complete days plus today-so-far, and every figure on this dashboard read
 * roughly a seventh low against Streamlit for the same "7D" label.
 *
 * If you ever change this, change `app.py` in the same commit, or the two
 * dashboards stop agreeing.
 */
export const RANGES = {
  '7d': { label: '7D', lookbackDays: 7 },
  '30d': { label: '30D', lookbackDays: 30 },
  '90d': { label: '90D', lookbackDays: 90 },
} as const

export type RangeKey = keyof typeof RANGES

export function isRangeKey(v: string | undefined): v is RangeKey {
  return v === '7d' || v === '30d' || v === '90d'
}

/** Today's NY calendar date, as YYYY-MM-DD. */
export function todayNy(): string {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: NY_TZ,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(new Date())
}

export function addDays(date: string, days: number): string {
  const d = new Date(`${date}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

/** Inclusive start date for a preset ending on `end`. */
export function rangeStart(end: string, range: RangeKey): string {
  return addDays(end, -RANGES[range].lookbackDays)
}

/**
 * Number of calendar dates the preset actually covers, inclusive. Use this
 * for user-facing copy -- `lookbackDays` is one less and saying "the prior 7
 * days" next to an 8-date range is the kind of small lie that costs an hour
 * later.
 */
export function rangeSpanDays(range: RangeKey): number {
  return RANGES[range].lookbackDays + 1
}
