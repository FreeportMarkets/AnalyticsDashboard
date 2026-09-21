import { addDays } from './ranges'

/** Both APIs use inclusive dates, but their cohort calendars differ. */
export function journeyPreset(days: number, now = new Date()) {
  const to = now.toISOString().slice(0, 10)
  const accountTo = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(now)
  return { from: addDays(to, 1 - days), to, accountFrom: addDays(accountTo, 1 - days), accountTo }
}
