import { AccountCohortsSection } from './AccountCohortsSection'
import type { AccountPopulation } from '@/lib/accountMetrics'
import { todayNy, addDays } from '@/lib/ranges'

/** Account-created cohorts come only from the versioned backend read model. */
export async function JourneyDaily({ from, to, population }: { from?: string; to?: string; population?: AccountPopulation } = {}) {
  const end = to ?? todayNy()
  // Preserve malformed user input for the API's bad-range response; never throw
  // in date arithmetic before the section can render that response.
  const validEnd = /^\d{4}-\d{2}-\d{2}$/.test(end) && Number.isFinite(Date.parse(`${end}T00:00:00Z`))
  return AccountCohortsSection({ from: from ?? (validEnd ? addDays(end, -29) : end), to: end, initialMetric: 'firstTrade', population })
}
