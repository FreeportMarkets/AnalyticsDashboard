import type { FunnelStep } from './funnelApi'

export type JourneyOrder = 'reach' | 'journey'

/** These are independent milestone counts, never nested funnel steps. */
export function orderMilestones(steps: FunnelStep[], order: JourneyOrder): FunnelStep[] {
  return [...steps].sort((a, b) =>
    (order === 'reach' ? b.users_reached - a.users_reached : 0) || a.step_index - b.step_index,
  )
}

export function milestoneGroup(key: string): string {
  if (key.startsWith('deposit_')) return 'Funding · optional path'
  if (key.startsWith('trade_')) return 'Trading'
  if (['app_first_open', 'intro_started', 'intro_completed', 'auth_succeeded'].includes(key)) {
    return 'Onboarding'
  }
  return 'Other activity'
}

/** Only the cohort entry and one outcome are guaranteed to be nested. */
export function tradeOutcome(steps: FunnelStep[]) {
  const entry = steps.find(step => step.step_key === 'intro_started')
  const trade = steps.find(step => step.step_key === 'trade_succeeded')
  const cohortSize = entry?.cohort_size ?? steps[0]?.cohort_size ?? 0
  const valid = Number.isInteger(cohortSize) && cohortSize >= 0 && steps.every(step =>
    step.cohort_size === cohortSize && Number.isInteger(step.users_reached) &&
    step.users_reached >= 0 && step.users_reached <= cohortSize,
  ) && (!entry || entry.users_reached === cohortSize)

  // A missing outcome is unknown, not zero. Inconsistent counts must never
  // produce negative "no trade" counts or a conversion rate above 100%.
  if (!valid || !entry || !trade || cohortSize === 0) {
    return { cohortSize, traded: null, withoutTrade: null, conversion: null, valid }
  }
  return {
    cohortSize,
    traded: trade.users_reached,
    withoutTrade: cohortSize - trade.users_reached,
    conversion: trade.users_reached / cohortSize,
    valid,
  }
}
