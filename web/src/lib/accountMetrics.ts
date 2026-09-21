/** Version 1 contract from the trading backend account measurement read model. */
export interface AccountHorizon {
  days: number;
  mature: boolean;
  eligibleAccounts: number;
  eligibleMobileAccounts: number;
  fundedAccounts: number | null;
  firstTradeAccounts: number | null;
  appReturningAccounts: number | null;
  tradingReturningAccounts: number | null;
  fundingRate: number | null;
  firstTradeRate: number | null;
  appReturnRate: number | null;
  tradingReturnRate: number | null;
  observedFeeRevenueUsd: string | null;
  observedFeeRevenuePerAccountUsd: string | null;
}
export interface AccountCohort {
  cohortDate: string;
  accounts: number;
  mobileLinkedAccounts: number;
  observedFundedAccounts: number | null;
  observedFirstTradeAccounts: number | null;
  horizons: AccountHorizon[];
}
export interface AccountMetrics {
  schemaVersion: 1;
  generatedAt: string;
  timezone: 'America/New_York';
  cohortBasis: 'account_created';
  range: { from: string; to: string };
  coverage: {
    accountsAsOf: string | null; activityAsOf: string | null;
    activitySourceFrom: string | null; activitySourceThrough: string | null;
    status: 'partial' | 'unavailable' | 'stale';
    warnings: string[];
  };
  rows: AccountCohort[];
  expectedLtv: null;
  acquisitionCost: null;
}


export type AccountMetric = 'appReturn' | 'tradingReturn' | 'funding' | 'firstTrade' | 'revenue'
export const COHORT_DAYS = [1, 7, 14, 30] as const

export function metricValue(cell: AccountHorizon, metric: AccountMetric) {
  const denominator = metric === 'appReturn' ? cell.eligibleMobileAccounts : cell.eligibleAccounts
  const numerator = metric === 'appReturn' ? cell.appReturningAccounts
    : metric === 'tradingReturn' ? cell.tradingReturningAccounts
    : metric === 'funding' ? cell.fundedAccounts : cell.firstTradeAccounts
  const value = metric === 'revenue'
    ? cell.observedFeeRevenueUsd === null ? null : Number(cell.observedFeeRevenueUsd)
    : numerator
  return { denominator, value, rate: value === null || denominator === 0 ? null : value / denominator }
}

/** Weight by eligible accounts; an immature or uncovered cell contributes nothing. */
export function summarizeHorizon(rows: AccountCohort[], days: number, metric: AccountMetric) {
  let numerator = 0, denominator = 0, includedCohorts = 0
  for (const row of rows) {
    const cell = row.horizons.find(h => h.days === days)
    if (!cell?.mature) continue
    const point = metricValue(cell, metric)
    if (point.value === null || point.denominator <= 0) continue
    numerator += point.value
    denominator += point.denominator
    includedCohorts += 1
  }
  return { numerator, denominator, includedCohorts, rate: denominator > 0 ? numerator / denominator : null }
}

export function unavailableCellLabel(row: AccountCohort, cell: AccountHorizon | undefined, metric: AccountMetric) {
  if (cell && !cell.mature) return 'Still observing'
  if (metric === 'appReturn' && cell?.appReturningAccounts !== null && cell?.appReturningAccounts !== undefined && cell.eligibleMobileAccounts === 0) return 'No eligible linked accounts'
  if (metric === 'appReturn' && row.mobileLinkedAccounts === 0) return 'No linked accounts'
  return 'Source unavailable'
}
