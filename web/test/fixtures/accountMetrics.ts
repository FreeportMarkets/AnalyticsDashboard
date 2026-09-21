import type { AccountMetrics, AccountHorizon } from '@/lib/accountMetrics'
export const horizon = (overrides: Partial<AccountHorizon> = {}): AccountHorizon => ({
  days: 1, mature: true, eligibleAccounts: 10, eligibleMobileAccounts: 5,
  fundedAccounts: 2, firstTradeAccounts: 3, appReturningAccounts: 1, tradingReturningAccounts: 2,
  fundingRate: 0.2, firstTradeRate: 0.3, appReturnRate: 0.2, tradingReturnRate: 0.2,
  observedFeeRevenueUsd: '1.50', observedFeeRevenuePerAccountUsd: '0.15', ...overrides,
})
export const immature = (days: number): AccountHorizon => horizon({ days, mature: false, eligibleAccounts: 0, eligibleMobileAccounts: 0, fundedAccounts: null, firstTradeAccounts: null, appReturningAccounts: null, tradingReturningAccounts: null, fundingRate: null, firstTradeRate: null, appReturnRate: null, tradingReturnRate: null, observedFeeRevenueUsd: null, observedFeeRevenuePerAccountUsd: null })
export const accountFixture = (): AccountMetrics => ({
  schemaVersion: 1, generatedAt: '2026-09-21T14:30:00Z', timezone: 'America/New_York', cohortBasis: 'account_created',
  range: { from: '2026-09-15', to: '2026-09-21' },
  coverage: { status: 'partial', accountsAsOf: '2026-09-21T14:00:00Z', activityAsOf: '2026-09-21T14:29:00Z', activitySourceFrom: '2026-09-10T00:00:00Z', activitySourceThrough: '2026-09-21T14:29:00Z', warnings: ['Direct crypto funding is not covered.'] },
  rows: [{ cohortDate: '2026-09-15', accounts: 10, mobileLinkedAccounts: 5, observedFundedAccounts: 2, observedFirstTradeAccounts: 3, horizons: [horizon(), immature(7), immature(14), immature(30)] }],
  expectedLtv: null, acquisitionCost: null,
})
