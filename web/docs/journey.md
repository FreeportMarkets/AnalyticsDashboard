# Journey metrics

The page separates daily account activity from the intro device cohort.

## Daily growth

- New signups: unique non-guest Privy DIDs by `created_at`, including walletless
  accounts. Imported/pre-created accounts are included in this creation measure.
  Source contract: [Privy get users](https://docs.privy.io/api-reference/users/get-all).
- First successful deposit: earliest stored `deposit_success`,
  `deposit_completed`, or `deposit_funds_arrived` for the account's linked wallets.
  Payment completion is not necessarily spendable funds; arrival also covers
  transfers without checkout.
- First filled trade: earliest stored `trade_success`/`trade_succeeded` event or
  swap/perps trade record with status `success` or `filled`.
- Same-day signup conversion: the above first actions occurring after account
  creation and before the next New York midnight on its signup day. Deposit and
  trading groups overlap; do not sum them or treat deposit as required for trade.

All daily buckets use America/New_York, including DST. MIN runs over all stored
history **before** applying the visible date range. Repeat actions and multiple
linked wallets count once per account. EVM casing is normalized; Solana casing
is preserved. Ambiguous wallet ownership and unlinked wallets are excluded.
Activity predating account creation is not moved onto the signup date.

These are first **recorded** actions, not proven lifetime firsts. Missing old
history, deleted accounts, changed wallet links, and incomplete event coverage
can affect historical counts. The page exposes this definition and actual sync
watermarks. A populated sync cursor is not proof of full historical coverage.

The complete Privy account snapshot is held in Next's server data cache for
five minutes. Errors, invalid records, repeated cursors, and pagination limits
never produce a partial count. This preserves walletless signups and avoids
the nightly identity mirror, which has neither signup timestamps nor today's
new accounts. Only daily aggregate counts reach the client chart. A cold load
can wait for pagination; the section streams with a loading state.

Uses existing `PRIVY_APP_ID`, `PRIVY_APP_SECRET`, and `DATABASE_URL`; no schema
migration. If activity reads fail or either sync source has no cursor, action
counts render unavailable, not zero. Signup failure withholds the daily view.
Today remains partial and historical zeros mean no matching stored record,
not verified absence of activity. Production source coverage must be audited
before these recorded firsts are used as lifetime acquisition metrics.

## Intro cohort

The trading API returns independent counts of devices that reached milestones
within 24h/7d after first non-replay `intro_started`. It does not return device
intersections or ordered transition counts.

- Default `Most reached` orders these counts descending without changing them.
- `Journey order` preserves the API's `step_index` order.
- Every milestone share uses its intro cohort size as denominator.
- Intro-to-trade conversion and no-recorded-trade counts use the cohort entry
  and `trade_succeeded` outcome. Missing/inconsistent outcomes are unavailable.
- Adjacent differences cannot establish drop-off, skip rate, or failure rate.
- Payment paths are optional. Recent cohorts remain separate and collapsed.
- Cohort dates stay UTC, matching the API contract, and are labeled separately
  from the daily New York chart controls.

A true descending multistep funnel needs a backend query grouping by device,
requiring every prior event in timestamp order within the observation window.
Separate funding branches (checkout, transfer, referral points) before measuring
transition loss. Do not emulate this by clamping or sorting raw counts and
then assigning drop-off percentages.
