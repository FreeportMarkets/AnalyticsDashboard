# Dashboard account measurement

## Sources and meaning

Users, Funnels and Journey read the versioned trading-backend endpoint
`GET /v1/analytics/accounts?from=YYYY-MM-DD&to=YYYY-MM-DD`. The shared
`ANALYTICS_FUNNEL_READ_SECRET` stays server-side in the `x-funnel-secret` header.
The optional `TRADING_API_BASE_URL` defaults to the existing production API.

Account cohorts use durable account creation dates in New York time, not the
first event inside the selected range. Account creation includes web, imported
and walletless accounts; it is not app-install conversion. The backend maintains
a complete account snapshot and daily production-linked mobile activity.

- **App return:** activity on the exact D1/D7/D14/D30 calendar day. The denominator
  includes only eligible accounts linked to a production install before that
  return day began. It differs from total linked accounts in the row.
- **Trading return:** a recorded confirmed trade on the exact return day, divided
  by eligible created accounts. Historical venue coverage is incomplete.
- **Verified card funding:** a verified positive card onramp inside the first
  24 hours / 7 / 14 / 30 days. This is a lower bound. Direct crypto transfers and
  unproven historical orders are not covered; they cannot be called unfunded.
- **First recorded trade:** first confirmed recorded fill within that same window.
  Trading does not require a card deposit.
- **Observed fees:** verified gross Freeport Hyperliquid USDC and Relay fees per
  eligible account. Unknown fees and costs are not zero; this is not profit/LTV.
- **Observed so far:** current source-bounded funding/first-trade counts through
  the report time. Includes immature cohorts, has no fixed-age conversion rate,
  and is never mixed into mature horizon numerators or denominators.

A horizon stays **Still observing** until its full return day has closed.
Unavailable values stay null. Totals sum eligible numerators and denominators,
never average cohort percentages. Missing observations are excluded, not zeroed.
The UI shows account/activity/report timestamps and source limitations.
Responses with cohort observations but no account snapshot are rejected. Missing
activity timestamps or coverage bounds prohibit app-return observations; valid
account funding, trading and fees remain available when only activity is missing.

Journey presets carry independent inclusive date ranges: `accountFrom` /
`accountTo` follow New York account cohorts; legacy intro `from` / `to` remain
UTC. Diagnostic window and sort controls preserve both. Old intro-only URLs do
not reinterpret UTC dates as account dates; account cohorts use their default
New York range.

Expected LTV has no validated forecast yet. Backend creator/referral/terms
records do not establish complete ad hoc deal costs. The September 21 read found
two creators, two creator-linked accounts and no current earnings entries; the
legacy referral store has reward eligibility, not paid creator costs. Costs
have not been reconciled into this report. Apple Ads campaigns run independently
and spend is not imported. CAC and
ROAS stay unavailable. Shared creator referral codes cannot prove which of a
creator's simultaneous campaigns acquired an account.

## Legacy diagnostics retained

Users retains explicitly web-only wallet/session/activity queries and trade
leaderboards. Funnels retains a web-only event funnel builder; all steps require
`platform='web'`. Journey retains device intro milestone reach with restored
sessions, replay markers and instrumentation caveats. None substitutes for mobile
account conversion. Overview discloses its legacy event source; trade/volume
read paths are preserved.

Trades deposit event counts are not unique orders or funded accounts. Their
success/initiation ratio can exceed 100%; it is not conversion. This HTTP request
runs inside its own Suspense boundary. Timeout/failure only replaces that section,
while trades and identities render independently. Identity enrichment starts as
soon as recent trades arrive.

`View rendered` measures the server-render timestamp, not refresh-request start
or source currency. `Mirror checked` describes legacy mirror cursors; it does not
claim current mobile coverage.

## Rollout and verification

Deploy the backend schema, worker and account endpoint before expecting measured
cohorts. Until then, missing configuration, absent endpoint, failed source or
incompatible v1 data display unavailable. There is no fallback to Neon retention.
No new dashboard secret is required if the existing funnel read secret is set.

Run from `web/`:

```sh
npm test
npx tsc --noEmit
```

Tests cover actual route dependency isolation, independent failure rendering,
web-only query predicates, contract validation, weighted denominators, immature
nulls, current progress, and missing-source states.

2026-09-21 validation: actual AccountCohortReport and production CSS rendered in
Chrome with real source data replayed through the backend implementation (1,592
accounts / 21 creation-date cohorts). All five metric toggles were exercised, including current funding/first-trade counts kept separate from fixed-age rates;
wide layout and narrow table scrolling were inspected. The endpoint itself was
not deployed. Real web-only Neon queries were invoked successfully (17–202ms
in this local sample). Trades slow/failing deposits were tested through the
actual route/section implementations with a controlled pending dependency;
production streaming-network TTFB has not been measured after this change.


## Journey population and interpretation controls (September 21 follow-up)

Journey can request `population=all|mobile_linked|unlinked`. The default remains
all accounts. Mobile-linked means at least one server-verified, nonambiguous
install binding observed by report time. No verified link does not prove web
acquisition: it includes older mobile users without trustworthy bindings.
Bindings can arrive later and change segment membership; these are account
creation cohorts filtered by observed app use, not mobile signup/install cohorts.

Backend filtering happens before aggregation. The dashboard verifies that the
response identifies the requested population; an old unfiltered response cannot
silently substitute for a segment. Date presets and diagnostic controls preserve
the selected population. Device intro diagnostics keep their independent scope.
Deploy the backend population parameter before enabling segmented reads; an older
endpoint continues to serve the default all-account report, while unsupported
segments display unavailable.

Return columns now say **On day 1/7/14/30**. Card funding, first recorded trade and
fees say **Within 24h / 7 / 14 / 30 days**. Aggregate columns can contain different
eligible cohorts; compare a single creation-date row across horizons. Show a
small-sample notice for fewer than 30 eligible accounts, without implying that 30
proves statistical significance. Explicit card-only coverage stays visible:
external crypto funding remains unmeasured, never reported as zero. Source-chain
USDC arrivals can be card autobridges or token-sale proceeds and cannot establish
new principal without origin classification.

The inline Journey guide explains population, activation, exact-day return and
observed gross fees. LTV and churn are not inferred from short-window return.
No mobile app change or OTA is needed for this dashboard batch.

## Operating Journey alongside AppsFlyer

Start with a question and keep its population and time window fixed:

| Question | Report | Read it as |
| --- | --- | --- |
| Do newly created accounts start trading? | Journey, First recorded trade, a mature Within 7 days column | Recorded traders divided by the eligible accounts shown in that cell |
| Do mobile-linked accounts return? | Journey, Mobile-linked, App return, On day 7 | Observed app use on that exact day; not seven-day cumulative retention or permanent churn |
| Are card deposits converting? | Journey, Verified card funding | Proven card funding only; a zero does not establish no external crypto funding |
| Which installed-user cohorts or campaigns produce trades and fees? | AppsFlyer, acquisition view | Installs and unique event users within the same install cohort, subject to identity and event coverage |
| How much fee revenue arrived during a reporting period? | AppsFlyer, activity view, reconciled to the backend delivery ledger | All eligible users' reported fees, including users installed before the selected period |

For a founder readout, retain the date range, selected population, horizon,
numerator, denominator and source timestamps alongside each percentage. Compare
equal-age mature cohorts; never treat the difference between independent Journey
tabs as funnel drop-off. A mobile-linked account segment is not an AppsFlyer
install cohort, so matching date labels alone do not make their rates comparable.

AppsFlyer delivery verification has three steps: confirm the underlying receipt,
check the backend delivery ledger, then match the event in AppsFlyer's reports.
HTTP acceptance alone does not prove reporting. Compare the same platform,
timezone and date basis. An aggregate dollar match proves aggregate reporting;
event-by-event reconciliation additionally requires raw records and their
install/account identifiers. Delayed S2S events can be dated at arrival rather
than their original occurrence, per the
[AppsFlyer timestamp rules](https://support.appsflyer.com/hc/en-us/articles/207034486-Server-to-server-events-API-for-mobile-S2S-mobile).

After the verified card exporter is enabled and genuine events appear, configure
AppsFlyer acquisition widgets for unique `first_card_deposit` users and their
conversion rate, grouped by media source and campaign. `card_deposit` counts
individual verified receipts. Sum `deposit_amount_usd` from `card_deposit` only
in a report that supports summing that custom parameter; standard dashboard
support for that sum has not been established. Do not add the first-card amount
again, and never report deposit principal as `af_revenue`. A missing card event
before worker activation is a rollout gap, not proof that no deposit happened.

Rollout acceptance: verify the backend population contract before deploying this
dashboard; validate card selection in staging preview before production export;
then reconcile a genuine production receipt through to AppsFlyer. No synthetic
conversions, duplicate historical sends or new customer payments are needed to
validate the reporting path. External-funding measurement and expected-LTV
modeling remain separately scoped work.


Follow-up validation: 290 dashboard tests passed (two optional DynamoDB tests
skipped), TypeScript and date-query lint passed. Actual updated components were
rendered in Chrome with three real backend query outputs: 1,687 total accounts,
128 mobile-linked and 1,559 unlinked. All population controls and five metric
modes were exercised; exact-day versus cumulative columns, null return coverage,
small samples, and the inline guide were checked. A 390px viewport kept overflow
inside the 780px table, with no document overflow or browser errors. These are
saved-source replay figures before historical funding recovery, not live counts.
The separately approved production recovery was verified in the live existing
Journey page: its selected creation window showed 14 card-funded accounts.
