# Analytics v2 — Streamlit → Vercel Design

**Date:** 2026-07-20
**Status:** Approved, ready for implementation planning
**Repos touched:** `analytics-dashboard` (rewrite), `FreeApp` (telemetry), `Web_Terminal` (telemetry), `Swap_Server` (deferred ingest hardening only)

---

## Context

### The problem

The current dashboard is a single 3785-line Streamlit script (`app.py`) deployed at
`freeportanalyticsdashboard.streamlit.app`. It is stale, slow, requires constant manual
refreshing, and roughly half of it covers infrastructure health that nobody reads.

The slowness is **not** because it is Python. It is because:

1. Every page load re-reads raw DynamoDB partitions for the whole selected date range
   (`app.py:617-699`), then does all aggregation in pandas inside the request.
2. Streamlit reruns the entire script on every widget interaction, so changing a date
   re-triggers the full load behind `@st.cache_data(ttl=300)`.
3. DynamoDB cannot answer product-analytics questions (funnels, retention cohorts,
   dwell-time percentiles, arbitrary breakdowns) without full scans.

Porting the same query pattern to Next.js would be equally slow. The fix is a query layer,
not a language change.

### What is missing

Two-thirds of the product is invisible to analytics today.

**FreeApp** (`services/AnalyticsService.ts:66`) emits 38 event names but has **no screen
tracking and no dwell timers**. The only navigation signal is `tab_switch`
(`hooks/analytics/useAnalytics.ts:35`) across 5 top-level tabs. Uninstrumented surfaces
include: all 9 Home sub-tabs (`components/feed/HomeTab.tsx:93-112`), `PredictDetailPage`,
`RelatedListPage`, `TaxCenterPage`, the entire `components/analyst/*` tree,
`TickerSentimentPage`, `BriefingDetailPage`, `FreddyChatSheet`, `BucketDetailPage`,
`DailyHistoryPage`, `TransactionHistory`, all `components/settings/*` and
`components/profile/*` sheets, `components/friends/*`, `components/insider/*`,
`components/stock/*`.

**Web Terminal** has **no product analytics at all** — 7 error-only events to Vercel
Analytics (`Web_Terminal/app/src/lib/telemetry.ts:9`). Nothing on login, order placement,
deposit, market selection, view navigation, modal opens, or news reads.

### Current data topology

| Fact | Location |
|---|---|
| FreeApp events → `POST /api/events/batch` | `FreeApp/services/AnalyticsService.ts:8` → `Swap_Server/src/index.ts:162` |
| Ingest handler (fire-and-forget write) | `Swap_Server/src/routes/analytics.ts:38-76` |
| DynamoDB write | `Swap_Server/src/services/analytics.ts:70-110` |
| Table `freeport-analytics-events` | PK `date` (S), SK `sk` = `{timestamp}#{wallet8}#{rand8}`; GSIs `wallet-date-index`, `event-date-index`; PAY_PER_REQUEST; **no stream, no TTL, no PITR**. Defined only in the one-off script `Swap_Server/scripts/create-analytics-table.ts:34-60` |
| Table `freeport-trades-history` | PK `wallet_address`, SK `timestamp`; GSI `trade_date-timestamp-index` (managed out-of-band, `Swap_Server/infra/cdk/src/tables.ts:51-81`) |
| Points data | Postgres in trading-backend, reached via `https://trading-api.freeportmarkets.com/v1/points` gated by `x-admin-key` |

**Live measurements (2026-07-20):** 1,507,165 items / 464 MB in the analytics table.
~2–5k events/day at current DAU. Projected ~300k events/day at 1000 DAU with full screen
instrumentation (~9M/month). This is comfortably Postgres-sized; ClickHouse would be overkill.

### Intended outcome

A fast, secure, always-fresh Vercel dashboard that covers **both** FreeApp and Web Terminal,
answers time-on-page and feature-usage questions across every surface including the
dynamically backend-generated ones, and supports composable funnel analysis — without
altering a single line of trade, volume, or balance logic.

---

## Non-goals / hard constraints

- **No change to any money path.** Trade, volume, balance, and order write paths are frozen.
  Telemetry additions are strictly additive, fire-and-forget, and failure-isolated.
- **No change to the existing ingest pipeline in this project.** DynamoDB remains the source
  of truth. Postgres is a derived read replica. If sync breaks, apps are unaffected.
- **Volume math is ported verbatim**, not reinterpreted. Parity with Streamlit is a gate.
- Infrastructure-health tabs (Backend, Services) are deleted, not migrated. CloudWatch
  dashboards and alarms already own that job.

---

## Architecture

```
FreeApp ──┐
          ├─► swap-server /api/events/batch ──► DynamoDB freeport-analytics-events  [UNCHANGED]
WT ───────┘                                              │
                                                         │ cron pull, every 60s
                                                         ▼
                              Neon Postgres ◄── incremental sync + rollup refresh
                                    │
                                    ▼
                     Next.js on Vercel (RSC + SQL) ── Google SSO allowlist
```

### Why cron-pull instead of DynamoDB Streams

The sort key `{timestamp}#{wallet8}#{rand8}` is lexicographically time-sortable, so an
incremental read is a cheap key-condition range query, not a scan:

```
KeyConditionExpression: #date = :d AND sk > :watermark
```

Streams would require enabling a stream on a table created by an untracked one-off script,
plus a Lambda and a CDK stack in a second repo (`Swap_Server`). Cron-pull achieves ~60s
freshness with all code in one repo and zero mutations to a live table.

### Sync job

`/api/cron/sync`, Vercel cron at 60s.

1. Read watermark from `sync_state`.
2. Query DynamoDB for events after the watermark, across the current and previous UTC date
   partition (events near UTC midnight land in the next partition — the same reason
   `app.py:631-646` fetches one extra day).
3. Bulk upsert into Postgres with `ON CONFLICT (date, sk) DO NOTHING`. Idempotent; safe to
   re-run or overlap.
4. Advance the watermark to **now minus 10 minutes**, never to now.
5. Refresh affected `daily_rollups` and `screen_sessions` rows.

**Late-arrival handling.** The client flushes every 30s and requeues up to 500 events on
failure (`FreeApp/services/AnalyticsService.ts:5-7`, `:150-153`), so events can arrive minutes
after their `timestamp` and would slip behind a live watermark. Two defenses: the 10-minute
watermark lag, and a nightly reconciliation pass that re-reads the previous 2 days in full.
Because upserts are idempotent, re-reading is free of side effects.

**Backfill.** The same code path, invoked as a one-off script over all historical dates.
1.5M rows, expected ~30 minutes.

**Trades** mirror on the identical pattern via the `trade_date-timestamp-index` GSI
(the query already exists at `app.py:649-664`).

### Failure behavior

- Sync failure is loud but non-blocking: the dashboard renders last-synced data with a
  visible staleness badge showing watermark age. It never silently shows stale numbers as live.
- A watermark older than 10 minutes surfaces as a banner, not a hidden degradation. This is
  the single biggest complaint about the Streamlit version.
- Neon unavailability degrades to an error state; it cannot affect apps or ingest.

---

## Postgres schema

```sql
-- Raw event mirror, monthly range partitions on date
events (
  date          date        not null,
  sk            text        not null,
  ts            timestamptz not null,
  event         text        not null,
  screen        text,
  component     text,
  wallet_address text,
  session_id    text,
  platform      text,          -- 'ios' | 'android' | 'web'
  app_version   text,
  metadata      jsonb,
  primary key (date, sk)
) partition by range (date);
-- indexes per partition: (event, ts), (wallet_address, ts), (session_id, ts)

-- Trade mirror, same sync pattern
trades (wallet_address, timestamp) primary key, ...;

-- Cursor per source
sync_state (source text primary key, watermark_ts timestamptz, updated_at timestamptz);

-- Derived: paired screen_view -> screen_exit
screen_sessions (
  wallet_address, session_id, screen, screen_params jsonb,
  entered_at timestamptz, exited_at timestamptz,
  dwell_ms bigint, exit_reason text, is_lower_bound boolean
);

-- Precomputed metrics for instant page loads
daily_rollups (date, metric text, dims jsonb, value numeric);

-- Who mutated what, from the dashboard
audit_log (id, actor_email, action, target, payload jsonb, created_at);
```

**Retention:** raw `events` kept 90 days hot (drop old partitions); `daily_rollups` and
`screen_sessions` kept indefinitely. Neon Launch tier (~$19/mo, 10 GB) covers this with
headroom at projected volume.

**`screen_sessions` construction.** The cron pairs each `screen_view` with the next
`screen_exit` for the same `(session_id, screen)`. An unpaired `screen_view` — app killed,
event lost — yields a lower-bound dwell computed against the next event in the session, with
`is_lower_bound = true` so dwell percentiles can exclude or annotate it. This is the table
that answers "how long do users stay on pages".

---

## Dashboard application

**Stack:** Next.js 15 App Router, TypeScript, Tailwind + shadcn/ui, Framer Motion, Recharts.
Server Components query Neon directly via `@neondatabase/serverless`. No AWS SDK and no
credentials ever reach the browser.

**Freshness model:** historical panels use `unstable_cache` + `revalidateTag`; live tiles
poll a narrow `today` query every 15s. Every view displays watermark age.

**Auth:** Auth.js with the Google provider plus an email allowlist in env, enforced in
middleware on every route including API handlers. AWS credentials and the points admin key
are server-only. This replaces a dashboard that is currently **publicly reachable** on
streamlit.app while holding promo-code mutation powers.

**Mutations:** promo-code CRUD and beneficiary attachment (currently `app.py:1238-1312`) move
to Server Actions, each writing an `audit_log` row. Streamlit has no record of who did what.

### Tabs

| Tab | Disposition |
|---|---|
| Overview | Ported, upgraded |
| Users | Ported, upgraded |
| Retention | Ported (cohort heatmap, retention curve) |
| **Funnels** | **Rebuilt — composable** |
| **Screens & Engagement** | **New** |
| Trades / Volume | Ported, math verbatim |
| Notifications | Ported |
| Referrals admin | Ported + audit log |
| ~~Backend~~ | **Deleted** |
| ~~Services~~ | **Deleted** |

**Funnels, rebuilt.** `app.py` hardcodes three funnels (notification→trade, trade, deposit).
The replacement lets any ordered event sequence be selected with a conversion window and a
breakdown dimension (platform, app_version, cohort, screen), backed by a single SQL window
function over `events`. This is the "way better info from funnels" requirement.

**Screens & Engagement, new.** Dwell-time distribution per screen (p50/p90/max), screen-flow
Sankey from `referrer_screen`, feature-usage leaderboard from `feature_use`, and per-screen
drop-off. Split and comparable by `platform`, which is what makes FreeApp-vs-WT analysis work.

### Volume-math parity gate

`hl_volume.py` (100 lines) and `apply_perps_leverage()` (`app.py:136-173`), including the
Ostium notional-vs-base-units special case (`app.py:127-135`), port 1:1 to TypeScript.
`test_hl_volume.py` (154 lines) is translated to a TS test suite. **Cutover is blocked until
the new dashboard reproduces Streamlit's volume numbers over the same date ranges.**

---

## Telemetry specification

Three new event names, additive. All 38 existing names keep their current shape.

```ts
screen_view  { screen, screen_params: { entity_type, entity_id, source }, referrer_screen }
screen_exit  { screen, dwell_ms, exit_reason: 'back' | 'nav' | 'background' | 'kill' }
feature_use  { feature, surface, ...ctx }
```

**Page identity.** `screen` is a stable, low-cardinality key (`token_detail`). Identity of the
specific instance lives in `screen_params`, e.g.
`{ entity_type: 'perp', entity_id: 'BTC', source: 'recommender' }`. Rollups group on `screen`;
drill-down filters on params. This handles pages generated dynamically by different backends
(recommender, TradeNews, HL market list) without cardinality explosion, and it works
identically for FreeApp's portal-based detail pages and WT's view/modal model.

`screen_params.source` is what makes backend attribution possible — which upstream service
produced the page a user is looking at.

### FreeApp

- One hook, `useScreenTracking(screen, params)`, added one line per screen component.
- A module-scoped screen stack (consistent with the render-stability discipline already used
  in this codebase: module-scoped setters, leaf-level store subscriptions) supplies
  `referrer_screen` and identifies the active screen.
- `AppState` background transition emits `screen_exit` for the active screen with
  `exit_reason: 'background'` and flushes, reusing the existing flush-on-background path at
  `services/AnalyticsService.ts:146`.
- Touches ~40 view components. **Zero changes to trading, balance, or order code.**
- JS/TS-only, no native dependencies → ships OTA on the current `runtimeVersion`.

### Web Terminal

- New `src/lib/analytics.ts` mirroring `AnalyticsService`: in-memory queue, 30s flush,
  50-event flush threshold, `navigator.sendBeacon` on `pagehide` so the final `screen_exit`
  survives tab close.
- `platform: 'web'`, `wallet_address` from `useAuth()` (`src/hooks/useAuth.ts:58-63`),
  `app_version` from the build SHA.
- Wired into `useView` (`src/lib/useView.ts:30`), `modal.tsx` (`src/lib/modal.tsx:3`), and
  `NewsView` tab state (`src/views/NewsView.tsx:13`).
- Posts to the same `/api/events/batch`; wildcard CORS already permits it
  (`Swap_Server/src/index.ts:41-50`).
- Vercel Analytics and Speed Insights stay for web vitals.

---

## Deferred: ingest hardening

Explicitly **out of scope** for this project, specified here so it can be picked up as a
standalone PR against `Swap_Server`.

Current state of `POST /api/events/batch`:

| Issue | Location |
|---|---|
| Zero authentication — `wallet_address` is client-asserted and forgeable by anyone | `Swap_Server/src/routes/analytics.ts:38` |
| `Access-Control-Allow-Origin: *`, no origin allowlist | `Swap_Server/src/index.ts:41-50` |
| No rate limiting of any kind (only a 200-event batch cap) | `Swap_Server/src/routes/analytics.ts:16` |
| `BatchWriteCommand` `UnprocessedItems` never inspected or retried → **silent data loss under throttle**, while the response still counts them as saved | `Swap_Server/src/services/analytics.ts:103-109` |
| `sk` uses `Math.random()` with no idempotency key → client retries create duplicate rows | `Swap_Server/src/services/analytics.ts:73` |
| No TTL on the table → unbounded growth | `Swap_Server/scripts/create-analytics-table.ts:59` |

Proposed fix, dual-mode so nothing breaks: new clients attach the Privy access token, the
server verifies it and stamps `verified: true`; legacy builds are still accepted as
`verified: false`. Add IP rate limiting, an origin allowlist, `UnprocessedItems` retry with
backoff, a client-supplied idempotency key in `sk`, and a TTL. The dashboard filters to
verified events by default once coverage is high.

Note the duplicate-row risk interacts with the sync job: `ON CONFLICT (date, sk) DO NOTHING`
dedupes identical rows, but two DynamoDB rows created by a client retry have different random
`sk` values and will both mirror. Volume metrics are unaffected (they come from `trades` and
HL per-fill data), but raw event counts may be marginally inflated until this is fixed.

---

## Sequencing

**Phase 1 — Pipeline.** Neon provisioning, schema and partitions, sync job, historical
backfill, rollup refresh. Verified by row-count and spot-value parity against DynamoDB.

**Phase 2 — Dashboard.** Next.js app, Google SSO, all surviving tabs on the existing 38
events, volume-math parity gate, Streamlit decommissioned. **This is the point at which the
current pain stops.**

**Phase 3 — Telemetry.** WT analytics client, then FreeApp screen tracking broken into
per-surface PRs (top-level tabs → detail pages → sheets/modals) rather than one mega-PR, per
the one-PR-at-a-time discipline in this codebase. New data flows into a dashboard that already
exists, so each PR is immediately verifiable.

**Phase 4 (separate project) — Ingest hardening.** As specified above.

---

## Verification

- **Sync correctness:** row counts per date in Postgres match DynamoDB `--select COUNT` for a
  sample of dates; a targeted event's fields match byte-for-byte.
- **Volume parity:** ported TS volume math passes the translated `test_hl_volume.py` suite,
  and total perp/swap/deposit volume matches Streamlit across 1d / 7d / 30d ranges. Blocking
  gate for cutover.
- **Freshness:** watermark age stays under 2 minutes in steady state; the staleness banner
  fires correctly when the cron is deliberately paused.
- **Auth:** unauthenticated and non-allowlisted requests are rejected at middleware on every
  route including API handlers; no AWS credential or admin key appears in any client bundle.
- **Telemetry:** for a scripted session across N screens, `screen_sessions` dwell values match
  wall-clock within tolerance; app-kill produces a `is_lower_bound` row rather than a gap.
- **Non-disturbance:** trade, volume, and balance code paths show zero diff in Phase 3 PRs.
