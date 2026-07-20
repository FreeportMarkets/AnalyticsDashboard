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

## Priority order

When any two of these conflict, the earlier one wins:

1. **Data correctness.** A wrong number is worse than no number. Every metric must be
   provably equal to Streamlit's, or provably better for a documented reason.
2. **Usability.** Fast, obvious, no manual refreshing, honest staleness indication.
3. **Coverage.** New screens, dwell, funnels.
4. **Polish.** Animation and visual refinement come last and never at the cost of the above.

---

## Non-goals / hard constraints

- **No change to any money path.** Trade, volume, balance, and order write paths are frozen.
  Telemetry additions are strictly additive, fire-and-forget, and failure-isolated.
- **No change to the existing ingest pipeline in this project.** DynamoDB remains the source
  of truth. Postgres is a derived read replica. If sync breaks, apps are unaffected.
- **Volume math is ported verbatim**, not reinterpreted. Parity with Streamlit is a gate.
- Infrastructure-health tabs (Backend, Services) are deleted, not migrated. CloudWatch
  dashboards and alarms already own that job.
- **Streamlit is not retired until the replacement has run correct in parallel for a full
  shadow period.** See "Shadow run and cutover".

---

## Volume integrity — the load-bearing guarantee

Volume has broken before (`project_trade_logger_phantom_fills_jun18`, and commit `c79fd3e`
killing a close-side double-count). It is the metric this project is least permitted to
disturb. Three structural facts make that guarantee hold by construction rather than by
discipline.

### 1. The dashboard is read-only

`app.py` and `hl_volume.py` contain **zero** `put_item`, `batch_writer`, `update_item`,
`delete_item`, or `BatchWrite` calls. The only non-GET HTTP calls are an Arbitrum RPC balance
read (`app.py:749`) and three promo-code admin POSTs (`app.py:1248`, `:1270`, `:1298`) which
hit the points API and never touch trades. The replacement keeps this property: **the
dashboard's only writes are to its own Neon database and to the points admin API.**

### 2. Authoritative perp volume is not stored in our data source

`hl_perp_volume_usd()` (`app.py:702`) recomputes perp volume live from Hyperliquid
`userFillsByTime` on every load: Σ(px × size) over the four perp directions, deduped by `tid`,
paginated past HL's 2000-fill response cap (`hl_volume.py:hl_fills_fetcher`). This is the same
per-fill notional HL charges builder fees on.

**Hyperliquid is the source of truth.** `freeport-trades-history` contributes only the wallet
list. There is therefore nothing in our storage for this project to corrupt — the failure mode
is reduced to "the new dashboard computes it wrong", which the parity harness gates on
directly.

### 3. Both volume writers live outside this project's blast radius

| Writer | Location | Touched by this project |
|---|---|---|
| Perp trade rows | `freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts:195` | No |
| Swap trade rows | `Swap_Server/src/services/dynamodb.ts:67-107` | No |
| Order submission | `FreeApp/hooks/trading/useSwap.ts`, `hooks/trading/usePerpsHandlers.ts` | No |
| Web-vs-mobile attribution (`x-client` header) | `Web_Terminal/app/src/lib/tradingApi.ts:480` → `trade-logger.ts:189` | No |

**Do-not-touch list.** The files above must show a zero diff in every PR of this project,
including Phase 3 telemetry PRs. Enforced by a CI check that fails if any of them appear in
the diff. This matters most for `tradingApi.ts`: Phase 3 instruments Web Terminal, and the
mobile-vs-web volume split depends on that file's `x-client` header.

**Telemetry isolation rule.** Screen and feature instrumentation is added at *view* components
and navigation boundaries, never inside trading, balance, or order-submission hooks. If a
trade-related interaction needs a `feature_use` event, it is emitted from the component that
renders the control, not from the hook that executes the trade.

### Two pre-existing behaviors being carried forward, not fixed

Documented so they are not mistaken for regressions introduced here.

**Two different volume numbers coexist today.** `app.py:1445` uses authoritative HL data — on
the Overview tab only. Every other volume figure uses `_volume_usd`, the DB reconstruction,
across roughly 20 call sites including the iOS/Android split (`:1556`), the mobile-vs-web perp
split (`:2094`), per-asset breakdowns (`:1632`, `:1639`), the Trades tab (`:2076-2078`), and
the user deep-dive (`:1853`). The docstring at `app.py:705` states this reconstruction **runs
~15% hot**, because trade rows store intended order size rather than filled size.

Both are ported **exactly as they are**. The only change is presentational: figures derived
from `_volume_usd` are labeled as estimates rather than rendering identically to the
HL-sourced number. Unifying everything onto HL per-fill data would change published numbers
and is therefore a separate, explicit decision — not something this migration does silently.

**HL failure silently degrades accuracy.** `hl_perp_volume_usd` returns `None` on any
exception (`app.py:719`) and the caller falls back to the DB estimate with no indication, so
Overview volume can quietly switch from exact to ~15% hot. The fallback behavior is preserved;
the silence is not. The new dashboard shows a visible badge when volume is served from the
fallback.

---

## Known correctness risks

Enumerated deliberately. Each has a named defense; the shadow run exists to catch the ones
not listed here.

### 1. Timezone skew — highest risk

`app.py:1040-1048` re-derives `date`, `hour`, and `day_of_week` in **America/New_York** from
the raw timestamp. The DynamoDB partition key `date` and its denormalized `hour` /
`day_of_week` columns are **UTC** (`Swap_Server/src/services/analytics.ts:70-92`).

Grouping SQL by the mirrored `date` column shifts every daily figure by up to 5 hours and
disagrees with Streamlit in a way that looks plausible.

**Rule: the `date` partition key is storage only and must never appear in a `GROUP BY` or a
date-range filter for a metric.** All aggregation derives from
`ts AT TIME ZONE 'America/New_York'`. Enforced by a lint rule or code-review checklist item,
and by the parity harness, which would catch a violation immediately.

The sync job must also read both the current and previous UTC date partitions for the same
reason `app.py:631-646` fetches one extra day.

### 2. Decimal precision

`decimal_to_float()` (`app.py:110`) casts DynamoDB `Decimal` to Python `float`. Mirroring into
Postgres `numeric` preserves full precision, making the new dashboard **more** correct.

Volume parity will therefore show small diffs that are the new implementation being right.
The parity harness compares within a documented tolerance and flags any diff that exceeds it,
so an actual bug is not hidden inside expected float noise.

### 3. Silent row drops

pandas `errors="coerce"` turns unparseable timestamps into `NaT`, and those rows silently
disappear from Streamlit's numbers. SQL will instead fail the whole insert batch. Neither is
acceptable.

**Defense:** rows that fail parse or validation go to a `quarantine` table with the raw item
and a reason. The quarantine count is displayed in the dashboard. Never silently dropped,
never batch-fatal.

### 4. Partition management

Inserting into a month with no partition errors out. **Defense:** partitions are created
ahead of time by the cron, plus a `DEFAULT` partition as a backstop so an insert can never
fail on a missing partition. A non-empty default partition raises an alert.

### 5. Rollup drift

If backfill and incremental sync overlap, incrementally-maintained rollups double-count.
**Defense:** `daily_rollups` is always **recomputed from raw** for affected dates, never
incremented. Recomputation is idempotent.

### 6. Ported filter and math semantics

Easy to lose in translation, each silently shifting numbers:

| Semantic | Source |
|---|---|
| `SYSTEM_WALLETS` exclusion — `server`, `unknown`, `system`, `''`, plus `platform != 'server'` | `app.py:1155-1163` |
| Ostium `size` is USD notional, not base units | `app.py:127-135` |
| Perps leverage application | `app.py:136-173` |
| HL per-fill volume, pagination past the 2000-cap, no close double-count | `hl_volume.py`, commits `c79fd3e`, `27148c3` |
| `amount_usd` on perps rows is **margin**, not notional | `freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts:39-46` |

Each is covered by an explicit parity assertion.

### 7. Trade row overwrites

Both writers use `PutCommand` with no updates (`trade-logger.ts:195`; Swap_Server swap path),
but a Put on an existing `(wallet_address, timestamp)` key overwrites in place. **Defense:**
trades sync uses `ON CONFLICT DO UPDATE`, not `DO NOTHING`, so a rewritten row propagates.
Events remain `DO NOTHING` — they are append-only.

### 8. Event lateness — bounded, and pre-existing

The client queue is **in-memory only** (`FreeApp/services/AnalyticsService.ts:32`) with no
persistence. Events are therefore *lost* on app kill, not delayed; maximum lateness is bounded
by a single foreground session. The 10-minute watermark lag plus the nightly 2-day
reconciliation covers this comfortably.

Note this loss is **pre-existing and unchanged** — Streamlit reads the same incomplete data.
It is a known gap in absolute event counts, not a migration risk, and not something this
project introduces or is required to fix.

### 9. Backfill throughput

1.5M rows. **Defense:** chunked inserts with a resumable cursor, so a failure resumes rather
than restarts, and the job cannot exhaust a Neon connection pool.

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

-- Rows that failed parse or validation. Never silently dropped.
quarantine (id, source text, raw jsonb, reason text, created_at timestamptz);

-- Daily metric-by-metric diff between Streamlit and the new implementation
parity_runs (run_at, metric, dims jsonb, streamlit_value numeric,
             postgres_value numeric, abs_diff numeric, pct_diff numeric, passed boolean);
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

## Shadow run and cutover

Streamlit stays live and untouched throughout. The new dashboard runs in parallel against the
same underlying data, and **cutover is gated on measured agreement, not on a subjective
look-over.**

### Parity harness

A script (`scripts/parity.ts`, plus a `/api/cron/parity` daily run) computes the same metric
set two ways — Streamlit's exact pandas logic against DynamoDB, and the new SQL against
Postgres — and writes every comparison to `parity_runs`.

Metric set, each across 1d / 7d / 30d windows:

- event counts, total and per event name
- DAU / WAU / MAU, unique wallets
- session count and total session duration
- swap volume, perp volume, deposit volume — total and per venue
- perp volume split by surface (mobile vs web)
- trade counts by type and status
- funnel step counts for the three legacy hardcoded funnels
- retention curve values (D1, D7, D30)
- notification tap rate, time-to-trade-after-tap
- top-10 traders and top-10 active users, compared as ordered sets

Pass criteria:

- Count metrics: **exact equality.** Any diff is a bug.
- Currency metrics: within a documented float-precision tolerance. Anything above tolerance
  fails and must be explained before it is waived (see risk #2).
- Ordered-set metrics: identical membership and order.

A failed run surfaces as a banner in the new dashboard naming the failing metric. Parity is
not "checked once at the end" — it runs every day of the shadow period, so an intermittent or
date-boundary-specific bug has a chance to appear.

### Shadow period

**Minimum 7 consecutive days with zero unexplained parity failures**, not one. Seven days
because it is the shortest window that covers:

- a full weekly cycle, including a weekend traffic trough
- at least 7 UTC-midnight boundaries, where the timezone risk (#1) would manifest
- a D1 and D7 retention window computed entirely on mirrored data
- at least one full 90-day partition-management and rollup-refresh cycle tick

During the shadow period the team uses the new dashboard for real work while Streamlit remains
the source of truth for any decision. Usability problems found here are fixed before cutover,
not after.

### Cutover

Only when all of the following hold:

1. 7 consecutive clean parity days.
2. Watermark age held under 2 minutes in steady state for the full period.
3. Quarantine table empty, or every entry individually explained.
4. Volume-math TS test suite green (translated `test_hl_volume.py`).
5. The team says the new dashboard is genuinely more usable — an explicit sign-off, not an
   assumption.

Streamlit is then **left running but unlinked** for a further 2 weeks as a rollback path, and
only deleted after that. It costs nothing to leave up.

**Rollback:** re-share the Streamlit URL. It reads DynamoDB, which this project never modifies,
so it cannot be broken by anything here.

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
events, volume-math parity gate. Streamlit stays live.

**Phase 2.5 — Shadow run.** Both dashboards running, parity harness green for 7 consecutive
days, team using the new one for real work. Streamlit retired only at the end. **This is the
point at which the current pain stops** — and it is a gate, not a formality.

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
- **Full parity harness:** 7 consecutive clean daily runs across the whole metric set (see
  "Shadow run and cutover"). The single blocking gate for retiring Streamlit.
- **Timezone:** a deliberate test asserts that a metric grouped by the UTC `date` partition key
  disagrees with the same metric grouped in America/New_York — proving the harness would
  actually catch risk #1 rather than assuming it.
- **Quarantine:** an intentionally malformed event lands in `quarantine` with a reason, does
  not abort its batch, and raises the visible count.
- **Freshness:** watermark age stays under 2 minutes in steady state; the staleness banner
  fires correctly when the cron is deliberately paused.
- **Auth:** unauthenticated and non-allowlisted requests are rejected at middleware on every
  route including API handlers; no AWS credential or admin key appears in any client bundle.
- **Telemetry:** for a scripted session across N screens, `screen_sessions` dwell values match
  wall-clock within tolerance; app-kill produces a `is_lower_bound` row rather than a gap.
- **Non-disturbance (CI-enforced):** every PR in this project is checked against the
  do-not-touch list in "Volume integrity". A diff touching `trade-logger.ts`,
  `Swap_Server/src/services/dynamodb.ts`, `useSwap.ts`, `usePerpsHandlers.ts`, or
  `Web_Terminal/app/src/lib/tradingApi.ts` fails CI. This is a mechanical gate, not a review
  convention.
- **Volume write-path audit:** before Phase 3 merges, confirm the new dashboard issues no
  DynamoDB write of any kind — assert the IAM role backing the sync job holds
  read-only permissions on `freeport-trades-history` and `freeport-analytics-events`. The
  guarantee should be enforced by IAM, not only by code review.
