# Analytics v2 — Phase 1: DynamoDB → Postgres Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror the `freeport-analytics-events` and `freeport-trades-history` DynamoDB tables into Neon Postgres via an idempotent incremental sync, with a verified historical backfill, so later phases can query analytics in SQL.

**Architecture:** A Vercel cron hits `/api/cron/sync` every 60s. It reads a watermark from `sync_state`, issues a DynamoDB key-condition range query (`date = :d AND sk > :watermark`) against the current and previous UTC date partitions, maps items to rows, and bulk-upserts into a monthly-partitioned `events` table with `ON CONFLICT DO NOTHING`. Trades sync identically but with `ON CONFLICT DO UPDATE`. Malformed rows go to `quarantine` rather than failing the batch. Backfill reuses the same mapping and insert code with a resumable cursor.

**Tech Stack:** Next.js 15 (App Router), TypeScript, `@neondatabase/serverless`, `@aws-sdk/client-dynamodb` + `@aws-sdk/lib-dynamodb`, Vitest, raw SQL migrations (no ORM — every query in this project is hand-written analytical SQL).

**Spec:** `docs/superpowers/specs/2026-07-20-analytics-v2-design.md`

---

## Global Constraints

Every task's requirements implicitly include this section.

- **The `date` column is storage only.** It must never appear in a `GROUP BY` or in a metric's date-range filter. All time bucketing derives from `ts AT TIME ZONE 'America/New_York'`. (Spec risk #1 — the highest-severity correctness risk in this project.)
- **Read-only against AWS.** This project issues no `PutItem`, `UpdateItem`, `DeleteItem`, or `BatchWriteItem` against any DynamoDB table, ever. The IAM principal must hold only `dynamodb:Query`, `dynamodb:GetItem`, and `dynamodb:DescribeTable`.
- **Do-not-touch list.** No PR in this project may modify: `freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts`, `Swap_Server/src/services/dynamodb.ts`, `Swap_Server/src/services/analytics.ts`, `FreeApp/hooks/trading/useSwap.ts`, `FreeApp/hooks/trading/usePerpsHandlers.ts`, `Web_Terminal/app/src/lib/tradingApi.ts`. Phase 1 touches no repo other than `analytics-dashboard`.
- **The Python Streamlit app at the repo root is untouched.** `app.py`, `hl_volume.py`, `test_hl_volume.py`, and `requirements.txt` stay exactly as they are. All new code lives under `web/`.
- **Never silently drop a row.** Parse and validation failures go to `quarantine` with the raw item and a reason. Never a silent skip, never a batch-fatal throw.
- **All sync operations are idempotent.** Re-running any sync or backfill over any range must not change the resulting data.
- **Neon driver API.** `@neondatabase/serverless` has NO `sql.query()` method. The value
  returned by `neon()` is called as a tagged template (``sql`...` ``) or directly as
  `sql(text, params, opts)`. Each call sends exactly ONE statement over HTTP -- a
  multi-statement string is rejected. Pass `{ fullResults: true }` when you need
  `rowCount`; without it the call resolves to a bare rows array.
- Node 20+. Package manager: `npm`.
- Timezone constant, used everywhere: `America/New_York`.

---

## File Structure

```
web/
  package.json                       deps + scripts
  tsconfig.json
  next.config.ts
  vercel.json                        cron schedule
  vitest.config.ts
  .env.example                       documents every required env var
  db/
    migrations/0001_init.sql         schema: events, trades, sync_state, quarantine, parity_runs, audit_log
    migrate.ts                       applies migrations in order, tracks applied set
  src/lib/
    db.ts                            Neon client, single export
    ddb.ts                           DynamoDB doc client, read-only
    time.ts                          NY timezone helpers, single source of truth
    sync/
      types.ts                       EventRow, TradeRow, SyncResult
      mapEvent.ts                    DDB item -> EventRow | QuarantineReason   (pure)
      mapTrade.ts                    DDB item -> TradeRow | QuarantineReason   (pure)
      partitions.ts                  ensure monthly partitions exist
      quarantine.ts                  insert quarantined rows
      watermark.ts                   read / advance sync_state
      insertEvents.ts                bulk upsert events
      insertTrades.ts                bulk upsert trades
      syncEvents.ts                  orchestration, injected fetcher
      syncTrades.ts                  orchestration, injected fetcher
    ddbFetchers.ts                   real network fetchers (thin, untested by unit tests)
  src/app/api/cron/sync/route.ts     cron entrypoint
  scripts/backfill.ts                resumable historical backfill
  scripts/verify.ts                  row-count + spot-value parity vs DynamoDB
  test/
    mapEvent.test.ts
    mapTrade.test.ts
    partitions.test.ts
    watermark.test.ts
    syncEvents.test.ts
    syncTrades.test.ts
    timezone.test.ts                 proves the harness catches Global Constraint #1
```

**Boundary rationale:** mapping is pure and heavily tested; orchestration takes an injected
fetcher so it is testable without network (mirroring the existing `hl_volume.compute_perp_volume`
pattern in this repo); network fetchers are thin and isolated so the untestable surface is as
small as possible.

---

## Task 0: Provisioning (requires human action)

**This task is not code.** It gathers credentials the remaining tasks need. Flag to the user and wait.

- [ ] **Step 1: Create the Neon project**

Via the Vercel dashboard (Storage → Neon) or the Neon console. Create a project named
`freeport-analytics`, region `us-east-1` to sit near the DynamoDB tables. Launch tier
(~$19/mo, 10 GB) — the free 0.5 GB tier cannot hold the 464 MB backfill plus indexes.

Create two branches: `main` (production) and `test` (used by integration tests, so tests can
never touch production data).

Record: `DATABASE_URL` (main branch, pooled) and `TEST_DATABASE_URL` (test branch, pooled).

- [ ] **Step 2: Create a read-only AWS IAM user**

The dashboard must be structurally incapable of writing to DynamoDB (Global Constraint #2).

```bash
aws iam create-user --user-name freeport-analytics-dashboard-ro
aws iam put-user-policy --user-name freeport-analytics-dashboard-ro \
  --policy-name analytics-readonly --policy-document '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["dynamodb:Query", "dynamodb:GetItem", "dynamodb:DescribeTable"],
    "Resource": [
      "arn:aws:dynamodb:us-east-1:273354655279:table/freeport-analytics-events",
      "arn:aws:dynamodb:us-east-1:273354655279:table/freeport-analytics-events/index/*",
      "arn:aws:dynamodb:us-east-1:273354655279:table/freeport-trades-history",
      "arn:aws:dynamodb:us-east-1:273354655279:table/freeport-trades-history/index/*"
    ]
  }]
}'
aws iam create-access-key --user-name freeport-analytics-dashboard-ro
```

Record `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

- [ ] **Step 3: Verify the policy actually denies writes**

```bash
AWS_ACCESS_KEY_ID=<new> AWS_SECRET_ACCESS_KEY=<new> aws dynamodb put-item \
  --table-name freeport-analytics-events --region us-east-1 \
  --item '{"date":{"S":"1970-01-01"},"sk":{"S":"iam-denial-probe"}}'
```

Expected: `AccessDeniedException`. **If this succeeds, stop — the guarantee is broken.** Delete
the probe row and fix the policy before continuing.

- [ ] **Step 4: Create the Vercel project**

Link `analytics-dashboard`, set **Root Directory to `web`**. Do not deploy yet.

Record the project name. Confirm the plan supports 60-second cron (Hobby is limited to daily
crons; this needs Pro).

---

## Task 1: Scaffold + schema

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/next.config.ts`, `web/vitest.config.ts`, `web/.env.example`, `web/.gitignore`
- Create: `web/db/migrations/0001_init.sql`, `web/db/migrate.ts`, `web/src/lib/db.ts`

**Interfaces:**
- Produces: `sql` — the Neon tagged-template query function, exported from `web/src/lib/db.ts`. Every later task imports this.
- Produces: the schema all later tasks insert into.

- [ ] **Step 1: Create `web/package.json`**

```json
{
  "name": "freeport-analytics-web",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "test": "vitest run",
    "test:watch": "vitest",
    "migrate": "tsx db/migrate.ts",
    "backfill": "tsx scripts/backfill.ts",
    "verify": "tsx scripts/verify.ts"
  },
  "dependencies": {
    "@aws-sdk/client-dynamodb": "^3.700.0",
    "@aws-sdk/lib-dynamodb": "^3.700.0",
    "@neondatabase/serverless": "^0.10.4",
    "next": "^15.1.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@types/node": "^22.10.0",
    "@types/react": "^19.0.0",
    "tsx": "^4.19.0",
    "typescript": "^5.7.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 2: Create `web/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM"],
    "module": "esnext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "jsx": "preserve",
    "incremental": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 3: Create `web/next.config.ts` and `web/.gitignore`**

`web/next.config.ts`:
```typescript
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {}

export default nextConfig
```

`web/.gitignore`:
```
node_modules
.next
.env
.env.local
next-env.d.ts
```

- [ ] **Step 4: Create `web/vitest.config.ts`**

```typescript
import { defineConfig } from 'vitest/config'
import path from 'node:path'

export default defineConfig({
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
})
```

- [ ] **Step 5: Create `web/.env.example`**

```bash
# Neon — main branch, pooled connection string
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
# Neon — test branch. Integration tests refuse to run against DATABASE_URL.
TEST_DATABASE_URL=postgresql://user:pass@host/db-test?sslmode=require

# Read-only IAM user (dynamodb:Query, GetItem, DescribeTable only)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Shared secret; the cron route rejects requests without it
CRON_SECRET=

ANALYTICS_TABLE=freeport-analytics-events
TRADES_TABLE=freeport-trades-history
```

- [ ] **Step 6: Create `web/db/migrations/0001_init.sql`**

```sql
-- Analytics v2 Phase 1 schema.
-- NOTE: `date` is the DynamoDB partition key, mirrored for provenance and
-- partitioning ONLY. It is UTC-derived. Never GROUP BY or range-filter a metric
-- on it -- always use (ts AT TIME ZONE 'America/New_York'). See spec risk #1.

CREATE TABLE IF NOT EXISTS events (
  date            date        NOT NULL,
  sk              text        NOT NULL,
  ts              timestamptz NOT NULL,
  event           text        NOT NULL,
  screen          text,
  component       text,
  wallet_address  text,
  session_id      text,
  platform        text,
  app_version     text,
  metadata        jsonb,
  synced_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (date, sk)
) PARTITION BY RANGE (date);

-- Backstop so an insert can never fail on a missing partition.
-- A non-empty default partition is an alert condition (see scripts/verify.ts).
CREATE TABLE IF NOT EXISTS events_default PARTITION OF events DEFAULT;

CREATE TABLE IF NOT EXISTS trades (
  wallet_address     text        NOT NULL,
  timestamp          text        NOT NULL,
  ts                 timestamptz NOT NULL,
  trade_date         date,
  id                 text,
  type               text,
  amount_usd         numeric,
  status             text,
  source             text,
  client             text,
  -- swap fields
  from_token         text,
  from_mint          text,
  to_token           text,
  to_mint            text,
  amount_from_token  numeric,
  amount_to_token    numeric,
  tx_signature       text,
  request_id         text,
  tweet_handle       text,
  tweet_ticker       text,
  tweet_timestamp    text,
  -- perps fields
  asset              text,
  display_symbol     text,
  side               text,
  size               numeric,
  price              numeric,
  leverage           numeric,
  order_type         text,
  is_close           boolean,
  is_hip3            boolean,
  category           text,
  trace_id           text,
  raw                jsonb       NOT NULL,
  synced_at          timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (wallet_address, timestamp)
);

CREATE INDEX IF NOT EXISTS trades_ts_idx        ON trades (ts);
CREATE INDEX IF NOT EXISTS trades_type_ts_idx   ON trades (type, ts);
CREATE INDEX IF NOT EXISTS trades_client_ts_idx ON trades (client, ts);

CREATE TABLE IF NOT EXISTS sync_state (
  source        text PRIMARY KEY,
  watermark_ts  timestamptz NOT NULL,
  updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quarantine (
  id          bigserial PRIMARY KEY,
  source      text        NOT NULL,
  raw         jsonb       NOT NULL,
  reason      text        NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS parity_runs (
  id               bigserial PRIMARY KEY,
  run_at           timestamptz NOT NULL DEFAULT now(),
  metric           text        NOT NULL,
  dims             jsonb,
  streamlit_value  numeric,
  postgres_value   numeric,
  abs_diff         numeric,
  pct_diff         numeric,
  passed           boolean     NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
  id           bigserial PRIMARY KEY,
  actor_email  text        NOT NULL,
  action       text        NOT NULL,
  target       text,
  payload      jsonb,
  created_at   timestamptz NOT NULL DEFAULT now()
);
```

Note: `events` indexes are created per-partition by `partitions.ts` in Task 4, not here —
indexes on a partitioned parent propagate but are cheaper to manage per-partition at this scale.

- [ ] **Step 7: Create `web/src/lib/db.ts`**

```typescript
import { neon } from '@neondatabase/serverless'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')

export const sql = neon(url)
```

- [ ] **Step 8: Create `web/db/migrate.ts`**

```typescript
import { neon } from '@neondatabase/serverless'
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')
const sql = neon(url)

/**
 * Split a migration file into individual statements.
 *
 * Neon's HTTP driver executes exactly ONE statement per call and rejects
 * multi-statement strings, so a migration file cannot be sent as a single query.
 *
 * This splitter is deliberately simple: it strips `--` line comments and splits
 * on semicolons. That is sufficient for plain DDL and is all this project's
 * migrations contain. If a future migration introduces a function body, a
 * dollar-quoted string, or a semicolon inside a string literal, this splitter
 * MUST be replaced with a real parser -- it will silently split such a file in
 * the wrong place.
 */
export function splitStatements(sqlText: string): string[] {
  return sqlText
    .split('\n')
    .map(line => line.replace(/--.*$/, ''))
    .join('\n')
    .split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0)
}

async function main() {
  await sql`CREATE TABLE IF NOT EXISTS _migrations (
    name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now()
  )`
  const applied = new Set(
    (await sql`SELECT name FROM _migrations`).map((r: any) => r.name as string)
  )
  const dir = path.join(import.meta.dirname, 'migrations')
  for (const file of readdirSync(dir).filter(f => f.endsWith('.sql')).sort()) {
    if (applied.has(file)) {
      console.log(`skip ${file}`)
      continue
    }
    console.log(`apply ${file}`)
    for (const stmt of splitStatements(readFileSync(path.join(dir, file), 'utf8'))) {
      await sql(stmt)
    }
    await sql`INSERT INTO _migrations (name) VALUES (${file})`
  }
  console.log('migrations complete')
}

main().catch(e => { console.error(e); process.exit(1) })
```

- [ ] **Step 9: Install and run the migration against the test branch**

```bash
cd web && npm install
DATABASE_URL="$TEST_DATABASE_URL" npm run migrate
```

Expected: `apply 0001_init.sql` then `migrations complete`. Re-run it; expected: `skip 0001_init.sql`.

- [ ] **Step 10: Commit**

```bash
git add web
git commit -m "feat(web): scaffold Next.js app and Phase 1 Postgres schema"
```

---

## Task 2: Event mapping (pure)

**Files:**
- Create: `web/src/lib/sync/types.ts`, `web/src/lib/sync/mapEvent.ts`
- Test: `web/test/mapEvent.test.ts`

**Interfaces:**
- Produces: `type EventRow`, `type MapResult<T>`, `mapEvent(item: unknown): MapResult<EventRow>`.
  Consumed by Task 6 (`syncEvents`) and Task 9 (backfill).

- [ ] **Step 1: Write the failing test**

`web/test/mapEvent.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { mapEvent } from '@/lib/sync/mapEvent'

// Shape produced by Swap_Server/src/services/analytics.ts:70-92
const valid = {
  date: '2026-07-19',
  sk: '2026-07-19T14:32:11.482Z#0xabcdef#k3j4h5g6',
  event: 'trade_success',
  screen: 'trade',
  component: 'SwapModal',
  wallet_address: '0xABCDEF0123456789',
  session_id: 'm4k2j1-x9',
  timestamp: '2026-07-19T14:32:11.482Z',
  metadata: { asset: 'BTC', amount_usd: 250.5 },
  platform: 'ios',
  app_version: '1.4.2',
  hour: 14,
  day_of_week: 0,
}

describe('mapEvent', () => {
  it('maps a well-formed item', () => {
    const r = mapEvent(valid)
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.date).toBe('2026-07-19')
    expect(r.value.sk).toBe(valid.sk)
    expect(r.value.ts.toISOString()).toBe('2026-07-19T14:32:11.482Z')
    expect(r.value.event).toBe('trade_success')
    expect(r.value.metadata).toEqual({ asset: 'BTC', amount_usd: 250.5 })
  })

  it('preserves wallet_address case exactly', () => {
    // Downstream joins lowercase explicitly; the mirror must not pre-normalize
    // or it stops being a faithful copy of the source row.
    const r = mapEvent(valid)
    expect(r.ok && r.value.wallet_address).toBe('0xABCDEF0123456789')
  })

  it('drops the denormalized UTC hour and day_of_week', () => {
    // These are UTC-derived; the dashboard must recompute in America/New_York.
    // Carrying them forward invites a GROUP BY on the wrong value (spec risk #1).
    const r = mapEvent(valid)
    expect(r.ok && 'hour' in (r.value as object)).toBe(false)
    expect(r.ok && 'day_of_week' in (r.value as object)).toBe(false)
  })

  it('allows absent optional fields', () => {
    const r = mapEvent({
      date: '2026-07-19', sk: 'a', event: 'session_start',
      wallet_address: '0x1', session_id: 's', timestamp: '2026-07-19T00:00:00.000Z',
    })
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.screen).toBeNull()
    expect(r.value.platform).toBeNull()
    expect(r.value.metadata).toBeNull()
  })

  it('quarantines a missing required field', () => {
    const { date, ...noDate } = valid
    const r = mapEvent(noDate)
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/date/)
  })

  it('quarantines an unparseable timestamp', () => {
    const r = mapEvent({ ...valid, timestamp: 'not-a-date' })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/timestamp/)
  })

  it('quarantines a non-object', () => {
    expect(mapEvent(null).ok).toBe(false)
    expect(mapEvent('nope').ok).toBe(false)
  })

  it('quarantines non-string metadata rather than coercing', () => {
    const r = mapEvent({ ...valid, metadata: 'a string' })
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/metadata/)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd web && npx vitest run test/mapEvent.test.ts
```
Expected: FAIL — `Failed to resolve import "@/lib/sync/mapEvent"`.

- [ ] **Step 3: Write `web/src/lib/sync/types.ts`**

```typescript
export type MapResult<T> =
  | { ok: true; value: T }
  | { ok: false; reason: string }

export interface EventRow {
  date: string            // YYYY-MM-DD, UTC-derived. Storage only -- never a GROUP BY key.
  sk: string
  ts: Date
  event: string
  screen: string | null
  component: string | null
  wallet_address: string | null
  session_id: string | null
  platform: string | null
  app_version: string | null
  metadata: Record<string, unknown> | null
}

export interface SyncResult {
  scanned: number
  inserted: number
  quarantined: number
  watermark: string
}
```

- [ ] **Step 4: Write `web/src/lib/sync/mapEvent.ts`**

```typescript
import type { EventRow, MapResult } from './types'

const REQUIRED = ['date', 'sk', 'event', 'timestamp'] as const

function optString(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null
}

/**
 * Map a raw DynamoDB item from `freeport-analytics-events` to an EventRow.
 *
 * The source shape is written by Swap_Server/src/services/analytics.ts:70-92.
 * The denormalized `hour` and `day_of_week` columns are deliberately dropped:
 * they are UTC-derived, while every metric in this project buckets in
 * America/New_York. Carrying them forward would invite grouping on the wrong
 * value (spec risk #1).
 */
export function mapEvent(item: unknown): MapResult<EventRow> {
  if (typeof item !== 'object' || item === null) {
    return { ok: false, reason: 'item is not an object' }
  }
  const o = item as Record<string, unknown>

  for (const f of REQUIRED) {
    if (typeof o[f] !== 'string' || (o[f] as string).length === 0) {
      return { ok: false, reason: `missing or non-string required field: ${f}` }
    }
  }

  const ts = new Date(o.timestamp as string)
  if (Number.isNaN(ts.getTime())) {
    return { ok: false, reason: `unparseable timestamp: ${String(o.timestamp)}` }
  }

  let metadata: Record<string, unknown> | null = null
  if (o.metadata !== undefined && o.metadata !== null) {
    if (typeof o.metadata !== 'object' || Array.isArray(o.metadata)) {
      return { ok: false, reason: 'metadata is present but not an object' }
    }
    metadata = o.metadata as Record<string, unknown>
  }

  return {
    ok: true,
    value: {
      date: o.date as string,
      sk: o.sk as string,
      ts,
      event: o.event as string,
      screen: optString(o.screen),
      component: optString(o.component),
      wallet_address: optString(o.wallet_address),
      session_id: optString(o.session_id),
      platform: optString(o.platform),
      app_version: optString(o.app_version),
      metadata,
    },
  }
}
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd web && npx vitest run test/mapEvent.test.ts
```
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/sync web/test/mapEvent.test.ts
git commit -m "feat(sync): pure event mapping with quarantine results"
```

---

## Task 3: Timezone guard test

This task exists to prove the test suite would actually catch Global Constraint #1, rather than
assuming it. It is deliberately separate and deliberately early — every later task depends on
this invariant holding.

**Files:**
- Create: `web/src/lib/time.ts`
- Test: `web/test/timezone.test.ts`

**Interfaces:**
- Produces: `NY_TZ`, `nyDateExpr(tsColumn: string): string`. Consumed by every metric query in Phase 2.

- [ ] **Step 1: Write the failing test**

`web/test/timezone.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { NY_TZ, nyDateExpr, utcDateOf, nyDateOf } from '@/lib/time'

describe('timezone handling', () => {
  it('uses America/New_York', () => {
    expect(NY_TZ).toBe('America/New_York')
  })

  it('builds a NY-bucketing SQL expression', () => {
    expect(nyDateExpr('ts')).toBe("(ts AT TIME ZONE 'America/New_York')::date")
  })

  // THE guard. An event at 01:30 UTC on Jul 20 is 21:30 EDT on Jul 19.
  // The DynamoDB partition key says 2026-07-20; every metric must say 2026-07-19.
  // If this ever passes as equal, the highest-severity risk in the spec is live.
  it('proves the UTC partition key disagrees with the NY bucket', () => {
    const ts = new Date('2026-07-20T01:30:00.000Z')
    expect(utcDateOf(ts)).toBe('2026-07-20')
    expect(nyDateOf(ts)).toBe('2026-07-19')
    expect(utcDateOf(ts)).not.toBe(nyDateOf(ts))
  })

  it('agrees mid-afternoon, when both fall on the same calendar day', () => {
    const ts = new Date('2026-07-20T18:00:00.000Z')
    expect(utcDateOf(ts)).toBe(nyDateOf(ts))
  })

  it('handles the EST/EDT boundary', () => {
    // 2026-11-01 06:30Z is 02:30 EDT (UTC-4), still Nov 1 in NY.
    expect(nyDateOf(new Date('2026-11-01T06:30:00.000Z'))).toBe('2026-11-01')
    // 2026-01-15 02:30Z is 21:30 EST (UTC-5) on Jan 14.
    expect(nyDateOf(new Date('2026-01-15T02:30:00.000Z'))).toBe('2026-01-14')
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd web && npx vitest run test/timezone.test.ts
```
Expected: FAIL — cannot resolve `@/lib/time`.

- [ ] **Step 3: Write `web/src/lib/time.ts`**

```typescript
export const NY_TZ = 'America/New_York'

/**
 * SQL expression that buckets a timestamptz column into a New York calendar date.
 *
 * Use this for EVERY metric. The mirrored `date` column is the DynamoDB partition
 * key -- UTC-derived, storage only. Grouping on it shifts daily figures by up to
 * 5 hours in a way that looks entirely plausible (spec risk #1).
 */
export function nyDateExpr(tsColumn: string): string {
  return `(${tsColumn} AT TIME ZONE '${NY_TZ}')::date`
}

export function utcDateOf(ts: Date): string {
  return ts.toISOString().slice(0, 10)
}

export function nyDateOf(ts: Date): string {
  // en-CA yields YYYY-MM-DD.
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: NY_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(ts)
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd web && npx vitest run test/timezone.test.ts
```
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/time.ts web/test/timezone.test.ts
git commit -m "feat(time): NY timezone helpers with UTC-vs-NY divergence guard"
```

---

## Task 4: Partition management

**Files:**
- Create: `web/src/lib/sync/partitions.ts`
- Test: `web/test/partitions.test.ts`

**Interfaces:**
- Produces: `partitionNameFor(date: string): string`, `partitionBoundsFor(date: string): {from: string, to: string}`, `ensurePartitions(sql, dates: string[]): Promise<string[]>`. Consumed by Tasks 6 and 9.

- [ ] **Step 1: Write the failing test**

`web/test/partitions.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { partitionNameFor, partitionBoundsFor } from '@/lib/sync/partitions'

describe('partition naming', () => {
  it('names a partition per month', () => {
    expect(partitionNameFor('2026-07-19')).toBe('events_2026_07')
    expect(partitionNameFor('2026-01-01')).toBe('events_2026_01')
  })

  it('computes half-open month bounds', () => {
    expect(partitionBoundsFor('2026-07-19')).toEqual({ from: '2026-07-01', to: '2026-08-01' })
  })

  it('rolls the year over in December', () => {
    expect(partitionBoundsFor('2026-12-31')).toEqual({ from: '2026-12-01', to: '2027-01-01' })
  })

  it('rejects a malformed date rather than generating a bad DDL identifier', () => {
    expect(() => partitionNameFor('nope')).toThrow(/invalid date/i)
    expect(() => partitionNameFor("2026-07-01'; DROP TABLE events; --")).toThrow(/invalid date/i)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd web && npx vitest run test/partitions.test.ts
```
Expected: FAIL — cannot resolve `@/lib/sync/partitions`.

- [ ] **Step 3: Write `web/src/lib/sync/partitions.ts`**

```typescript
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/

function assertDate(date: string): void {
  // Partition names and bounds go into DDL, which cannot be parameterized.
  // Validate strictly so nothing but a literal date can ever reach the string.
  if (!DATE_RE.test(date)) throw new Error(`invalid date: ${date}`)
  const d = new Date(`${date}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) throw new Error(`invalid date: ${date}`)
}

export function partitionNameFor(date: string): string {
  assertDate(date)
  return `events_${date.slice(0, 4)}_${date.slice(5, 7)}`
}

export function partitionBoundsFor(date: string): { from: string; to: string } {
  assertDate(date)
  const year = Number(date.slice(0, 4))
  const month = Number(date.slice(5, 7))
  const from = `${year}-${String(month).padStart(2, '0')}-01`
  const nextYear = month === 12 ? year + 1 : year
  const nextMonth = month === 12 ? 1 : month + 1
  const to = `${nextYear}-${String(nextMonth).padStart(2, '0')}-01`
  return { from, to }
}

import type { neon } from '@neondatabase/serverless'

type SqlTag = ReturnType<typeof neon>

/**
 * Create any missing monthly partitions plus their indexes.
 * Idempotent -- safe to call on every sync tick.
 *
 * NOTE ON THE NEON API: `@neondatabase/serverless` has NO `sql.query()` method.
 * The function returned by `neon()` is called either as a tagged template
 * (sql`...`) or directly as sql(text, params, opts). Each call sends exactly ONE
 * statement over HTTP -- multi-statement strings are rejected. Every raw query in
 * this codebase therefore uses the sql(text, params) form, one statement per call.
 */
export async function ensurePartitions(sql: SqlTag, dates: string[]): Promise<string[]> {
  const names = new Set<string>()
  for (const d of dates) names.add(partitionNameFor(d))

  const created: string[] = []
  for (const name of names) {
    const date = `${name.slice(7, 11)}-${name.slice(12, 14)}-01`
    const { from, to } = partitionBoundsFor(date)
    // DDL cannot be parameterized; `name`, `from`, and `to` are all derived from
    // a string already validated against DATE_RE by assertDate().
    await sql(
      `CREATE TABLE IF NOT EXISTS ${name} PARTITION OF events
       FOR VALUES FROM ('${from}') TO ('${to}')`
    )
    await sql(`CREATE INDEX IF NOT EXISTS ${name}_event_ts_idx   ON ${name} (event, ts)`)
    await sql(`CREATE INDEX IF NOT EXISTS ${name}_wallet_ts_idx  ON ${name} (wallet_address, ts)`)
    await sql(`CREATE INDEX IF NOT EXISTS ${name}_session_ts_idx ON ${name} (session_id, ts)`)
    created.push(name)
  }
  return created
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd web && npx vitest run test/partitions.test.ts
```
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/sync/partitions.ts web/test/partitions.test.ts
git commit -m "feat(sync): idempotent monthly partition management"
```

---

## Task 5: Watermark

**Files:**
- Create: `web/src/lib/sync/watermark.ts`
- Test: `web/test/watermark.test.ts`

**Interfaces:**
- Produces: `WATERMARK_LAG_MS`, `readWatermark(sql, source, fallback): Promise<Date>`, `computeNextWatermark(now: Date): Date`, `advanceWatermark(sql, source, ts): Promise<void>`. Consumed by Tasks 6 and 7.

- [ ] **Step 1: Write the failing test**

`web/test/watermark.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { computeNextWatermark, WATERMARK_LAG_MS, datesToScan } from '@/lib/sync/watermark'

describe('watermark', () => {
  it('lags 10 minutes behind now', () => {
    expect(WATERMARK_LAG_MS).toBe(10 * 60 * 1000)
    const now = new Date('2026-07-20T14:00:00.000Z')
    expect(computeNextWatermark(now).toISOString()).toBe('2026-07-20T13:50:00.000Z')
  })

  it('never advances to now, so late arrivals are still captured', () => {
    const now = new Date('2026-07-20T14:00:00.000Z')
    expect(computeNextWatermark(now).getTime()).toBeLessThan(now.getTime())
  })

  it('scans the current and previous UTC date partitions', () => {
    // Events near UTC midnight land in the next partition. app.py:631-646 fetches
    // an extra day for the same reason.
    expect(datesToScan(new Date('2026-07-20T00:05:00.000Z')))
      .toEqual(['2026-07-19', '2026-07-20'])
  })

  it('rolls across a month boundary', () => {
    expect(datesToScan(new Date('2026-08-01T00:05:00.000Z')))
      .toEqual(['2026-07-31', '2026-08-01'])
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd web && npx vitest run test/watermark.test.ts
```
Expected: FAIL — cannot resolve `@/lib/sync/watermark`.

- [ ] **Step 3: Write `web/src/lib/sync/watermark.ts`**

```typescript
import { utcDateOf } from '@/lib/time'

/**
 * How far behind wall-clock the watermark is held.
 *
 * The FreeApp client flushes every 30s and requeues up to 500 events on failure
 * (FreeApp/services/AnalyticsService.ts:5-7, :150-153), so an event's `timestamp`
 * can precede its arrival by minutes. Advancing the watermark to `now` would step
 * over those rows permanently.
 */
export const WATERMARK_LAG_MS = 10 * 60 * 1000

export function computeNextWatermark(now: Date): Date {
  return new Date(now.getTime() - WATERMARK_LAG_MS)
}

/** Current and previous UTC date partitions -- events near UTC midnight land in the next one. */
export function datesToScan(now: Date): string[] {
  const prev = new Date(now.getTime() - 24 * 60 * 60 * 1000)
  return [utcDateOf(prev), utcDateOf(now)]
}

type SqlTag = ReturnType<typeof import('@neondatabase/serverless').neon>

export async function readWatermark(sql: SqlTag, source: string, fallback: Date): Promise<Date> {
  const rows = (await sql`
    SELECT watermark_ts FROM sync_state WHERE source = ${source}
  `) as Array<{ watermark_ts: string | Date }>
  if (rows.length === 0) return fallback
  return new Date(rows[0]!.watermark_ts)
}

export async function advanceWatermark(sql: SqlTag, source: string, ts: Date): Promise<void> {
  // GREATEST guards against a concurrent or replayed run moving the watermark backwards.
  await sql`
    INSERT INTO sync_state (source, watermark_ts, updated_at)
    VALUES (${source}, ${ts.toISOString()}, now())
    ON CONFLICT (source) DO UPDATE
      SET watermark_ts = GREATEST(sync_state.watermark_ts, EXCLUDED.watermark_ts),
          updated_at   = now()
  `
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd web && npx vitest run test/watermark.test.ts
```
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/sync/watermark.ts web/test/watermark.test.ts
git commit -m "feat(sync): watermark with 10-minute lag for late arrivals"
```

---

## Task 6: Event sync orchestration

**Files:**
- Create: `web/src/lib/sync/quarantine.ts`, `web/src/lib/sync/insertEvents.ts`, `web/src/lib/sync/syncEvents.ts`
- Test: `web/test/syncEvents.test.ts`

**Interfaces:**
- Consumes: `mapEvent` (Task 2), `ensurePartitions` (Task 4), `computeNextWatermark` / `datesToScan` / `readWatermark` / `advanceWatermark` (Task 5).
- Produces: `type EventFetcher = (date: string, afterSk: string) => Promise<unknown[]>`, `syncEvents(deps): Promise<SyncResult>`. Consumed by Tasks 8 and 9.

- [ ] **Step 1: Write the failing test**

`web/test/syncEvents.test.ts`:
```typescript
import { describe, it, expect, vi } from 'vitest'
import { syncEvents } from '@/lib/sync/syncEvents'

function item(overrides: Record<string, unknown> = {}) {
  return {
    date: '2026-07-20',
    sk: '2026-07-20T12:00:00.000Z#0xabc#aaaa1111',
    event: 'session_start',
    wallet_address: '0xabc',
    session_id: 's1',
    timestamp: '2026-07-20T12:00:00.000Z',
    ...overrides,
  }
}

function deps(items: unknown[]) {
  const inserted: unknown[][] = []
  const quarantined: Array<{ raw: unknown; reason: string }> = []
  return {
    inserted,
    quarantined,
    args: {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      ensurePartitions: vi.fn(async () => []),
      fetchEvents: vi.fn(async () => items),
      insertEvents: vi.fn(async (rows: unknown[]) => { inserted.push(rows); return rows.length }),
      quarantine: vi.fn(async (raw: unknown, reason: string) => { quarantined.push({ raw, reason }) }),
    },
  }
}

describe('syncEvents', () => {
  it('inserts mapped rows and reports counts', async () => {
    const d = deps([item(), item({ sk: 'b', event: 'trade_success' })])
    const r = await syncEvents(d.args as never)
    // fetchEvents is called once per scanned date (previous + current UTC day)
    expect(r.scanned).toBe(4)
    expect(r.inserted).toBe(4)
    expect(r.quarantined).toBe(0)
  })

  it('quarantines bad rows without failing the batch', async () => {
    const d = deps([item(), { garbage: true }])
    const r = await syncEvents(d.args as never)
    expect(r.quarantined).toBe(2)
    expect(r.inserted).toBe(2)
    expect(d.quarantined[0]!.reason).toMatch(/missing/)
  })

  it('advances the watermark to now minus the lag, not to now', async () => {
    const d = deps([item()])
    await syncEvents(d.args as never)
    const arg = (d.args.advanceWatermark as ReturnType<typeof vi.fn>).mock.calls[0]![1] as Date
    expect(arg.toISOString()).toBe('2026-07-20T13:50:00.000Z')
  })

  it('ensures partitions before inserting', async () => {
    const d = deps([item()])
    await syncEvents(d.args as never)
    expect(d.args.ensurePartitions).toHaveBeenCalledWith(['2026-07-19', '2026-07-20'])
  })

  it('does not advance the watermark when an insert throws', async () => {
    const d = deps([item()])
    d.args.insertEvents = vi.fn(async () => { throw new Error('neon down') })
    await expect(syncEvents(d.args as never)).rejects.toThrow('neon down')
    expect(d.args.advanceWatermark).not.toHaveBeenCalled()
  })

  it('is a no-op when there is nothing new', async () => {
    const d = deps([])
    const r = await syncEvents(d.args as never)
    expect(r.inserted).toBe(0)
    // The watermark still advances -- an empty window is a successfully synced window.
    expect(d.args.advanceWatermark).toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd web && npx vitest run test/syncEvents.test.ts
```
Expected: FAIL — cannot resolve `@/lib/sync/syncEvents`.

- [ ] **Step 3: Write `web/src/lib/sync/quarantine.ts`**

```typescript
import type { neon } from '@neondatabase/serverless'

type SqlTag = ReturnType<typeof neon>

export async function quarantineRow(
  sql: SqlTag, source: string, raw: unknown, reason: string
): Promise<void> {
  await sql`
    INSERT INTO quarantine (source, raw, reason)
    VALUES (${source}, ${JSON.stringify(raw)}::jsonb, ${reason})
  `
}
```

- [ ] **Step 4: Write `web/src/lib/sync/insertEvents.ts`**

```typescript
import type { neon } from '@neondatabase/serverless'
import type { EventRow } from './types'

type SqlTag = ReturnType<typeof neon>

const CHUNK = 500

/**
 * Bulk upsert events. ON CONFLICT DO NOTHING -- events are append-only, and a
 * re-read of an already-synced window must be a no-op (idempotency requirement).
 */
export async function insertEvents(sql: SqlTag, rows: EventRow[]): Promise<number> {
  let total = 0
  for (let i = 0; i < rows.length; i += CHUNK) {
    const chunk = rows.slice(i, i + CHUNK)
    // `fullResults: true` is REQUIRED. Without it neon resolves to a bare rows
    // array and rowCount is undefined -- the function would then report the
    // ATTEMPTED count rather than the count actually inserted, silently hiding
    // how many rows ON CONFLICT skipped. On a project whose entire premise is
    // data correctness, the sync must not overstate what it wrote.
    const result = (await sql(
      `INSERT INTO events (date, sk, ts, event, screen, component,
                           wallet_address, session_id, platform, app_version, metadata)
       SELECT * FROM UNNEST(
         $1::date[], $2::text[], $3::timestamptz[], $4::text[], $5::text[], $6::text[],
         $7::text[], $8::text[], $9::text[], $10::text[], $11::jsonb[]
       )
       ON CONFLICT (date, sk) DO NOTHING`,
      [
        chunk.map(r => r.date),
        chunk.map(r => r.sk),
        chunk.map(r => r.ts.toISOString()),
        chunk.map(r => r.event),
        chunk.map(r => r.screen),
        chunk.map(r => r.component),
        chunk.map(r => r.wallet_address),
        chunk.map(r => r.session_id),
        chunk.map(r => r.platform),
        chunk.map(r => r.app_version),
        chunk.map(r => (r.metadata === null ? null : JSON.stringify(r.metadata))),
      ],
      { fullResults: true }
    )) as unknown as { rowCount: number }
    total += result.rowCount
  }
  return total
}
```

- [ ] **Step 5: Write `web/src/lib/sync/syncEvents.ts`**

```typescript
import { mapEvent } from './mapEvent'
import { computeNextWatermark, datesToScan } from './watermark'
import type { EventRow, SyncResult } from './types'

export type EventFetcher = (date: string, afterSk: string) => Promise<unknown[]>

export interface SyncEventsDeps {
  now: Date
  readWatermark: () => Promise<Date>
  advanceWatermark: (source: string, ts: Date) => Promise<void>
  ensurePartitions: (dates: string[]) => Promise<string[]>
  fetchEvents: EventFetcher
  insertEvents: (rows: EventRow[]) => Promise<number>
  quarantine: (raw: unknown, reason: string) => Promise<void>
}

export const EVENTS_SOURCE = 'events'

/**
 * One incremental sync tick.
 *
 * Order matters: partitions are ensured before any insert, and the watermark is
 * advanced only after every insert has succeeded. A throw anywhere leaves the
 * watermark untouched, so the next tick re-reads the same window -- safe because
 * inserts are ON CONFLICT DO NOTHING.
 */
export async function syncEvents(deps: SyncEventsDeps): Promise<SyncResult> {
  const watermark = await deps.readWatermark()
  const dates = datesToScan(deps.now)
  await deps.ensurePartitions(dates)

  // The sort key is `{timestamp}#{wallet8}#{rand8}`, so it sorts by time and a
  // bare ISO timestamp is a valid lower bound for a key-condition range query.
  const afterSk = watermark.toISOString()

  let scanned = 0
  let inserted = 0
  let quarantined = 0

  for (const date of dates) {
    const items = await deps.fetchEvents(date, afterSk)
    scanned += items.length

    const rows: EventRow[] = []
    for (const item of items) {
      const mapped = mapEvent(item)
      if (mapped.ok) rows.push(mapped.value)
      else {
        await deps.quarantine(item, mapped.reason)
        quarantined += 1
      }
    }
    if (rows.length > 0) inserted += await deps.insertEvents(rows)
  }

  const next = computeNextWatermark(deps.now)
  await deps.advanceWatermark(EVENTS_SOURCE, next)

  return { scanned, inserted, quarantined, watermark: next.toISOString() }
}
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd web && npx vitest run test/syncEvents.test.ts
```
Expected: PASS, 6 tests.

- [ ] **Step 7: Commit**

```bash
git add web/src/lib/sync web/test/syncEvents.test.ts
git commit -m "feat(sync): event sync orchestration with quarantine and safe watermark ordering"
```

---

## Task 7: Trade mapping and sync

**Files:**
- Create: `web/src/lib/sync/mapTrade.ts`, `web/src/lib/sync/insertTrades.ts`, `web/src/lib/sync/syncTrades.ts`
- Test: `web/test/mapTrade.test.ts`, `web/test/syncTrades.test.ts`

**Interfaces:**
- Produces: `type TradeRow`, `mapTrade(item: unknown): MapResult<TradeRow>`, `syncTrades(deps): Promise<SyncResult>`.

- [ ] **Step 1: Write the failing test**

`web/test/mapTrade.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { mapTrade } from '@/lib/sync/mapTrade'

// Perps shape: freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts:167-192
const perp = {
  wallet_address: '0xabc', timestamp: '2026-07-20T12:00:00.000Z', trade_date: '2026-07-20',
  id: '1753000000000-x9k2', type: 'perps', asset: 'BTC', display_symbol: 'BTC',
  side: 'long', size: 0.05, price: 61000, leverage: 5, amount_usd: 610,
  order_type: 'market', is_close: false, is_hip3: false, category: 'hl',
  status: 'success', source: 'perps', client: 'web', trace_id: 'tr-1',
}

// Swap shape: Swap_Server/src/services/dynamodb.ts:67-107
const swap = {
  wallet_address: '0xdef', timestamp: '2026-07-20T13:00:00.000Z', trade_date: '2026-07-20',
  id: 'sw-1', type: 'swap', amount_usd: 42.5, status: 'success', source: 'swap',
  from_token: 'USDC', to_token: 'SOL', amount_from_token: 42.5, amount_to_token: 0.3,
  tx_signature: 'sig123', request_id: 'req-1',
}

describe('mapTrade', () => {
  it('maps a perps row', () => {
    const r = mapTrade(perp)
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.type).toBe('perps')
    expect(r.value.amount_usd).toBe(610)
    expect(r.value.client).toBe('web')
    expect(r.value.leverage).toBe(5)
    expect(r.value.ts.toISOString()).toBe('2026-07-20T12:00:00.000Z')
  })

  it('maps a swap row, leaving perps fields null', () => {
    const r = mapTrade(swap)
    expect(r.ok).toBe(true)
    if (!r.ok) return
    expect(r.value.type).toBe('swap')
    expect(r.value.asset).toBeNull()
    expect(r.value.leverage).toBeNull()
    expect(r.value.from_token).toBe('USDC')
  })

  it('retains the full raw item', () => {
    // amount_usd on perps rows is MARGIN, not notional (trade-logger.ts:39-46),
    // and volume math may need fields this schema does not name. Keep everything.
    const r = mapTrade(perp)
    expect(r.ok && r.value.raw).toEqual(perp)
  })

  it('defaults a missing client to unknown, matching the writer', () => {
    const { client, ...noClient } = perp
    const r = mapTrade(noClient)
    expect(r.ok && r.value.client).toBe('unknown')
  })

  it('preserves numeric precision as a string-safe value', () => {
    const r = mapTrade({ ...perp, amount_usd: 0.1 + 0.2 })
    expect(r.ok && r.value.amount_usd).toBeCloseTo(0.30000000000000004, 15)
  })

  it('quarantines a missing key field', () => {
    const { wallet_address, ...bad } = perp
    const r = mapTrade(bad)
    expect(r.ok).toBe(false)
    if (r.ok) return
    expect(r.reason).toMatch(/wallet_address/)
  })

  it('quarantines an unparseable timestamp', () => {
    expect(mapTrade({ ...perp, timestamp: 'nope' }).ok).toBe(false)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd web && npx vitest run test/mapTrade.test.ts
```
Expected: FAIL — cannot resolve `@/lib/sync/mapTrade`.

- [ ] **Step 3: Write `web/src/lib/sync/mapTrade.ts`**

```typescript
import type { MapResult } from './types'

export interface TradeRow {
  wallet_address: string
  timestamp: string
  ts: Date
  trade_date: string | null
  id: string | null
  type: string | null
  amount_usd: number | null
  status: string | null
  source: string | null
  client: string
  from_token: string | null
  from_mint: string | null
  to_token: string | null
  to_mint: string | null
  amount_from_token: number | null
  amount_to_token: number | null
  tx_signature: string | null
  request_id: string | null
  tweet_handle: string | null
  tweet_ticker: string | null
  tweet_timestamp: string | null
  asset: string | null
  display_symbol: string | null
  side: string | null
  size: number | null
  price: number | null
  leverage: number | null
  order_type: string | null
  is_close: boolean | null
  is_hip3: boolean | null
  category: string | null
  trace_id: string | null
  raw: Record<string, unknown>
}

function optString(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null
}
function optNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}
function optBool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null
}

/**
 * Map a raw item from `freeport-trades-history`.
 *
 * Two writers produce this table with different field sets:
 *   perps -- freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts:167-192
 *   swaps -- Swap_Server/src/services/dynamodb.ts:67-107
 *
 * The full raw item is retained in `raw` because `amount_usd` on perps rows is
 * MARGIN rather than notional (trade-logger.ts:39-46), and volume math may need
 * fields this schema does not name explicitly.
 */
export function mapTrade(item: unknown): MapResult<TradeRow> {
  if (typeof item !== 'object' || item === null) {
    return { ok: false, reason: 'item is not an object' }
  }
  const o = item as Record<string, unknown>

  for (const f of ['wallet_address', 'timestamp'] as const) {
    if (typeof o[f] !== 'string' || (o[f] as string).length === 0) {
      return { ok: false, reason: `missing or non-string required field: ${f}` }
    }
  }

  const ts = new Date(o.timestamp as string)
  if (Number.isNaN(ts.getTime())) {
    return { ok: false, reason: `unparseable timestamp: ${String(o.timestamp)}` }
  }

  return {
    ok: true,
    value: {
      wallet_address: o.wallet_address as string,
      timestamp: o.timestamp as string,
      ts,
      trade_date: optString(o.trade_date),
      id: optString(o.id),
      type: optString(o.type),
      amount_usd: optNumber(o.amount_usd),
      status: optString(o.status),
      source: optString(o.source),
      // Matches the writer's own default at trade-logger.ts:189.
      client: optString(o.client) ?? 'unknown',
      from_token: optString(o.from_token),
      from_mint: optString(o.from_mint),
      to_token: optString(o.to_token),
      to_mint: optString(o.to_mint),
      amount_from_token: optNumber(o.amount_from_token),
      amount_to_token: optNumber(o.amount_to_token),
      tx_signature: optString(o.tx_signature),
      request_id: optString(o.request_id),
      tweet_handle: optString(o.tweet_handle),
      tweet_ticker: optString(o.tweet_ticker),
      tweet_timestamp: optString(o.tweet_timestamp),
      asset: optString(o.asset),
      display_symbol: optString(o.display_symbol),
      side: optString(o.side),
      size: optNumber(o.size),
      price: optNumber(o.price),
      leverage: optNumber(o.leverage),
      order_type: optString(o.order_type),
      is_close: optBool(o.is_close),
      is_hip3: optBool(o.is_hip3),
      category: optString(o.category),
      trace_id: optString(o.trace_id),
      raw: o,
    },
  }
}
```

- [ ] **Step 4: Run the mapping test to verify it passes**

```bash
cd web && npx vitest run test/mapTrade.test.ts
```
Expected: PASS, 7 tests.

- [ ] **Step 5: Write the failing sync test**

`web/test/syncTrades.test.ts`:
```typescript
import { describe, it, expect, vi } from 'vitest'
import { syncTrades } from '@/lib/sync/syncTrades'

const row = {
  wallet_address: '0xabc', timestamp: '2026-07-20T12:00:00.000Z',
  trade_date: '2026-07-20', type: 'perps', amount_usd: 610, status: 'success',
}

describe('syncTrades', () => {
  it('inserts mapped trades across both scanned dates', async () => {
    const args = {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      fetchTrades: vi.fn(async () => [row]),
      insertTrades: vi.fn(async (rows: unknown[]) => rows.length),
      quarantine: vi.fn(async () => {}),
    }
    const r = await syncTrades(args as never)
    expect(r.inserted).toBe(2)
    expect(args.advanceWatermark).toHaveBeenCalledWith('trades', expect.any(Date))
  })

  it('quarantines bad rows without failing the batch', async () => {
    const args = {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      fetchTrades: vi.fn(async () => [row, { nope: 1 }]),
      insertTrades: vi.fn(async (rows: unknown[]) => rows.length),
      quarantine: vi.fn(async () => {}),
    }
    const r = await syncTrades(args as never)
    expect(r.quarantined).toBe(2)
    expect(r.inserted).toBe(2)
  })

  it('does not advance the watermark when an insert throws', async () => {
    const args = {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      fetchTrades: vi.fn(async () => [row]),
      insertTrades: vi.fn(async () => { throw new Error('neon down') }),
      quarantine: vi.fn(async () => {}),
    }
    await expect(syncTrades(args as never)).rejects.toThrow('neon down')
    expect(args.advanceWatermark).not.toHaveBeenCalled()
  })
})
```

- [ ] **Step 6: Run it to verify it fails**

```bash
cd web && npx vitest run test/syncTrades.test.ts
```
Expected: FAIL — cannot resolve `@/lib/sync/syncTrades`.

- [ ] **Step 7: Write `web/src/lib/sync/insertTrades.ts`**

```typescript
import type { neon } from '@neondatabase/serverless'
import type { TradeRow } from './mapTrade'

type SqlTag = ReturnType<typeof neon>

const COLUMNS = [
  'wallet_address', 'timestamp', 'ts', 'trade_date', 'id', 'type', 'amount_usd',
  'status', 'source', 'client', 'from_token', 'from_mint', 'to_token', 'to_mint',
  'amount_from_token', 'amount_to_token', 'tx_signature', 'request_id',
  'tweet_handle', 'tweet_ticker', 'tweet_timestamp', 'asset', 'display_symbol',
  'side', 'size', 'price', 'leverage', 'order_type', 'is_close', 'is_hip3',
  'category', 'trace_id', 'raw',
] as const

/**
 * Upsert trades.
 *
 * DO UPDATE, not DO NOTHING: both writers use PutCommand, which overwrites in
 * place on a repeated (wallet_address, timestamp) key. DO NOTHING would pin the
 * mirror to a superseded version of the row (spec risk #7).
 */
export async function insertTrades(sql: SqlTag, rows: TradeRow[]): Promise<number> {
  if (rows.length === 0) return 0
  let total = 0
  for (let i = 0; i < rows.length; i += 500) {
    const chunk = rows.slice(i, i + 500)
    const values = chunk
      .map((_, r) => `(${COLUMNS.map((_, c) => `$${r * COLUMNS.length + c + 1}`).join(',')})`)
      .join(',')
    const params = chunk.flatMap(r => [
      r.wallet_address, r.timestamp, r.ts.toISOString(), r.trade_date, r.id, r.type,
      r.amount_usd, r.status, r.source, r.client, r.from_token, r.from_mint,
      r.to_token, r.to_mint, r.amount_from_token, r.amount_to_token, r.tx_signature,
      r.request_id, r.tweet_handle, r.tweet_ticker, r.tweet_timestamp, r.asset,
      r.display_symbol, r.side, r.size, r.price, r.leverage, r.order_type,
      r.is_close, r.is_hip3, r.category, r.trace_id, JSON.stringify(r.raw),
    ])
    const updates = COLUMNS
      .filter(c => c !== 'wallet_address' && c !== 'timestamp')
      .map(c => `${c} = EXCLUDED.${c}`)
      .join(', ')
    const result = (await sql(
      `INSERT INTO trades (${COLUMNS.join(',')}) VALUES ${values}
       ON CONFLICT (wallet_address, timestamp) DO UPDATE SET ${updates}, synced_at = now()`,
      params,
      { fullResults: true }
    )) as unknown as { rowCount: number }
    total += result.rowCount
  }
  return total
}
```

- [ ] **Step 8: Write `web/src/lib/sync/syncTrades.ts`**

```typescript
import { mapTrade, type TradeRow } from './mapTrade'
import { computeNextWatermark, datesToScan } from './watermark'
import type { SyncResult } from './types'

export type TradeFetcher = (tradeDate: string, afterTimestamp: string) => Promise<unknown[]>

export interface SyncTradesDeps {
  now: Date
  readWatermark: () => Promise<Date>
  advanceWatermark: (source: string, ts: Date) => Promise<void>
  fetchTrades: TradeFetcher
  insertTrades: (rows: TradeRow[]) => Promise<number>
  quarantine: (raw: unknown, reason: string) => Promise<void>
}

export const TRADES_SOURCE = 'trades'

export async function syncTrades(deps: SyncTradesDeps): Promise<SyncResult> {
  const watermark = await deps.readWatermark()
  const dates = datesToScan(deps.now)
  const after = watermark.toISOString()

  let scanned = 0
  let inserted = 0
  let quarantined = 0

  for (const date of dates) {
    const items = await deps.fetchTrades(date, after)
    scanned += items.length

    const rows: TradeRow[] = []
    for (const item of items) {
      const mapped = mapTrade(item)
      if (mapped.ok) rows.push(mapped.value)
      else {
        await deps.quarantine(item, mapped.reason)
        quarantined += 1
      }
    }
    if (rows.length > 0) inserted += await deps.insertTrades(rows)
  }

  const next = computeNextWatermark(deps.now)
  await deps.advanceWatermark(TRADES_SOURCE, next)

  return { scanned, inserted, quarantined, watermark: next.toISOString() }
}
```

- [ ] **Step 9: Run both trade tests to verify they pass**

```bash
cd web && npx vitest run test/mapTrade.test.ts test/syncTrades.test.ts
```
Expected: PASS, 10 tests total.

- [ ] **Step 10: Commit**

```bash
git add web/src/lib/sync web/test/mapTrade.test.ts web/test/syncTrades.test.ts
git commit -m "feat(sync): trade mapping and sync with DO UPDATE for overwritten rows"
```

---

## Task 8: DynamoDB fetchers and the cron route

**Files:**
- Create: `web/src/lib/ddb.ts`, `web/src/lib/ddbFetchers.ts`, `web/src/app/api/cron/sync/route.ts`, `web/vercel.json`

**Interfaces:**
- Consumes: `syncEvents` (Task 6), `syncTrades` (Task 7), `sql` (Task 1).
- Produces: `GET /api/cron/sync` returning `{ events: SyncResult, trades: SyncResult }`.

- [ ] **Step 1: Write `web/src/lib/ddb.ts`**

```typescript
import { DynamoDBClient } from '@aws-sdk/client-dynamodb'
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb'

/**
 * Read-only DynamoDB access.
 *
 * The IAM principal behind these credentials holds only Query, GetItem, and
 * DescribeTable. Never import a write command into this project (Global
 * Constraint #2) -- the permission boundary is the real guarantee, but the
 * absence of any write import is the thing a reviewer can actually see.
 */
const client = new DynamoDBClient({ region: process.env.AWS_REGION ?? 'us-east-1' })

export const ddb = DynamoDBDocumentClient.from(client, {
  marshallOptions: { removeUndefinedValues: true },
})

export const ANALYTICS_TABLE = process.env.ANALYTICS_TABLE ?? 'freeport-analytics-events'
export const TRADES_TABLE = process.env.TRADES_TABLE ?? 'freeport-trades-history'
```

- [ ] **Step 2: Write `web/src/lib/ddbFetchers.ts`**

```typescript
import { QueryCommand } from '@aws-sdk/lib-dynamodb'
import { ddb, ANALYTICS_TABLE, TRADES_TABLE } from './ddb'

/**
 * Fetch events for one UTC date partition with sk strictly greater than `afterSk`.
 *
 * The sort key is `{timestamp}#{wallet8}#{rand8}`
 * (Swap_Server/src/services/analytics.ts:73), so it sorts lexicographically by
 * time and a bare ISO timestamp is a valid exclusive lower bound. This is a
 * key-condition range query, not a scan.
 */
export async function fetchEvents(date: string, afterSk: string): Promise<unknown[]> {
  const out: unknown[] = []
  let last: Record<string, unknown> | undefined
  do {
    const resp = await ddb.send(new QueryCommand({
      TableName: ANALYTICS_TABLE,
      KeyConditionExpression: '#d = :d AND sk > :sk',
      ExpressionAttributeNames: { '#d': 'date' },
      ExpressionAttributeValues: { ':d': date, ':sk': afterSk },
      ExclusiveStartKey: last,
    }))
    out.push(...(resp.Items ?? []))
    last = resp.LastEvaluatedKey
  } while (last)
  return out
}

/** Fetch trades for one trade_date via the trade_date-timestamp-index GSI (see app.py:649-664). */
export async function fetchTrades(tradeDate: string, afterTimestamp: string): Promise<unknown[]> {
  const out: unknown[] = []
  let last: Record<string, unknown> | undefined
  do {
    const resp = await ddb.send(new QueryCommand({
      TableName: TRADES_TABLE,
      IndexName: 'trade_date-timestamp-index',
      KeyConditionExpression: 'trade_date = :d AND #ts > :ts',
      ExpressionAttributeNames: { '#ts': 'timestamp' },
      ExpressionAttributeValues: { ':d': tradeDate, ':ts': afterTimestamp },
      ExclusiveStartKey: last,
    }))
    out.push(...(resp.Items ?? []))
    last = resp.LastEvaluatedKey
  } while (last)
  return out
}
```

- [ ] **Step 3: Write `web/src/app/api/cron/sync/route.ts`**

```typescript
import { NextResponse } from 'next/server'
import { sql } from '@/lib/db'
import { fetchEvents, fetchTrades } from '@/lib/ddbFetchers'
import { syncEvents, EVENTS_SOURCE } from '@/lib/sync/syncEvents'
import { syncTrades, TRADES_SOURCE } from '@/lib/sync/syncTrades'
import { readWatermark, advanceWatermark } from '@/lib/sync/watermark'
import { ensurePartitions } from '@/lib/sync/partitions'
import { insertEvents } from '@/lib/sync/insertEvents'
import { insertTrades } from '@/lib/sync/insertTrades'
import { quarantineRow } from '@/lib/sync/quarantine'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

// Only used the very first time a source runs, before sync_state has a row.
const COLD_START = new Date('2026-07-01T00:00:00.000Z')

export async function GET(request: Request) {
  const secret = process.env.CRON_SECRET
  const auth = request.headers.get('authorization')
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 })
  }

  const now = new Date()

  const events = await syncEvents({
    now,
    readWatermark: () => readWatermark(sql, EVENTS_SOURCE, COLD_START),
    advanceWatermark: (source, ts) => advanceWatermark(sql, source, ts),
    ensurePartitions: dates => ensurePartitions(sql, dates),
    fetchEvents,
    insertEvents: rows => insertEvents(sql, rows),
    quarantine: (raw, reason) => quarantineRow(sql, EVENTS_SOURCE, raw, reason),
  })

  const trades = await syncTrades({
    now,
    readWatermark: () => readWatermark(sql, TRADES_SOURCE, COLD_START),
    advanceWatermark: (source, ts) => advanceWatermark(sql, source, ts),
    fetchTrades,
    insertTrades: rows => insertTrades(sql, rows),
    quarantine: (raw, reason) => quarantineRow(sql, TRADES_SOURCE, raw, reason),
  })

  return NextResponse.json({ events, trades })
}
```

- [ ] **Step 4: Write `web/vercel.json`**

```json
{
  "crons": [
    { "path": "/api/cron/sync", "schedule": "* * * * *" }
  ]
}
```

- [ ] **Step 5: Typecheck**

```bash
cd web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 6: Verify the whole suite still passes**

```bash
cd web && npm test
```
Expected: PASS, all tests from Tasks 2–7.

- [ ] **Step 7: Commit**

```bash
git add web
git commit -m "feat(sync): DynamoDB range-query fetchers and 60s cron route"
```

---

## Task 9: Backfill

**Files:**
- Create: `web/scripts/backfill.ts`

**Interfaces:**
- Consumes: `mapEvent`, `mapTrade`, `insertEvents`, `insertTrades`, `ensurePartitions`, `fetchEvents`, `fetchTrades`.
- Produces: a resumable historical load. Prerequisite for Task 10.

- [ ] **Step 1: Write `web/scripts/backfill.ts`**

```typescript
/**
 * Historical backfill. Resumable: progress is recorded per date in sync_state
 * under `backfill:<source>`, so a crash resumes at the next unfinished date
 * rather than restarting the whole 1.5M-row load.
 *
 * Usage:
 *   tsx scripts/backfill.ts events 2025-01-01 2026-07-20
 *   tsx scripts/backfill.ts trades 2025-01-01 2026-07-20
 */
import { neon } from '@neondatabase/serverless'
import { fetchEvents, fetchTrades } from '../src/lib/ddbFetchers'
import { mapEvent } from '../src/lib/sync/mapEvent'
import { mapTrade } from '../src/lib/sync/mapTrade'
import { insertEvents } from '../src/lib/sync/insertEvents'
import { insertTrades } from '../src/lib/sync/insertTrades'
import { ensurePartitions } from '../src/lib/sync/partitions'
import { quarantineRow } from '../src/lib/sync/quarantine'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')
const sql = neon(url)

function eachDate(start: string, end: string): string[] {
  const out: string[] = []
  const cur = new Date(`${start}T00:00:00Z`)
  const stop = new Date(`${end}T00:00:00Z`)
  while (cur <= stop) {
    out.push(cur.toISOString().slice(0, 10))
    cur.setUTCDate(cur.getUTCDate() + 1)
  }
  return out
}

async function markDone(source: string, date: string) {
  await sql`
    INSERT INTO sync_state (source, watermark_ts, updated_at)
    VALUES (${`backfill:${source}`}, ${`${date}T00:00:00.000Z`}, now())
    ON CONFLICT (source) DO UPDATE
      SET watermark_ts = GREATEST(sync_state.watermark_ts, EXCLUDED.watermark_ts),
          updated_at = now()
  `
}

async function lastDone(source: string): Promise<string | null> {
  const rows = (await sql`
    SELECT watermark_ts FROM sync_state WHERE source = ${`backfill:${source}`}
  `) as Array<{ watermark_ts: string | Date }>
  if (rows.length === 0) return null
  return new Date(rows[0]!.watermark_ts).toISOString().slice(0, 10)
}

async function main() {
  const [source, start, end] = process.argv.slice(2)
  if (source !== 'events' && source !== 'trades') {
    throw new Error('usage: backfill.ts <events|trades> <start YYYY-MM-DD> <end YYYY-MM-DD>')
  }
  if (!start || !end) throw new Error('start and end dates are required')

  const resume = await lastDone(source)
  const dates = eachDate(start, end).filter(d => !resume || d > resume)
  if (resume) console.log(`resuming after ${resume}; ${dates.length} dates remain`)

  let totalIn = 0
  let totalQ = 0

  for (const date of dates) {
    if (source === 'events') {
      await ensurePartitions(sql, [date])
      const items = await fetchEvents(date, '')
      const rows = []
      for (const item of items) {
        const m = mapEvent(item)
        if (m.ok) rows.push(m.value)
        else { await quarantineRow(sql, 'events', item, m.reason); totalQ++ }
      }
      if (rows.length) totalIn += await insertEvents(sql, rows)
      console.log(`${date}  scanned=${items.length}  inserted=${rows.length}`)
    } else {
      const items = await fetchTrades(date, '')
      const rows = []
      for (const item of items) {
        const m = mapTrade(item)
        if (m.ok) rows.push(m.value)
        else { await quarantineRow(sql, 'trades', item, m.reason); totalQ++ }
      }
      if (rows.length) totalIn += await insertTrades(sql, rows)
      console.log(`${date}  scanned=${items.length}  inserted=${rows.length}`)
    }
    await markDone(source, date)
  }

  console.log(`backfill complete: inserted=${totalIn} quarantined=${totalQ}`)
}

main().catch(e => { console.error(e); process.exit(1) })
```

- [ ] **Step 2: Dry-run a single day against the test branch**

```bash
cd web && DATABASE_URL="$TEST_DATABASE_URL" npx tsx scripts/backfill.ts events 2026-07-19 2026-07-19
```
Expected: one `2026-07-19 scanned=N inserted=N` line, then `backfill complete`.

- [ ] **Step 3: Prove idempotency**

```bash
cd web && DATABASE_URL="$TEST_DATABASE_URL" psql "$TEST_DATABASE_URL" -c "SELECT count(*) FROM events"
# re-run the same day, bypassing the resume cursor
psql "$TEST_DATABASE_URL" -c "DELETE FROM sync_state WHERE source = 'backfill:events'"
DATABASE_URL="$TEST_DATABASE_URL" npx tsx scripts/backfill.ts events 2026-07-19 2026-07-19
psql "$TEST_DATABASE_URL" -c "SELECT count(*) FROM events"
```
Expected: **identical counts.** A changed count means `ON CONFLICT DO NOTHING` is not holding — stop and fix before proceeding.

- [ ] **Step 4: Commit**

```bash
git add web/scripts/backfill.ts
git commit -m "feat(backfill): resumable historical load reusing sync mapping"
```

---

## Task 10: Verification script and production backfill

**Files:**
- Create: `web/scripts/verify.ts`

**Interfaces:**
- Consumes: everything prior. Produces the evidence that Phase 1 is correct.

- [ ] **Step 1: Write `web/scripts/verify.ts`**

```typescript
/**
 * Verify the mirror against DynamoDB.
 *
 * For each date: compare the Postgres row count against a DynamoDB COUNT query,
 * then report quarantine and default-partition health. A non-empty default
 * partition means a row landed outside every declared monthly range and is an
 * alert condition.
 *
 * Usage: tsx scripts/verify.ts 2026-07-14 2026-07-20
 */
import { neon } from '@neondatabase/serverless'
import { QueryCommand } from '@aws-sdk/lib-dynamodb'
import { ddb, ANALYTICS_TABLE } from '../src/lib/ddb'

const url = process.env.DATABASE_URL
if (!url) throw new Error('DATABASE_URL is not set')
const sql = neon(url)

async function ddbCount(date: string): Promise<number> {
  let total = 0
  let last: Record<string, unknown> | undefined
  do {
    const resp = await ddb.send(new QueryCommand({
      TableName: ANALYTICS_TABLE,
      KeyConditionExpression: '#d = :d',
      ExpressionAttributeNames: { '#d': 'date' },
      ExpressionAttributeValues: { ':d': date },
      Select: 'COUNT',
      ExclusiveStartKey: last,
    }))
    total += resp.Count ?? 0
    last = resp.LastEvaluatedKey
  } while (last)
  return total
}

async function main() {
  const [start, end] = process.argv.slice(2)
  if (!start || !end) throw new Error('usage: verify.ts <start> <end>')

  const cur = new Date(`${start}T00:00:00Z`)
  const stop = new Date(`${end}T00:00:00Z`)
  let failures = 0

  while (cur <= stop) {
    const date = cur.toISOString().slice(0, 10)
    const [pg] = (await sql`
      SELECT count(*)::int AS n FROM events WHERE date = ${date}
    `) as Array<{ n: number }>
    const ddbN = await ddbCount(date)
    const pgN = pg?.n ?? 0
    const ok = pgN === ddbN
    if (!ok) failures++
    console.log(`${date}  ddb=${ddbN}  pg=${pgN}  ${ok ? 'OK' : 'MISMATCH'}`)
    cur.setUTCDate(cur.getUTCDate() + 1)
  }

  const [q] = (await sql`SELECT count(*)::int AS n FROM quarantine`) as Array<{ n: number }>
  const [d] = (await sql`SELECT count(*)::int AS n FROM events_default`) as Array<{ n: number }>
  console.log(`quarantine rows: ${q?.n ?? 0}`)
  console.log(`default-partition rows: ${d?.n ?? 0} (must be 0)`)

  if (failures > 0 || (d?.n ?? 0) > 0) {
    console.error(`FAILED: ${failures} date mismatches`)
    process.exit(1)
  }
  console.log('verification passed')
}

main().catch(e => { console.error(e); process.exit(1) })
```

- [ ] **Step 2: Verify against the test branch**

```bash
cd web && DATABASE_URL="$TEST_DATABASE_URL" npx tsx scripts/verify.ts 2026-07-19 2026-07-19
```
Expected: `2026-07-19 ddb=N pg=N OK`, `default-partition rows: 0`, `verification passed`.

- [ ] **Step 3: Migrate production and run the full backfill**

The first date is chosen by finding the earliest date present in the table; `2025-01-01` is a
safe lower bound that costs only empty queries for dates before data existed.

```bash
cd web
npm run migrate                                    # against DATABASE_URL (main branch)
npx tsx scripts/backfill.ts events 2025-01-01 2026-07-20
npx tsx scripts/backfill.ts trades 2025-01-01 2026-07-20
```
Expected: per-date progress lines; roughly 1.5M events inserted. If it dies, re-run the same
command — it resumes from the recorded cursor.

- [ ] **Step 4: Verify production across a two-week window**

```bash
cd web && npx tsx scripts/verify.ts 2026-07-06 2026-07-20
```
Expected: every date `OK`, `default-partition rows: 0`, `verification passed`.

Investigate any `MISMATCH` before continuing — a mismatch here is exactly the class of silent
data failure this phase exists to rule out.

- [ ] **Step 5: Deploy and confirm the cron runs**

Set every var from `.env.example` in the Vercel project, deploy, then:

```bash
curl -s -H "Authorization: Bearer $CRON_SECRET" \
  https://<project>.vercel.app/api/cron/sync | jq
```
Expected: `{"events":{"scanned":N,"inserted":N,"quarantined":0,"watermark":"..."},"trades":{...}}`.

Wait five minutes, then confirm the watermark is advancing on its own:

```bash
psql "$DATABASE_URL" -c "SELECT source, watermark_ts, now() - watermark_ts AS age FROM sync_state"
```
Expected: `age` for `events` and `trades` stays near 10 minutes and does not grow — a growing
age means the cron is not firing.

- [ ] **Step 6: Commit**

```bash
git add web/scripts/verify.ts
git commit -m "feat(verify): row-count and partition-health verification against DynamoDB"
```

---

## Phase 1 Definition of Done

- [ ] `npm test` green — all tests from Tasks 2–7.
- [ ] `npx tsc --noEmit` clean.
- [ ] `scripts/verify.ts` passes over a 14-day window with zero mismatches.
- [ ] `quarantine` is empty, or every row is individually explained.
- [ ] `events_default` has zero rows.
- [ ] Watermark age holds near 10 minutes for 30+ minutes without growing.
- [ ] The IAM write-denial probe from Task 0 Step 3 still returns `AccessDeniedException`.
- [ ] `git diff --stat main -- app.py hl_volume.py test_hl_volume.py requirements.txt` is empty —
      the Streamlit app is untouched and still serving.

---

## Self-Review Notes

**Spec coverage.** Phase 1 sections of the spec map to tasks as follows: schema → Task 1;
cron-pull sync and watermark → Tasks 5, 6, 8; late-arrival handling → Task 5; backfill →
Task 9; quarantine (risk #3) → Tasks 2, 6, 7; partition management (risk #4) → Task 4;
trades `DO UPDATE` (risk #7) → Task 7; timezone rule (risk #1) → Task 3 plus the schema
comment in Task 1; read-only IAM → Task 0; verification → Task 10.

**Deliberately deferred to Phase 2**, and therefore absent here: `daily_rollups` population
(risk #5 covers its recompute-never-increment rule, but its contents are defined by the
metrics the dashboard renders — inventing them now would be guessing), the nightly 2-day
reconciliation pass (it is the backfill script run over a 2-day window, best scheduled once
the cron's real-world behavior is observed), and the 90-day retention partition drop (nothing
is 90 days old in the mirror yet).

**Not covered by unit tests, by design:** `ddbFetchers.ts` and `ddb.ts` are thin network
wrappers. They are exercised by Task 9's dry run and Task 10's verification against real data,
which is stronger evidence than a mock would provide.
