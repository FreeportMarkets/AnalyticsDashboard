# Analytics v2 — Phase 2A: Dashboard Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deployed, authenticated, non-indexed Next.js dashboard shell whose metric layer is provably timezone-correct, with a parity harness ready to compare against Streamlit.

**Architecture:** Next.js App Router on Vercel, root directory `web/`. Auth.js sets an httpOnly session cookie so `middleware.ts` enforces default-deny across every route in one place. Server Components query the Neon mirror built in Phase 1 directly — no API layer, no client-side credentials. All metric time-bucketing derives from `ts AT TIME ZONE 'America/New_York'`; the `date` column is storage only.

**Tech Stack:** Next.js 15, React 19, TypeScript, Auth.js v5 (`next-auth@5`) with the Google provider, Tailwind CSS v4, `@neondatabase/serverless`, Vitest.

**Spec:** `docs/superpowers/specs/2026-07-20-analytics-v2-design.md`
**Builds on:** `docs/superpowers/plans/2026-07-20-phase1-pipeline.md` (complete — 1.5M rows mirrored and verified)

---

## Global Constraints

Every task's requirements implicitly include this section.

- **The `date` column is storage and partitioning only.** It must NEVER appear in a `GROUP BY` or in a metric's date-range filter. All metric time bucketing derives from `ts AT TIME ZONE 'America/New_York'`. The DynamoDB partition key is UTC-derived while every metric buckets in New York; grouping on it shifts daily figures by up to 5 hours in a way that looks entirely plausible. This is the project's highest-severity correctness risk and it is now **enforced by Task 6's static scan**, not by discipline.
- **Timezone constant:** `America/New_York`, imported from `@/lib/time` as `NY_TZ`. Never re-declare it.
- **Read-only against AWS.** No `PutItem`, `UpdateItem`, `DeleteItem`, or `BatchWriteItem` anywhere. The IAM principal holds only `dynamodb:Query`, `GetItem`, `DescribeTable`.
- **Never silently drop a row.** Failures go to `quarantine` with the raw item and a reason.
- **The Python Streamlit app at the repo root is untouched and still serving production traffic.** `app.py`, `hl_volume.py`, `test_hl_volume.py`, `requirements.txt` — zero diff. All new code lives under `web/`.
- **Do-not-touch list** (CI-enforced, from the spec): `freeport-trading-backend/apps/api-gateway/src/services/trade-logger.ts`, `Swap_Server/src/services/dynamodb.ts`, `Swap_Server/src/services/analytics.ts`, `FreeApp/hooks/trading/useSwap.ts`, `FreeApp/hooks/trading/usePerpsHandlers.ts`, `Web_Terminal/app/src/lib/tradingApi.ts`. Phase 2A touches no repo other than `analytics-dashboard`.
- **Neon driver:** `@neondatabase/serverless` has NO `sql.query()`. Call it as a tagged template or `sql(text, params, opts)`. ONE statement per call. `{ fullResults: true }` is required to read `rowCount`.
- **Middleware must NOT block `/api/cron/*`.** Those routes authenticate with `CRON_SECRET` and are called by Vercel Cron, which cannot complete an OAuth flow. Blocking them silently kills the sync and reconciliation jobs.
- **Never commit `.env.local` or any credential.**
- Node 20+. npm. Integration tests read `TEST_DATABASE_URL` from `web/.env.local` and skip gracefully via `describe.skipIf(...)` when absent; they clean up via a unique marker in `finally` and never truncate a table.
- Vitest runs test files concurrently. Fixture years 2026, 2031, and 2088 are already used by existing tests — pick an unused year for any new real-DB fixture.

---

## File Structure

```
web/
  package.json                              + next-auth, tailwindcss v4, @tailwindcss/postcss
  postcss.config.mjs                        NEW
  src/app/globals.css                       NEW  tailwind entry
  src/app/layout.tsx                        NEW  root shell
  src/app/page.tsx                          NEW  overview (auth-gated)
  src/app/login/page.tsx                    NEW  sign-in
  src/app/robots.ts                         NEW  disallow all
  src/auth.ts                               NEW  Auth.js config + email allowlist
  src/app/api/auth/[...nextauth]/route.ts   NEW  Auth.js handlers
  src/middleware.ts                         NEW  default-deny gate
  src/lib/metrics/nyRange.ts                NEW  NY calendar range -> UTC bounds  (pure)
  src/lib/metrics/queries.ts                NEW  metric SQL, all NY-bucketed
  src/lib/metrics/staleness.ts              NEW  watermark age
  scripts/lint-no-date-grouping.ts          NEW  static scan, CI gate
  scripts/parity.ts                         NEW  harness skeleton
  vercel.json                               + X-Robots-Tag headers (keep both crons)
  test/nyRange.test.ts                      NEW
  test/metrics.integration.test.ts          NEW  THE divergent ts/date proof
  test/staleness.test.ts                    NEW
```

**Boundary rationale:** `nyRange.ts` is pure and unit-tested; `queries.ts` holds only SQL and is proven by a real-DB integration test, because a green unit suite concealed six defects in Phase 1. The static scan exists because a code-review convention is not an enforcement mechanism.

---

## Task 1: Tailwind + app shell

**Files:**
- Modify: `web/package.json`
- Create: `web/postcss.config.mjs`, `web/src/app/globals.css`, `web/src/app/layout.tsx`, `web/src/app/page.tsx`

**Interfaces:**
- Produces: a building Next.js app with a root layout. Every later task's UI renders inside it.

- [ ] **Step 1: Add dependencies**

```bash
cd web && npm install next-auth@^5.0.0-beta.25 && npm install -D tailwindcss@^4.0.0 @tailwindcss/postcss@^4.0.0
```

- [ ] **Step 2: Create `web/postcss.config.mjs`**

```javascript
export default { plugins: { '@tailwindcss/postcss': {} } }
```

- [ ] **Step 3: Create `web/src/app/globals.css`**

```css
@import "tailwindcss";

:root { color-scheme: dark; }

body {
  @apply bg-neutral-950 text-neutral-100 antialiased;
}
```

- [ ] **Step 4: Create `web/src/app/layout.tsx`**

```tsx
import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Freeport Analytics',
  description: 'Internal analytics dashboard',
  robots: { index: false, follow: false },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  )
}
```

- [ ] **Step 5: Create `web/src/app/page.tsx`**

```tsx
export default function OverviewPage() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Freeport Analytics</h1>
      <p className="mt-2 text-sm text-neutral-400">Phase 2A foundation.</p>
    </main>
  )
}
```

- [ ] **Step 6: Verify the build**

```bash
cd web && npx next build
```
Expected: build succeeds, `/` listed as a route. Fix any error before continuing — a broken build blocks every later task.

- [ ] **Step 7: Commit**

```bash
git add web && git commit -m "feat(web): Tailwind v4 and app shell"
```

---

## Task 2: Auth.js Google SSO with email allowlist

**Files:**
- Create: `web/src/auth.ts`, `web/src/app/api/auth/[...nextauth]/route.ts`, `web/src/middleware.ts`, `web/src/app/login/page.tsx`
- Modify: `web/.env.example`

**Interfaces:**
- Produces: `auth()` (session getter), `signIn`, `signOut`, exported from `@/auth`. Task 7 uses `auth()` to show the signed-in user.

**Why Auth.js and not Privy** (decided 2026-07-20, recorded in the spec): Privy stores its token in localStorage, which middleware cannot read, so gating would have to be repeated in every server component and API route. Auth.js sets an httpOnly cookie, giving default-deny at one choke point. Privy's user pool is also the product's end users — an internal dashboard that can mutate promo codes must not share an identity provider with them.

- [ ] **Step 1: Create `web/src/auth.ts`**

```typescript
import NextAuth from 'next-auth'
import Google from 'next-auth/providers/google'

/** Comma-separated allowlist, e.g. "a@x.com,b@x.com". Empty = deny everyone. */
function allowedEmails(): string[] {
  return (process.env.ALLOWED_EMAILS ?? '')
    .split(',')
    .map(e => e.trim().toLowerCase())
    .filter(Boolean)
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [Google],
  pages: { signIn: '/login' },
  callbacks: {
    /**
     * Deny by default. An unset or empty ALLOWED_EMAILS locks everyone out
     * rather than letting anyone with a Google account in — a misconfiguration
     * must fail closed.
     */
    signIn({ profile }) {
      const email = profile?.email?.toLowerCase()
      if (!email) return false
      return allowedEmails().includes(email)
    },
  },
})
```

- [ ] **Step 2: Create `web/src/app/api/auth/[...nextauth]/route.ts`**

```typescript
import { handlers } from '@/auth'

export const { GET, POST } = handlers
```

- [ ] **Step 3: Create `web/src/middleware.ts`**

```typescript
import { auth } from '@/auth'
import { NextResponse } from 'next/server'

/**
 * Default-deny gate for the entire app.
 *
 * `/api/cron/*` is deliberately excluded: those routes authenticate with
 * CRON_SECRET and are invoked by Vercel Cron, which cannot complete an OAuth
 * flow. Gating them here would silently kill the 60s sync and the nightly
 * reconciliation job.
 */
export default auth(req => {
  const { pathname } = req.nextUrl
  if (pathname.startsWith('/api/cron/')) return NextResponse.next()
  if (pathname.startsWith('/api/auth/')) return NextResponse.next()
  if (pathname === '/login') return NextResponse.next()

  if (!req.auth) {
    const url = new URL('/login', req.nextUrl.origin)
    return NextResponse.redirect(url)
  }
  return NextResponse.next()
})

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
```

- [ ] **Step 4: Create `web/src/app/login/page.tsx`**

```tsx
import { signIn } from '@/auth'

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-sm rounded-lg border border-neutral-800 p-8">
        <h1 className="text-lg font-semibold">Freeport Analytics</h1>
        <p className="mt-1 text-sm text-neutral-400">Internal access only.</p>
        <form
          className="mt-6"
          action={async () => {
            'use server'
            await signIn('google', { redirectTo: '/' })
          }}
        >
          <button
            type="submit"
            className="w-full rounded-md bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 hover:bg-white"
          >
            Sign in with Google
          </button>
        </form>
      </div>
    </main>
  )
}
```

- [ ] **Step 5: Document the new env vars in `web/.env.example`**

Append:
```bash
# Auth.js — generate with: openssl rand -base64 32
AUTH_SECRET=
# Google OAuth client (Google Cloud Console -> APIs & Services -> Credentials)
AUTH_GOOGLE_ID=
AUTH_GOOGLE_SECRET=
# Comma-separated allowlist. EMPTY OR UNSET DENIES EVERYONE (fail closed).
ALLOWED_EMAILS=
```

- [ ] **Step 6: Verify the build and that middleware compiles**

```bash
cd web && npx next build
```
Expected: build succeeds and `ƒ Middleware` appears in the route summary.

- [ ] **Step 7: Commit**

```bash
git add web && git commit -m "feat(auth): Google SSO with fail-closed email allowlist and default-deny middleware"
```

---

## Task 3: No-index

**Files:**
- Create: `web/src/app/robots.ts`
- Modify: `web/vercel.json`

**Interfaces:**
- Produces: `/robots.txt` disallowing everything, and an `X-Robots-Tag` response header on all routes.

Vercel sends `noindex` automatically on preview deployments but **not** on production `.vercel.app` URLs, so the explicit header is required rather than redundant.

- [ ] **Step 1: Create `web/src/app/robots.ts`**

```typescript
import type { MetadataRoute } from 'next'

export default function robots(): MetadataRoute.Robots {
  return { rules: [{ userAgent: '*', disallow: '/' }] }
}
```

- [ ] **Step 2: Add headers to `web/vercel.json`**

**Keep both existing cron entries.** The file must end up as:

```json
{
  "crons": [
    { "path": "/api/cron/reconcile", "schedule": "0 8 * * *" },
    { "path": "/api/cron/sync", "schedule": "* * * * *" }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Robots-Tag", "value": "noindex, nofollow, noarchive" }
      ]
    }
  ]
}
```

- [ ] **Step 3: Verify**

```bash
cd web && npx next build && cat vercel.json
```
Expected: build succeeds; both crons still present.

- [ ] **Step 4: Commit**

```bash
git add web && git commit -m "feat(web): no-index headers and robots.txt"
```

---

## Task 4: NY date-range helper (pure)

**Files:**
- Create: `web/src/lib/metrics/nyRange.ts`
- Test: `web/test/nyRange.test.ts`

**Interfaces:**
- Consumes: `NY_TZ` from `@/lib/time`.
- Produces: `nyRangeToUtc(startDate: string, endDate: string): { fromUtc: Date; toUtc: Date }`. Every metric query in Task 5 and in Phase 2B filters `ts` with these bounds.

A user picks New York calendar dates. Postgres stores `ts` as `timestamptz`. This converts an inclusive NY date range into half-open UTC instant bounds `[fromUtc, toUtc)`, so a metric filters on `ts` — never on the `date` column.

- [ ] **Step 1: Write the failing test**

`web/test/nyRange.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { nyRangeToUtc } from '@/lib/metrics/nyRange'

describe('nyRangeToUtc', () => {
  it('maps an EDT day to its UTC instants', () => {
    // 2026-07-20 00:00 EDT = 04:00Z; exclusive end is 2026-07-21 00:00 EDT = 04:00Z
    const r = nyRangeToUtc('2026-07-20', '2026-07-20')
    expect(r.fromUtc.toISOString()).toBe('2026-07-20T04:00:00.000Z')
    expect(r.toUtc.toISOString()).toBe('2026-07-21T04:00:00.000Z')
  })

  it('maps an EST day to its UTC instants', () => {
    const r = nyRangeToUtc('2026-01-15', '2026-01-15')
    expect(r.fromUtc.toISOString()).toBe('2026-01-15T05:00:00.000Z')
    expect(r.toUtc.toISOString()).toBe('2026-01-16T05:00:00.000Z')
  })

  it('spans a multi-day range', () => {
    const r = nyRangeToUtc('2026-07-01', '2026-07-07')
    expect(r.fromUtc.toISOString()).toBe('2026-07-01T04:00:00.000Z')
    expect(r.toUtc.toISOString()).toBe('2026-07-08T04:00:00.000Z')
  })

  it('rejects a malformed or non-calendar date', () => {
    expect(() => nyRangeToUtc('2026-02-30', '2026-03-01')).toThrow(/invalid date/i)
    expect(() => nyRangeToUtc('nope', '2026-03-01')).toThrow(/invalid date/i)
  })

  it('rejects an inverted range', () => {
    expect(() => nyRangeToUtc('2026-07-10', '2026-07-01')).toThrow(/range/i)
  })
})
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
cd web && npx vitest run test/nyRange.test.ts
```
Expected: FAIL — cannot resolve `@/lib/metrics/nyRange`.

- [ ] **Step 3: Implement `web/src/lib/metrics/nyRange.ts`**

```typescript
import { NY_TZ, isValidCalendarDate } from '@/lib/time'

/** UTC offset in minutes for a given instant in New York. */
function nyOffsetMinutes(at: Date): number {
  // 'shortOffset' yields e.g. "GMT-4". Parse the hour offset from it.
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: NY_TZ, timeZoneName: 'shortOffset',
  }).formatToParts(at)
  const tz = parts.find(p => p.type === 'timeZoneName')?.value ?? 'GMT+0'
  const m = /GMT([+-])(\d{1,2})(?::(\d{2}))?/.exec(tz)
  if (!m) throw new Error(`could not determine ${NY_TZ} offset from "${tz}"`)
  const sign = m[1] === '-' ? -1 : 1
  return sign * (Number(m[2]) * 60 + Number(m[3] ?? 0))
}

/** The UTC instant of local midnight in New York on the given calendar date. */
function nyMidnightUtc(date: string): Date {
  const naive = new Date(`${date}T00:00:00Z`).getTime()
  // Two passes: the offset itself depends on the instant (DST).
  let guess = new Date(naive - nyOffsetMinutes(new Date(naive)) * 60_000)
  guess = new Date(naive - nyOffsetMinutes(guess) * 60_000)
  return guess
}

/**
 * Convert an INCLUSIVE New York calendar date range into half-open UTC instant
 * bounds [fromUtc, toUtc).
 *
 * Metrics filter `ts` (timestamptz) with these bounds. They must NEVER filter or
 * group on the `date` column: that is the DynamoDB partition key, derived in UTC,
 * and using it shifts every daily figure by up to 5 hours in a way that still
 * looks plausible. See the project's Global Constraints.
 */
export function nyRangeToUtc(startDate: string, endDate: string): { fromUtc: Date; toUtc: Date } {
  if (!isValidCalendarDate(startDate)) throw new Error(`invalid date: ${startDate}`)
  if (!isValidCalendarDate(endDate)) throw new Error(`invalid date: ${endDate}`)
  if (startDate > endDate) throw new Error(`inverted range: ${startDate} > ${endDate}`)

  const fromUtc = nyMidnightUtc(startDate)
  const dayAfterEnd = new Date(`${endDate}T00:00:00Z`)
  dayAfterEnd.setUTCDate(dayAfterEnd.getUTCDate() + 1)
  const toUtc = nyMidnightUtc(dayAfterEnd.toISOString().slice(0, 10))
  return { fromUtc, toUtc }
}
```

- [ ] **Step 4: Run it, confirm it passes**

```bash
cd web && npx vitest run test/nyRange.test.ts
```
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add web && git commit -m "feat(metrics): NY calendar range to UTC instant bounds"
```

---

## Task 5: Metric queries + THE timezone proof

This is the task Phase 1 explicitly deferred here. Task 3 of Phase 1 built a guard that proves the UTC/NY divergence is real and that the *helpers* compute it correctly — it does **not** inspect any query. A metric query written with `GROUP BY date` would compile, run, and leave that suite green. This task closes it with a real-database proof.

**Files:**
- Create: `web/src/lib/metrics/queries.ts`
- Test: `web/test/metrics.integration.test.ts`

**Interfaces:**
- Consumes: `sql` from `@/lib/db`, `nyRangeToUtc` from `@/lib/metrics/nyRange`, `nyDateExpr` from `@/lib/time`.
- Produces: `dailyEventCounts(startDate, endDate): Promise<Array<{ day: string; count: number }>>` and `dailyActiveUsers(startDate, endDate): Promise<Array<{ day: string; users: number }>>`. Phase 2B builds every tab on this shape.

- [ ] **Step 1: Write the failing integration test**

`web/test/metrics.integration.test.ts`:
```typescript
import { describe, it, expect, afterAll } from 'vitest'
import { readFileSync } from 'node:fs'
import { neon } from '@neondatabase/serverless'
import { dailyEventCounts } from '@/lib/metrics/queries'

const env = Object.fromEntries(
  (() => { try { return readFileSync('.env.local', 'utf8').split('\n') } catch { return [] } })()
    .filter(l => l.includes('='))
    .map(l => { const i = l.indexOf('='); return [l.slice(0, i), l.slice(i + 1).replace(/^"|"$/g, '')] })
) as Record<string, string>

const url = env.TEST_DATABASE_URL
const MARK = 'tzproof_'
// Year 2077 is unused by other test files (2026, 2031, 2088 are taken).
const sql = url ? neon<boolean, boolean>(url) : null

describe.skipIf(!url)('metric timezone bucketing (real DB)', () => {
  afterAll(async () => {
    if (!sql) return
    await sql('DELETE FROM events WHERE wallet_address LIKE $1', [`${MARK}%`])
  })

  it('buckets by New York day, NOT by the UTC date partition key', async () => {
    if (!sql) return
    // 2077-03-16T02:30:00Z is 21:30 on 2077-03-15 in New York.
    // The DynamoDB-style `date` column says 2077-03-16. A correct metric says 03-15.
    await sql(
      `INSERT INTO events (date, sk, ts, event, wallet_address, session_id)
       VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING`,
      ['2077-03-16', `${MARK}a`, '2077-03-16T02:30:00Z', 'session_start', `${MARK}w1`, 's1']
    )
    // A second event squarely inside NY 03-16 as a control.
    await sql(
      `INSERT INTO events (date, sk, ts, event, wallet_address, session_id)
       VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT DO NOTHING`,
      ['2077-03-16', `${MARK}b`, '2077-03-16T18:00:00Z', 'session_start', `${MARK}w2`, 's2']
    )

    const rows = await dailyEventCounts('2077-03-15', '2077-03-16')
    const byDay = Object.fromEntries(rows.map(r => [r.day, r.count]))

    // THE ASSERTION. If a query ever groups by `date`, both rows land on
    // 2077-03-16 and this fails.
    expect(byDay['2077-03-15']).toBe(1)
    expect(byDay['2077-03-16']).toBe(1)
  })
})
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
cd web && npx vitest run test/metrics.integration.test.ts
```
Expected: FAIL — cannot resolve `@/lib/metrics/queries`.

- [ ] **Step 3: Implement `web/src/lib/metrics/queries.ts`**

```typescript
import { sql } from '@/lib/db'
import { nyDateExpr } from '@/lib/time'
import { nyRangeToUtc } from './nyRange'

/**
 * Every metric in this file follows the same two rules:
 *   1. Filter on `ts` using UTC instant bounds derived from the NY calendar range.
 *   2. Group with nyDateExpr('ts'), never with the `date` column.
 * The `date` column is the DynamoDB partition key, derived in UTC. Grouping on it
 * shifts daily figures by up to 5 hours and the result still looks plausible.
 * scripts/lint-no-date-grouping.ts enforces this mechanically.
 */

const SYSTEM_WALLETS = ['server', 'unknown', 'system', '']

export async function dailyEventCounts(
  startDate: string, endDate: string
): Promise<Array<{ day: string; count: number }>> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day, count(*)::int AS count
       FROM events
      WHERE ts >= $1 AND ts < $2
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ day: string | Date; count: number }>
  return rows.map(r => ({
    day: typeof r.day === 'string' ? r.day : r.day.toISOString().slice(0, 10),
    count: r.count,
  }))
}

export async function dailyActiveUsers(
  startDate: string, endDate: string
): Promise<Array<{ day: string; users: number }>> {
  const { fromUtc, toUtc } = nyRangeToUtc(startDate, endDate)
  const rows = (await sql(
    `SELECT ${nyDateExpr('ts')} AS day, count(DISTINCT wallet_address)::int AS users
       FROM events
      WHERE ts >= $1 AND ts < $2
        AND wallet_address IS NOT NULL
        AND wallet_address <> ALL($3::text[])
        AND (platform IS NULL OR platform <> 'server')
      GROUP BY 1
      ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString(), SYSTEM_WALLETS]
  )) as Array<{ day: string | Date; users: number }>
  return rows.map(r => ({
    day: typeof r.day === 'string' ? r.day : r.day.toISOString().slice(0, 10),
    users: r.users,
  }))
}
```

The `SYSTEM_WALLETS` exclusion and the `platform <> 'server'` filter are ported verbatim from `app.py:1155-1163`. Omitting them inflates user counts.

- [ ] **Step 4: Run it, confirm it passes**

```bash
cd web && npx vitest run test/metrics.integration.test.ts
```
Expected: PASS — one test, both day buckets correct.

- [ ] **Step 5: Prove the test is honest**

Temporarily change `dailyEventCounts` to `GROUP BY date` (and select `date AS day`). Re-run the test. It MUST fail with both events on `2077-03-16`. Restore the correct version and re-run to confirm PASS. Record both outputs in your report — a guard nobody has watched fail is not a guard.

- [ ] **Step 6: Commit**

```bash
git add web && git commit -m "feat(metrics): NY-bucketed daily metrics with real-DB timezone proof"
```

---

## Task 6: Static scan against `date` grouping

**Files:**
- Create: `web/scripts/lint-no-date-grouping.ts`
- Modify: `web/package.json` (add a `lint:dates` script)

**Interfaces:**
- Produces: a script that exits non-zero when a metric source groups or filters on the `date` column.

Task 5 proves one query is correct. This prevents the next one from being wrong. A review checklist is not an enforcement mechanism.

- [ ] **Step 1: Create `web/scripts/lint-no-date-grouping.ts`**

```typescript
/**
 * Fails if any metric source groups or range-filters on the `date` column.
 *
 * `date` is the DynamoDB partition key, derived in UTC. Every metric must bucket
 * with (ts AT TIME ZONE 'America/New_York'). Grouping on `date` shifts daily
 * figures by up to 5 hours and the wrong numbers look entirely plausible, which
 * is why this is enforced mechanically rather than by review.
 *
 * Scope: src/lib/metrics/** only. Sync, backfill, and verify legitimately use
 * `date` as a storage/partition key.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'

const ROOT = path.join(import.meta.dirname, '..', 'src', 'lib', 'metrics')

const BANNED: Array<{ re: RegExp; why: string }> = [
  { re: /GROUP\s+BY\s+[^\n]*\bdate\b/i, why: 'GROUP BY on the `date` column' },
  { re: /WHERE[^\n]*\bdate\b\s*(>=|<=|<|>|BETWEEN)/i, why: 'range filter on the `date` column' },
]

function walk(dir: string): string[] {
  const out: string[] = []
  for (const e of readdirSync(dir)) {
    const p = path.join(dir, e)
    if (statSync(p).isDirectory()) out.push(...walk(p))
    else if (p.endsWith('.ts')) out.push(p)
  }
  return out
}

let failures = 0
for (const file of walk(ROOT)) {
  const lines = readFileSync(file, 'utf8').split('\n')
  lines.forEach((line, i) => {
    if (line.trimStart().startsWith('//') || line.trimStart().startsWith('*')) return
    for (const { re, why } of BANNED) {
      if (re.test(line)) {
        console.error(`${path.relative(process.cwd(), file)}:${i + 1}: ${why}`)
        console.error(`  ${line.trim()}`)
        failures++
      }
    }
  })
}

if (failures > 0) {
  console.error(`\n${failures} violation(s). Metrics must bucket with (ts AT TIME ZONE 'America/New_York').`)
  process.exit(1)
}
console.log('lint:dates passed — no metric groups or range-filters on `date`')
```

- [ ] **Step 2: Add the script to `web/package.json`**

Add to `"scripts"`: `"lint:dates": "tsx scripts/lint-no-date-grouping.ts"`

- [ ] **Step 3: Verify it passes on the current tree**

```bash
cd web && npm run lint:dates
```
Expected: `lint:dates passed — no metric groups or range-filters on \`date\``

- [ ] **Step 4: Verify it actually catches a violation**

Temporarily add `// eslint-disable-next-line` free line `GROUP BY date` inside a template string in `queries.ts`, run `npm run lint:dates`, confirm it exits non-zero and names the file and line. Remove it and re-run to confirm it passes. Record both outputs — an unproven linter is decoration.

- [ ] **Step 5: Commit**

```bash
git add web && git commit -m "feat(lint): mechanical gate against grouping metrics on the date column"
```

---

## Task 7: Staleness badge and overview wiring

**Files:**
- Create: `web/src/lib/metrics/staleness.ts`
- Test: `web/test/staleness.test.ts`
- Modify: `web/src/app/page.tsx`

**Interfaces:**
- Consumes: `sql` from `@/lib/db`, `auth()` from `@/auth`, `dailyEventCounts`/`dailyActiveUsers` from `@/lib/metrics/queries`.
- Produces: `watermarkAge(): Promise<Array<{ source: string; ageSeconds: number }>>` and `formatAge(seconds: number): string`.

The single biggest complaint about the Streamlit dashboard is that it is silently stale. This surfaces staleness rather than hiding it: watermark age is displayed, and an age beyond 10 minutes is called out.

- [ ] **Step 1: Write the failing test**

`web/test/staleness.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { formatAge, isStale, STALE_THRESHOLD_SECONDS } from '@/lib/metrics/staleness'

describe('staleness', () => {
  it('treats 10 minutes as the threshold', () => {
    // The sync holds its watermark 10 minutes behind wall-clock by design,
    // so anything materially beyond that means the cron is not running.
    expect(STALE_THRESHOLD_SECONDS).toBe(15 * 60)
  })

  it('formats ages readably', () => {
    expect(formatAge(45)).toBe('45s')
    expect(formatAge(90)).toBe('1m 30s')
    expect(formatAge(3700)).toBe('1h 1m')
  })

  it('flags staleness past the threshold', () => {
    expect(isStale(600)).toBe(false)
    expect(isStale(901)).toBe(true)
  })
})
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
cd web && npx vitest run test/staleness.test.ts
```
Expected: FAIL — cannot resolve `@/lib/metrics/staleness`.

- [ ] **Step 3: Implement `web/src/lib/metrics/staleness.ts`**

```typescript
import { sql } from '@/lib/db'

/**
 * The incremental sync deliberately holds its watermark 10 minutes behind
 * wall-clock so late-arriving client events are not stepped over. 15 minutes
 * therefore means the cron itself has stopped, not that it is working normally.
 */
export const STALE_THRESHOLD_SECONDS = 15 * 60

export function isStale(ageSeconds: number): boolean {
  return ageSeconds > STALE_THRESHOLD_SECONDS
}

export function formatAge(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

export async function watermarkAge(): Promise<Array<{ source: string; ageSeconds: number }>> {
  const rows = (await sql(
    `SELECT source, EXTRACT(EPOCH FROM (now() - watermark_ts))::int AS age_seconds
       FROM sync_state
      WHERE source IN ('events','trades')
      ORDER BY source`
  )) as Array<{ source: string; age_seconds: number }>
  return rows.map(r => ({ source: r.source, ageSeconds: r.age_seconds }))
}
```

- [ ] **Step 4: Run it, confirm it passes**

```bash
cd web && npx vitest run test/staleness.test.ts
```
Expected: PASS, 3 tests.

- [ ] **Step 5: Wire the overview page**

Replace `web/src/app/page.tsx`:

```tsx
import { auth, signOut } from '@/auth'
import { dailyEventCounts, dailyActiveUsers } from '@/lib/metrics/queries'
import { watermarkAge, formatAge, isStale } from '@/lib/metrics/staleness'
import { NY_TZ } from '@/lib/time'

export const dynamic = 'force-dynamic'

function today(): string {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: NY_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date())
}

export default async function OverviewPage() {
  const session = await auth()
  const end = today()
  const startDate = new Date(`${end}T00:00:00Z`)
  startDate.setUTCDate(startDate.getUTCDate() - 6)
  const start = startDate.toISOString().slice(0, 10)

  const [counts, users, ages] = await Promise.all([
    dailyEventCounts(start, end),
    dailyActiveUsers(start, end),
    watermarkAge(),
  ])
  const stale = ages.filter(a => isStale(a.ageSeconds))

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Freeport Analytics</h1>
          <p className="mt-1 text-sm text-neutral-400">Last 7 days · {NY_TZ}</p>
        </div>
        <form action={async () => { 'use server'; await signOut({ redirectTo: '/login' }) }}>
          <button className="text-xs text-neutral-400 hover:text-neutral-200">
            {session?.user?.email} · sign out
          </button>
        </form>
      </header>

      {stale.length > 0 && (
        <div className="mt-6 rounded-md border border-amber-700/50 bg-amber-950/40 px-4 py-3 text-sm text-amber-200">
          Data is stale — {stale.map(s => `${s.source} ${formatAge(s.ageSeconds)} behind`).join(', ')}.
          The sync cron may not be running.
        </div>
      )}

      <section className="mt-8 grid gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-neutral-800 p-5">
          <div className="text-xs uppercase tracking-wide text-neutral-500">Events (7d)</div>
          <div className="mt-1 text-3xl font-semibold tabular-nums">
            {counts.reduce((a, r) => a + r.count, 0).toLocaleString()}
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 p-5">
          <div className="text-xs uppercase tracking-wide text-neutral-500">Peak DAU (7d)</div>
          <div className="mt-1 text-3xl font-semibold tabular-nums">
            {users.length ? Math.max(...users.map(u => u.users)).toLocaleString() : '0'}
          </div>
        </div>
      </section>

      <section className="mt-8 rounded-lg border border-neutral-800">
        <table className="w-full text-sm">
          <thead className="text-neutral-500">
            <tr className="border-b border-neutral-800">
              <th className="px-5 py-3 text-left font-medium">Day</th>
              <th className="px-5 py-3 text-right font-medium">Events</th>
              <th className="px-5 py-3 text-right font-medium">Active users</th>
            </tr>
          </thead>
          <tbody>
            {counts.map(row => (
              <tr key={row.day} className="border-b border-neutral-900 last:border-0">
                <td className="px-5 py-2.5 tabular-nums">{row.day}</td>
                <td className="px-5 py-2.5 text-right tabular-nums">{row.count.toLocaleString()}</td>
                <td className="px-5 py-2.5 text-right tabular-nums">
                  {(users.find(u => u.day === row.day)?.users ?? 0).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  )
}
```

- [ ] **Step 6: Verify the build and the full suite**

```bash
cd web && npx next build && npx vitest run && npx tsc --noEmit
```
Expected: build succeeds; all tests pass; typecheck clean.

- [ ] **Step 7: Commit**

```bash
git add web && git commit -m "feat(web): overview with honest staleness reporting"
```

---

## Task 8: Parity harness skeleton

**Files:**
- Create: `web/scripts/parity.ts`
- Modify: `web/package.json` (add a `parity` script)

**Interfaces:**
- Consumes: `dailyEventCounts`, `dailyActiveUsers`, `nyRangeToUtc`.
- Produces: a script that writes one `parity_runs` row per compared metric and exits non-zero on any failure. Phase 2B extends the metric list; the table already exists from Phase 1 migration `0001`.

Cutover to this dashboard is gated on 7 consecutive clean days of this harness. It exists now, with two metrics, so the gate is real infrastructure rather than an intention.

- [ ] **Step 1: Create `web/scripts/parity.ts`**

```typescript
/**
 * Parity harness. Computes a metric two ways and records the comparison.
 *
 * Phase 2A ships the mechanism with the metrics that exist today. Each added
 * metric gets a comparator here. Cutover from Streamlit is gated on 7
 * consecutive clean runs (see the spec's "Shadow run and cutover").
 *
 * Count metrics must match EXACTLY — any difference is a bug, not noise.
 * Currency metrics get a tolerance because the Postgres mirror uses `numeric`
 * while Streamlit casts DynamoDB Decimals to float; there the new value is the
 * more correct one. No currency metric exists yet.
 *
 * Usage: tsx scripts/parity.ts 2026-07-13 2026-07-20
 */
import { sql } from '../src/lib/db'
import { dailyEventCounts, dailyActiveUsers } from '../src/lib/metrics/queries'
import { nyRangeToUtc } from '../src/lib/metrics/nyRange'

type Comparison = {
  metric: string
  dims: Record<string, unknown>
  expected: number
  actual: number
  exact: boolean
}

/**
 * Reference implementation: the same figure computed independently of the
 * metric layer, so a bug in queries.ts cannot hide by being reused on both
 * sides. Deliberately spells out the NY bucketing inline.
 */
async function referenceEventCounts(start: string, end: string) {
  const { fromUtc, toUtc } = nyRangeToUtc(start, end)
  const rows = (await sql(
    `SELECT to_char(ts AT TIME ZONE 'America/New_York', 'YYYY-MM-DD') AS day,
            count(*)::int AS count
       FROM events WHERE ts >= $1 AND ts < $2 GROUP BY 1 ORDER BY 1`,
    [fromUtc.toISOString(), toUtc.toISOString()]
  )) as Array<{ day: string; count: number }>
  return rows
}

async function main() {
  const [start, end] = process.argv.slice(2)
  if (!start || !end) throw new Error('usage: parity.ts <start YYYY-MM-DD> <end YYYY-MM-DD>')

  const comparisons: Comparison[] = []

  const actual = await dailyEventCounts(start, end)
  const expected = await referenceEventCounts(start, end)
  const days = new Set([...actual.map(r => r.day), ...expected.map(r => r.day)])
  for (const day of [...days].sort()) {
    comparisons.push({
      metric: 'daily_event_count',
      dims: { day },
      expected: expected.find(r => r.day === day)?.count ?? 0,
      actual: actual.find(r => r.day === day)?.count ?? 0,
      exact: true,
    })
  }

  const dau = await dailyActiveUsers(start, end)
  for (const row of dau) {
    comparisons.push({
      metric: 'daily_active_users', dims: { day: row.day },
      expected: row.users, actual: row.users, exact: true,
    })
  }

  let failed = 0
  for (const c of comparisons) {
    const absDiff = Math.abs(c.actual - c.expected)
    const pctDiff = c.expected === 0 ? (c.actual === 0 ? 0 : 100) : (absDiff / c.expected) * 100
    const passed = c.exact ? absDiff === 0 : pctDiff < 0.01
    if (!passed) failed++
    await sql(
      `INSERT INTO parity_runs (metric, dims, streamlit_value, postgres_value, abs_diff, pct_diff, passed)
       VALUES ($1,$2::jsonb,$3,$4,$5,$6,$7)`,
      [c.metric, JSON.stringify(c.dims), c.expected, c.actual, absDiff, pctDiff, passed]
    )
    console.log(`${passed ? 'OK  ' : 'FAIL'} ${c.metric} ${JSON.stringify(c.dims)} expected=${c.expected} actual=${c.actual}`)
  }

  console.log(`\n${comparisons.length} comparison(s), ${failed} failure(s)`)
  if (failed > 0) process.exit(1)
}

main().catch(e => { console.error(e); process.exit(1) })
```

**Note for Phase 2B:** `daily_active_users` currently compares a value to itself, which is a placeholder, not a check. It must gain an independent reference implementation — or a direct comparison against Streamlit's own output — before the 7-day gate begins. Do not treat its passing as evidence.

- [ ] **Step 2: Add the script to `web/package.json`**

Add to `"scripts"`: `"parity": "tsx scripts/parity.ts"`

- [ ] **Step 3: Run it against the real mirror**

```bash
cd web && DATABASE_URL=$(grep '^DATABASE_URL=' .env.local | head -1 | cut -d= -f2- | tr -d '"') npm run parity -- 2026-07-13 2026-07-20
```
Expected: one `OK` line per day for `daily_event_count`, `0 failure(s)`, exit 0.

- [ ] **Step 4: Confirm rows landed**

```bash
cd web && node --input-type=module -e "
import {readFileSync} from 'node:fs'; import {neon} from '@neondatabase/serverless';
const env=Object.fromEntries(readFileSync('.env.local','utf8').split('\n').filter(l=>l.includes('=')).map(l=>{const i=l.indexOf('=');return [l.slice(0,i),l.slice(i+1).replace(/^\"|\"$/g,'')]}));
const sql=neon(env.DATABASE_URL);
console.log(await sql('SELECT metric, count(*)::int AS n, bool_and(passed) AS all_passed FROM parity_runs GROUP BY 1'));
"
```
Expected: a row per metric with `all_passed: true`.

- [ ] **Step 5: Commit**

```bash
git add web && git commit -m "feat(parity): harness recording comparisons to parity_runs"
```

---

## Phase 2A Definition of Done

- [ ] `npx next build` succeeds; `ƒ Middleware` present in the route summary.
- [ ] `npx vitest run` green; `npx tsc --noEmit` clean.
- [ ] `npm run lint:dates` passes, AND was observed failing on a deliberate violation.
- [ ] The timezone integration test was observed FAILING against a `GROUP BY date` implementation and passing against the correct one.
- [ ] Signed-out access to `/` redirects to `/login`; an email outside `ALLOWED_EMAILS` is refused.
- [ ] `/api/cron/sync` and `/api/cron/reconcile` still respond to a `CRON_SECRET` bearer request — middleware did not gate them.
- [ ] `curl -I` on a deployed URL shows `X-Robots-Tag: noindex, nofollow, noarchive`; `/robots.txt` disallows all.
- [ ] `npm run parity -- <7-day range>` exits 0 and writes `parity_runs` rows.
- [ ] `git diff main -- app.py hl_volume.py test_hl_volume.py requirements.txt` is empty.

---

## Human Prerequisite

Task 2 cannot be verified end-to-end without a Google OAuth client. In Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID → **Web application**:

- Authorized JavaScript origin: `https://<deployment>.vercel.app`
- Authorized redirect URI: `https://<deployment>.vercel.app/api/auth/callback/google`
- For local work also add `http://localhost:3000` and `http://localhost:3000/api/auth/callback/google`

Then set on the Vercel project: `AUTH_SECRET` (`openssl rand -base64 32`), `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `ALLOWED_EMAILS`.

Tasks 1 and 3–8 do not depend on this and can complete first.

---

## Self-Review Notes

**Spec coverage.** Auth → Task 2; no-index → Task 3; NY bucketing rule → Tasks 4, 5, 6; honest staleness → Task 7; parity harness → Task 8. The Phase 1 carry-forwards are covered: the query-level timezone proof (Task 5) and the static scan (Task 6) close the gap left by Phase 1's Task 3, whose guard covers only the helper functions.

**Deferred to Phase 2B, deliberately:** the eight tabs, the volume-math port and its parity assertions, promo-code admin with `audit_log`, the composable funnel builder, and the Screens & Engagement tab. Each depends on the metric layer this phase establishes. `daily_active_users` needs a genuine independent reference before the 7-day gate — flagged inline in Task 8.

**Also carried from the Phase 1 ledger, not yet scheduled:** hoisting a shared `SqlTag` into `types.ts` to remove the `sql as never` casts, and adding an allow-list to `nyDateExpr(tsColumn)` before any non-literal caller exists. Both are small; fold them into an early Phase 2B task.
