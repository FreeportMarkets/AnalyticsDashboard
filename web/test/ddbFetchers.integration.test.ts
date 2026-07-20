import { describe, it, expect } from 'vitest'
import { DescribeTableCommand } from '@aws-sdk/client-dynamodb'
import { ddb, ANALYTICS_TABLE } from '@/lib/ddb'
import { fetchEvents, FULL_READ_FLOOR } from '@/lib/ddbFetchers'

/**
 * REGRESSION for the bug that made the nightly reconciliation cron
 * non-functional: `reconcileEvents.ts`/`reconcileTrades.ts` passed `''` as
 * the exclusive lower bound for a "full read" query, and ddbFetchers.ts's
 * own doc comment claimed that was valid. It is not -- DynamoDB
 * unconditionally rejects an empty string for a key attribute in a
 * KeyConditionExpression, so `/api/cron/reconcile` threw on every run and
 * synced nothing.
 *
 * Every other test touching `fetchEvents`/`fetchTrades` injects a MOCKED
 * fetcher (see test/reconcile.test.ts's `eventDeps`/`tradeDeps`), so nothing
 * in the rest of the suite ever exercises the real DynamoDB call this bug
 * lived in. That is exactly why it shipped and stayed broken. This file
 * runs the real query against the real table so this class of bug can't
 * silently recur.
 *
 * This is a READ-ONLY query (`QueryCommand`/`DescribeTableCommand` only --
 * see src/lib/ddb.ts's doc comment: the IAM principal behind these
 * credentials holds only Query, GetItem, and DescribeTable). It writes
 * nothing to DynamoDB or Postgres.
 *
 * Follows test/integration.test.ts's skip-when-unconfigured pattern, adapted
 * for AWS: rather than checking a static env var (DynamoDB credentials come
 * from the ambient AWS SSO session, not a static key in `.env.local`), a
 * cheap `DescribeTable` call at module load time proves credentials both
 * exist AND are live (an expired SSO session fails this the same as no
 * credentials at all), and the whole suite skips gracefully rather than
 * failing when it can't reach AWS -- e.g. in CI, or a local run with an
 * expired `aws sso login` session.
 */
let AWS_AVAILABLE = false
try {
  await ddb.send(new DescribeTableCommand({ TableName: ANALYTICS_TABLE }))
  AWS_AVAILABLE = true
} catch {
  AWS_AVAILABLE = false
}

describe.skipIf(!AWS_AVAILABLE)('fetchEvents (real DynamoDB table)', () => {
  it(
    'REGRESSION: FULL_READ_FLOOR is a valid exclusive lower bound -- a full read of a ' +
      'real date partition returns rows instead of throwing ValidationException',
    async () => {
      // 2026-07-19 is a real date with real data, not a fixture -- measured
      // against the live `freeport-analytics-events` table 2026-07-20 at
      // 5213 events. Assert non-empty rather than the exact count so this
      // test doesn't need updating if more history is ever backfilled into
      // this date; the point is proving the query succeeds and pages
      // through real data, not pinning a row count.
      const items = await fetchEvents('2026-07-19', FULL_READ_FLOOR)
      expect(items.length).toBeGreaterThan(0)

      // Prove these are real event rows, not an empty/degenerate response.
      const sample = items[0] as Record<string, unknown>
      expect(typeof sample.sk).toBe('string')
      expect(typeof sample.event).toBe('string')
    }
  )

  it(
    "REGRESSION: '' throws ValidationException -- this is WHY FULL_READ_FLOOR exists; " +
      'a future refactor that quietly reintroduces `afterSk = \'\'` must fail this test',
    async () => {
      let caught: unknown
      try {
        await fetchEvents('2026-07-19', '')
      } catch (err) {
        caught = err
      }
      expect(caught).toBeDefined()
      expect((caught as { name?: string }).name).toBe('ValidationException')
      expect((caught as { message?: string }).message).toMatch(
        /AttributeValue for a key attribute cannot contain an empty string value/
      )
    }
  )
})
