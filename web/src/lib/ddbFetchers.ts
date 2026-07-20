import { QueryCommand } from '@aws-sdk/lib-dynamodb'
import { ddb, ANALYTICS_TABLE, TRADES_TABLE } from './ddb'

/**
 * Exclusive lower bound used for a "full read of this date/partition" query
 * against either table (`sk` on `freeport-analytics-events`, `timestamp` on
 * `freeport-trades-history` via `trade_date-timestamp-index`).
 *
 * `''` is NOT valid here, despite an earlier version of this file's doc
 * comment claiming otherwise. DynamoDB unconditionally rejects an empty
 * string for any KEY attribute in a KeyConditionExpression --
 * `ValidationException: The AttributeValue for a key attribute cannot
 * contain an empty string value` -- regardless of the comparison operator.
 * Verified against the real tables 2026-07-20 with `sk > ''`.
 *
 * Both sort keys are ISO-8601-prefixed strings (`sk` is
 * `{timestamp}#{wallet8}#{rand8}`, Swap_Server/src/services/analytics.ts:73;
 * the trades GSI sorts on a bare `timestamp`), so an ISO timestamp that
 * sorts before every real value is a valid substitute lower bound. Verified
 * against the real `freeport-analytics-events` table 2026-07-20:
 * `sk > '0000-01-01T00:00:00.000Z'` returns 3087 rows and paginates
 * correctly.
 *
 * This is the ONE definition -- `scripts/backfill.ts`, `reconcileEvents.ts`,
 * and `reconcileTrades.ts` all import it from here rather than keeping their
 * own copies, specifically so this class of bug (a full-read floor that
 * silently diverges into an invalid `''` in some call site) can't recur.
 */
export const FULL_READ_FLOOR = '0000-01-01T00:00:00.000Z'

/**
 * Fetch events for one UTC date partition with sk strictly greater than `afterSk`.
 *
 * The sort key is `{timestamp}#{wallet8}#{rand8}`
 * (Swap_Server/src/services/analytics.ts:73), so it sorts lexicographically by
 * time and a bare ISO timestamp is a valid exclusive lower bound. This is a
 * key-condition range query, not a scan. `afterSk` must never be `''` -- see
 * `FULL_READ_FLOOR` above for why, and pass that constant for a full read.
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
