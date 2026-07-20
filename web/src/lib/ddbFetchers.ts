import { QueryCommand } from '@aws-sdk/lib-dynamodb'
import { ddb, ANALYTICS_TABLE, TRADES_TABLE } from './ddb'

/**
 * Fetch events for one UTC date partition with sk strictly greater than `afterSk`.
 *
 * The sort key is `{timestamp}#{wallet8}#{rand8}`
 * (Swap_Server/src/services/analytics.ts:73), so it sorts lexicographically by
 * time and a bare ISO timestamp is a valid exclusive lower bound. This is a
 * key-condition range query, not a scan. An empty string is a valid lower
 * bound too (every real sk sorts after '') -- reconcile.ts's nightly full
 * re-read passes '' deliberately, the same way scripts/backfill.ts does.
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
