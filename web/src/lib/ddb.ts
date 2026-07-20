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
