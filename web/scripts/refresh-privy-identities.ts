/**
 * One-shot / manual invocation of the privy_identities refresh
 * (src/lib/privyIdentities.ts's `refreshPrivyIdentities`) -- the same logic
 * /api/cron/reconcile now runs nightly. Exists for:
 *   1. the initial population run (the table starts empty after the
 *      0004_privy_identities.sql migration -- pages fail soft to truncated
 *      addresses until this has run once), and
 *   2. ad-hoc manual refreshes without waiting for the nightly cron.
 *
 * Usage: npx tsx scripts/refresh-privy-identities.ts
 * (DATABASE_URL / PRIVY_APP_ID / PRIVY_APP_SECRET must be in the environment.)
 */
import { sql } from '../src/lib/db'
import { refreshPrivyIdentities } from '../src/lib/privyIdentities'

async function main() {
  const start = Date.now()
  const result = await refreshPrivyIdentities(sql)
  const ms = Date.now() - start
  console.log(`refreshPrivyIdentities: fetchedWallets=${result.fetchedWallets} upserted=${result.upserted} (${ms}ms)`)

  const rows = (await sql`SELECT count(*)::int AS count FROM privy_identities`) as Array<{ count: number }>
  console.log(`privy_identities row count: ${rows[0]?.count ?? 0}`)
}

main().catch(e => {
  console.error(e)
  process.exit(1)
})
