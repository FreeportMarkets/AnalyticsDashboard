-- Postgres mirror of Privy wallet identities. `fetchPrivyUsers()` (a live,
-- paginated Privy REST call indexing ~5,830 wallets) measured 14.4s cold and
-- was being called INSIDE the render path of /trades, /users, and /referrals
-- -- exactly the three pages reported as "slow as hell". On Vercel, every
-- cold serverless instance re-runs it, so clicking around repeatedly hits
-- fresh instances and stalls for up to a minute. The module-scope TTL cache
-- in src/lib/privy.ts only helps a WARM instance -- it does nothing for the
-- cold-start case, which is the common case on Vercel.
--
-- Every other data source in this project (events, trades) is already a
-- Postgres mirror of an external source of truth (DynamoDB). This table
-- extends that pattern to Privy: refreshed once nightly by the existing
-- /api/cron/reconcile cron (see src/lib/privyIdentities.ts), then read by
-- page render with a single indexed query scoped to just the wallets/DIDs
-- on that page, instead of a 14s live fetch of the entire user base.
--
-- `wallet_address` is the PRIMARY KEY and is always stored LOWERCASED (same
-- convention as PrivyWalletMap in src/lib/privy.ts) so lookups are a plain
-- equality/ANY() match with no case-folding at query time.
--
-- `did` has its own index (not unique -- a Privy user can own multiple
-- wallets, so the same did legitimately appears on multiple rows) so
-- /referrals' DID-keyed top-referrers lookup can also hit an index instead
-- of a sequential scan.
CREATE TABLE IF NOT EXISTS privy_identities (
  wallet_address text PRIMARY KEY,
  did            text,
  label          text,
  login_type     text,
  contact        text,
  synced_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS privy_identities_did_idx ON privy_identities (did);
