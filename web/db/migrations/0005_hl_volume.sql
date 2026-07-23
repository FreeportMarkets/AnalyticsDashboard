-- HL builder-fee-authoritative perps volume tracking.
-- See docs/volume-tracking.md for the full rationale.
--
-- Every table here is keyed on a LOWERCASED evm address. HL returns mixed
-- case; our trades table stores mixed case; the backend user_wallets PK is
-- mixed case. Lowercasing at the boundary is the only way these three join
-- without silent misses, so it is a hard invariant enforced by the app
-- (store.ts lowercases every address before read or write) and mirrored by
-- the lower() unique index below.

-- The enumeration set: every Freeport wallet we know to scan.
CREATE TABLE IF NOT EXISTS tracked_wallets (
  evm_address   text        PRIMARY KEY,
  -- Where this address came from: 'analytics' (trades∪events) or 'registry'
  -- (backend user_wallets). Lets a later registry sync supersede the
  -- bootstrap set without losing provenance.
  source        text        NOT NULL,
  first_seen_at timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_tracked_wallets_lower ON tracked_wallets (lower(evm_address));

-- Per wallet, per NY calendar day: the authoritative volume grain.
-- notional_usd    = Σ(|sz|*px) over ALL perp fills (any dir; spot/settlement
--                   excluded) -- includes liquidations / TP-SL auto-closes,
--                   which carry no builder fee. See docs/volume-tracking.md.
-- builder_fee_usd = Σ(builderFee) -- the fee-bearing subset, for reconciliation.
CREATE TABLE IF NOT EXISTS wallet_volume_daily (
  evm_address     text        NOT NULL,
  day             date        NOT NULL,
  notional_usd    numeric     NOT NULL DEFAULT 0,
  builder_fee_usd numeric     NOT NULL DEFAULT 0,
  fill_count      integer     NOT NULL DEFAULT 0,
  updated_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (evm_address, day)
);
-- The dashboard's only read shape: sum across wallets, grouped/filtered by day.
CREATE INDEX IF NOT EXISTS idx_wallet_volume_daily_day ON wallet_volume_daily (day);

-- Per-wallet incremental watermark: the max fill `time` (epoch ms) ingested.
-- The next sync fetches only fills strictly after this.
CREATE TABLE IF NOT EXISTS hl_fill_sync_state (
  evm_address  text        PRIMARY KEY,
  last_fill_ms bigint      NOT NULL DEFAULT 0,
  updated_at   timestamptz NOT NULL DEFAULT now()
);

-- Reconciliation health log: bottom-up Σ(builder_fee) vs the collector's
-- claimed+pending fees, per run. ratio ~1.0 means enumeration is complete.
CREATE TABLE IF NOT EXISTS volume_reconciliation (
  id                bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_at            timestamptz NOT NULL DEFAULT now(),
  bottom_up_fee_usd numeric     NOT NULL,
  top_down_fee_usd  numeric     NOT NULL,
  ratio             numeric     NOT NULL,
  wallets_tracked   integer     NOT NULL,
  note              text
);
