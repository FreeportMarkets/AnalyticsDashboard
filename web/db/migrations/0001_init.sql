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
