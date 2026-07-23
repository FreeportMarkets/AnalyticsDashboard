# Perps volume tracking (HL builder-fee authoritative)

## Why this exists

The dashboard used to compute perps volume by reconstructing notional from
our own `trades` table (`amount_usd × leverage`). That number is wrong two
ways, both proven against production:

1. **Per-wallet under-logging.** `trade-logger.ts` in the backend writes one
   row per order, fire-and-forget, only on the main `/v1/orders` path. TWAP,
   stop/take, copy, and liquidation fills carry our builder fee (so HL counts
   them as ours) but skip that logger. Measured: one wallet had 1 logged row
   vs 20 real HL fills.
2. **Whole-wallet omission.** ~Half of all builder fees we've collected come
   from wallets that appear in **neither** our `trades` nor `events` tables.
   Authentication (Privy) is complete; trade *logging* is not.

Net effect: the reconstruction shows roughly **half** the real volume.

## The authoritative source

Every Hyperliquid fill carries `builderFee`. HL charges that fee only on
orders placed through *our* builder code and credits it to our collector
wallet `0x9f4e80F17Ddb4A7efC1dc07fAE6B34AbAb77d6Df`. So:

> **Freeport perps volume = Σ(|sz| × px) over every HL fill with
> `builderFee > 0`, deduped by `tid`, across every Freeport wallet.**

Three properties make this the source of truth:

- **Rate-independent.** The builder-fee rate changed over time (5 → 10 → 6.5
  bps). Irrelevant: each fill states its own `sz` and `px`, so notional is
  read directly, never derived from the fee.
- **Client-agnostic.** Attribution is HL-side, so FreeApp, Web Terminal, and
  any future client are covered automatically. Immune to the logging bug.
- **Self-verifying.** Σ(`builderFee`) computed bottom-up MUST equal the fees
  HL actually paid our collector (claimed + pending). When they tie to <1%,
  enumeration is provably complete — you cannot reproduce the exact fee total
  any other way. This is the acceptance gate, not a nicety.

## Enumeration — the one hard dependency

The method needs the complete list of Freeport wallet addresses. The
`trades`/`events` tables are NOT it (that incompleteness is the bug). The
authoritative list is the backend `user_wallets` table (`evm_address`,
`privy_did`). `walletSource.ts` abstracts this:

- `analyticsWalletSource` — wallets already in our Neon `trades` ∪ `events`.
  Works today, but only reaches ~51% of fees. The current default.
- `registryWalletSource` — reads `user_wallets` from the backend via
  `WALLET_REGISTRY_DATABASE_URL`. The complete source; closes the gap to
  <1%. Wire this once a read-only prod URL (or a periodic sync of that table
  into Neon) is available.

The reconciliation check reports which coverage we actually have on every
run, so switching sources is verifiable rather than hopeful.

## Data model (analytics Neon)

- **`tracked_wallets`** — the enumeration set. `evm_address` (PK, lowercased),
  `source`, timestamps. Fed by the wallet source.
- **`wallet_volume_daily`** — per wallet per NY day: `notional_usd`,
  `builder_fee_usd`, `fill_count`. PK `(evm_address, day)`. The grain the
  dashboard aggregates. NY-day bucketing matches every other metric.
- **`hl_fill_sync_state`** — per-wallet watermark: `last_fill_ms` (the max
  fill `time` we've ingested). Incremental syncs fetch only fills after it.
- **`volume_reconciliation`** — one row per reconcile run: bottom-up fee,
  top-down fee, ratio. The <1% health log.

Volume totals come from HL fills. The **client** breakdown (mobile/web) is
the one thing HL fills cannot provide — it stays sourced from the trade log
and is therefore approximate; totals are authoritative, client split is not.

## Flow

1. **Backfill** (one-time) — enumerate every wallet, pull all HL fills back to
   builder-code launch (2026-05-07; nothing earlier carries a builder fee),
   write `wallet_volume_daily`, set each watermark to its max fill time.
2. **Incremental cron** — every N minutes: for each tracked wallet, fetch
   fills after its watermark, upsert affected days, advance the watermark.
   Durability matters: HL ages fills out of `userFillsByTime`, so capturing
   continuously is the only way to not lose the per-fill record permanently.
3. **Reconcile** — each run compares Σ(`builder_fee_usd`) to the collector's
   claimed+pending fees; logs the ratio. <1% gap = complete.
4. **Dashboard** — reads `wallet_volume_daily`. No HL calls on page load.

## Pre-fee tail (Feb–May 7)

Before builder code, fills carry no `builderFee`, so HL cannot attribute them
to Freeport. Privy embedded wallets are app-controlled, so a wallet's entire
pre-fee HL history is *probably* all-Freeport — but only if HL still retains
fills that far back. This tail has **no authoritative source**; it is
estimated separately (all-fills-if-retained, else the legacy logged estimate)
and reported as an estimate, never mixed into the authoritative total.
