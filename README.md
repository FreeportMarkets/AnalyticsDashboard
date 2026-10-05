# Analytics Dashboard

The live dashboard is the **Next.js app in [`web/`](./web)**, deployed on Vercel.

## Running locally

```bash
cd web
npm install
npm run dev
```

Environment variables (Vercel project settings / `web/.env.local`):
- `ANALYTICS_FUNNEL_READ_SECRET` — reads account cohorts, legacy device intro diagnostics and deposit events from the trading backend (`/v1/analytics/accounts`, `/v1/analytics/funnel`, `/v1/analytics/deposits`). Value is in Secrets Manager.
- `HL_VOLUME_SOURCE=backend` — after the backend fill-ledger endpoint is deployed and source parity is accepted, read Trades and Overview perp volume from `/v1/analytics/hl-ledger` using the same server-side secret. Until enabled, the existing Neon volume table remains the read source. The Trades page then shows recipient-proven fees separately from unresolved builder fees.
- Neon / DynamoDB / Privy credentials for the other pages — see `web/src/lib/`.

## Recent perp fills (October 5 candidate)

Deploy the backend `/v1/analytics/hl-ledger/trades` endpoint before this dashboard change. Recent trades uses that endpoint with the existing server-side read secret regardless of `HL_VOLUME_SOURCE`; spot rows remain in Neon. Perps are individual recorded fills with actual notional, not legacy order estimates. Partial fills stay separate; unavailable reads show an explicit error. Client/leverage are unknown where execution evidence does not supply them. Existing aggregate panels and volume-source selection are unchanged. Fills become visible after collector ingestion; this is not an instant order-status feed.

Identity and funding enrichment fail independently: loaded trades and any successful enrichment remain visible. Recorded sizes display the original decimal string, including quantities below eight decimal places. Equal-time pagination uses the same descending timestamp/wallet/text fill-ID ordering as the backend merged in PR527; mixed spot/perp and same-wallet ties have regression coverage.

Read-only production verification matched eight owner fills totaling $330.889960. The actual 91-day reader returned 1001 rows in 194ms including connection; PostgreSQL execution was 1.454ms. Deployed HTTP verification remains a backend rollout gate. The actual Trades page was rendered locally using two verified BTC fills, retaining full wallet controls and 0.00053 size precision. Three pre-existing integration suites require `TEST_DATABASE_URL` even to collect; no production database writes were used to run them.

## History

The original dashboard was a single-file **Streamlit** app (`app.py`) hosted at
`freeportanalyticsdashboard.streamlit.app`. It was **deprecated and removed on
2026-08-17** — it had already been superseded by the Next.js app in `web/`, which
ported its logic (see the "ported from app.py" notes in `web/src/lib/` and the
page files). Recover the old Streamlit code from git history if ever needed.

## Docs

- [Creator qualification and manual payments](docs/creator-payments.md) — the Creators page, ledger boundaries, payout steps and backend-first release order.

- [Account measurement and dashboard source guide](docs/account-measurement.md).
- Legacy device intro diagnostics: `docs/analytics-funnel.md` in `freeport-trading-backend`. These are event reach diagnostics, not verified new-account or funding conversion.
