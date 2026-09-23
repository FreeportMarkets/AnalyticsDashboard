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
- Neon / DynamoDB / Privy credentials for the other pages — see `web/src/lib/`.

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
