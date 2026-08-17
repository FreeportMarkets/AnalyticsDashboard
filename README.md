# Analytics Dashboard

The live dashboard is the **Next.js app in [`web/`](./web)**, deployed on Vercel.

## Running locally

```bash
cd web
npm install
npm run dev
```

Environment variables (Vercel project settings / `web/.env.local`):
- `ANALYTICS_FUNNEL_READ_SECRET` — reads the acquisition funnel from the trading backend (`/v1/analytics/funnel`). Value is in Secrets Manager.
- Neon / DynamoDB / Privy credentials for the other pages — see `web/src/lib/`.

## History

The original dashboard was a single-file **Streamlit** app (`app.py`) hosted at
`freeportanalyticsdashboard.streamlit.app`. It was **deprecated and removed on
2026-08-17** — it had already been superseded by the Next.js app in `web/`, which
ported its logic (see the "ported from app.py" notes in `web/src/lib/` and the
page files). Recover the old Streamlit code from git history if ever needed.

## Docs

- Acquisition funnel semantics (what the Journey tab measures, the new-user
  cohort, and how to read it): `docs/analytics-funnel.md` in
  `freeport-trading-backend`, and the team guide published from this repo.
