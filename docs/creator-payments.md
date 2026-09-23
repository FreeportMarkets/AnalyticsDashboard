# Creator qualification and manual payments

## Purpose and source

The **Creators** navigation item opens `/creators`. It uses the trading backend's
creator qualification records and two financial ledgers. Ordinary points referral
activation is not creator qualification. This page does not use AppsFlyer totals
to decide eligibility or creator payments.

The page supports creator filtering, paginated referral progress, ledger totals,
a full paginated CSV export, fee-proof coverage, and a persistent payment-batch
history. Source failures render unavailable states, never zero balances.

## Manual payment workflow

1. Review the referred account's confirmed card funding, verified balance days
   and separate perp positions. Missing data pauses qualification; a complete
   below-$100 day resets the balance count. Only post-approval referrals count.
2. Review ledger rows. Creator earnings include their $10 qualification bonus
   and fee share; referee bonuses are a separate $10 payable to each referee.
   Creator earnings have a seven-day cooling period. Referee bonuses have no
   extra cooling period after qualification. Risk flags require review.
3. Approve selected reviewable rows. Select approved rows belonging to **one
   beneficiary**, enter that exact DID and a verified destination including the
   payment network, then prepare the batch.
4. Review the frozen total, destination and allocated row IDs. Pay outside this
   dashboard. The software does not initiate or independently verify a transfer.
5. Record the transaction hash/payment reference and confirm completion. Exact
   rows become paid atomically. Retrying the same reference is idempotent.
6. Cancel only if no external payment was sent. Cancellation preserves the audit
   record and releases rows for another batch. After any timeout, check batch
   history before sending or preparing another payment.

USD amounts are shown without rounding their ledger precision. Unsupported fee
currencies and missing proof are not converted or included in payable USD.
Source coverage includes pre-qualification activity and is not a payment total.
CSV dates are ledger creation dates; statuses reflect each read. Frozen batch
allocations are the authoritative payment record.

## Access and rollout

Existing Google sign-in allowlists and server-side `ADMIN_API_KEY` protect this
surface. Every Server Action and export rechecks the session. The verified
session email supplies the backend audit label; credentials never reach the
browser. The backend still authenticates the dashboard through its shared admin
key, not independent human credentials. Optional `POINTS_API_BASE_URL` supports
isolated staging/testing and defaults to the existing production points API.

Ship [backend PR470](https://github.com/FreeportMarkets/freeport-trading-backend/pull/470) and migration0170 first, verify its running
version/endpoints, then ship this dashboard PR. A missing backend route fails
closed with an unavailable message. No FreeApp update is required for this page.
Neither PR creates payments or changes production merely by being opened.

## Validation

Actual captured production enrollment/progress was rendered through the page and
inspected in Chrome. A clearly labelled synthetic payout fixture verifies the
otherwise empty payment flow without creating real obligations. Tests exercise
session enforcement, actor provenance, review/cancel confirmations, idempotency,
CSV pagination/formula escaping and explicit incomplete-export errors. Backend
PostgreSQL tests cover atomic settlement, concurrent reservations, reversals,
precision and both beneficiary ledgers. Staging verification remains a release
gate; local rendering is not a production deployment claim.
