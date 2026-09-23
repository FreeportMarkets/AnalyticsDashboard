import { randomUUID } from "node:crypto";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/auth";
import { PageHeader } from "@/components/PageHeader";
import { DataTable } from "@/components/DataTable";
import {
  creatorRequest,
  creatorQuery,
  payoutPath,
  type Creator,
  type Statement,
  type Progress,
  type Coverage,
  type Batch,
} from "@/lib/creators";
import { creatorPayoutAction } from "./actions";

export const dynamic = "force-dynamic";
const input =
  "rounded-sm border border-hairline bg-surface px-3 py-2 text-sm text-ink-1 focus-visible:outline-accent";
const button =
  "rounded-sm border border-hairline px-3 py-2 text-sm hover:bg-surface focus-visible:outline-accent";
const date = (s: string | null) => (s ? s.slice(0, 10) : "Not yet");
const money = (s: string) => `$${s}`;
const labels: Record<string, string> = {
  balance_evidence_missing: "Paused — waiting for usable balance evidence",
  qualified: "Qualified",
  awaiting_evaluation: "Awaiting evaluation",
  awaiting_qualifying_deposit: "Needs a card deposit ≥ $100",
  awaiting_observed_hold_days: "Collecting verified balance days",
  awaiting_trade_gate: "Needs three separate perp positions",
};

export default async function CreatorsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const p = await searchParams;
  const val = (k: string) =>
    typeof p[k] === "string" ? (p[k] as string) : undefined;
  const creatorId = val("creator");
  const query = creatorQuery({ creator_id: creatorId });
  let creators: Creator[],
    statement: Statement,
    progress: { rows: Progress[]; next_cursor: string | null },
    coverage: Coverage[],
    batch: Batch | null,
    batches: { rows: Batch[]; next_cursor: string | null };
  try {
    [creators, statement, progress, coverage, batch, batches] =
      await Promise.all([
        creatorRequest<{ creators: Creator[] }>("creators").then(
          (r) => r.creators,
        ),
        creatorRequest<Statement>(
          `${payoutPath}/statement?${creatorQuery({ creator_id: creatorId, cursor: val("cursor") })}`,
        ),
        creatorRequest<{ rows: Progress[]; next_cursor: string | null }>(
          `${payoutPath}/progress?${creatorQuery({ creator_id: creatorId, after: val("after") })}`,
        ),
        creatorRequest<{ rows: Coverage[] }>(
          `${payoutPath}/coverage?${query}`,
        ).then((r) => r.rows),
        val("batch")
          ? creatorRequest<Batch>(
              `${payoutPath}/batches/${encodeURIComponent(val("batch")!)}`,
            )
          : Promise.resolve(null),
        creatorRequest<{ rows: Batch[]; next_cursor: string | null }>(
          `${payoutPath}/batches?${creatorQuery({ after: val("batches_after") })}`,
        ),
      ]);
  } catch (error) {
    return (
      <main className="mx-auto w-full max-w-[1600px] p-8">
        <PageHeader
          title="Creators"
          subtitle="Qualification, earnings and manual payments"
        />
        <p role="alert" className="mt-8 text-alert">
          {error instanceof Error
            ? error.message
            : "Creator reports unavailable."}
        </p>
        <Link href="/creators" className={`${button} mt-4 inline-block`}>
          Reload report
        </Link>
        <p className="mt-3 text-sm text-ink-2">
          Unavailable data is not a zero balance. If a payment request timed
          out, check its batch before paying again.
        </p>
      </main>
    );
  }
  const names = new Map(creators.map((c) => [c.id, c.display_name]));
  const nextLink = (changes: Record<string, string | undefined>) =>
    `/creators?${creatorQuery({ creator: creatorId, ...changes })}`;
  return (
    <main className="mx-auto min-w-0 w-full max-w-[1600px] px-4 py-8 sm:px-8">
      <PageHeader
        title="Creators"
        subtitle="Track qualification, review earnings and record manual payments."
      />
      {val("message") && (
        <p
          role="status"
          className={`mt-5 ${val("ok") === "1" ? "text-positive" : "text-alert"}`}
        >
          {val("message")}
        </p>
      )}
      <form method="get" className="my-6 flex flex-wrap items-end gap-3">
        <label className="grid gap-1 text-sm text-ink-2">
          Creator
          <select
            name="creator"
            defaultValue={creatorId ?? ""}
            className={input}
          >
            <option value="">All creators</option>
            {creators.map((c) => (
              <option key={c.id} value={c.id}>
                {c.display_name} · {c.status}
              </option>
            ))}
          </select>
        </label>
        <button className={button}>View report</button>
        <Link href={`/creators/export?${query}`} className={button}>
          Export full statement (CSV)
        </Link>
      </form>
      <section
        className="border-t border-hairline py-6"
        aria-labelledby="progress-heading"
      >
        <h2 id="progress-heading" className="text-lg font-semibold">
          Referral progress
        </h2>
        <p className="mt-2 max-w-[75ch] text-sm text-ink-2">
          One confirmed card/onramp deposit of at least $100, three separate
          perp positions, and 30 verified completed UTC days at $100 or more.
          Missing readings pause progress; a verified balance below $100 resets
          it. Trading losses, fees and withdrawals can put an account below the
          threshold.
        </p>
        <div className="mt-4 overflow-x-auto">
          <DataTable
            rows={progress.rows}
            rowKey={(r) => r.referee_did}
            columns={[
              {
                key: "creator",
                header: "Creator / code",
                render: (r) => (
                  <>
                    {names.get(r.creator_id) ?? r.creator_id}
                    <small className="block text-ink-2">
                      {r.referral_code}
                    </small>
                  </>
                ),
              },
              {
                key: "user",
                header: "Referred account",
                render: (r) => (
                  <span className="break-all text-xs">{r.referee_did}</span>
                ),
              },
              {
                key: "deposit",
                header: "Qualifying deposit",
                render: (r) =>
                  r.qualifying_deposit_usd
                    ? money(r.qualifying_deposit_usd)
                    : "Not yet",
              },
              {
                key: "days",
                header: "Verified days",
                render: (r) => `${r.qualifying_days} / 30`,
              },
              {
                key: "trades",
                header: "Positions",
                render: (r) => `${r.trades_opened} / 3`,
              },
              {
                key: "reason",
                header: "Progress",
                render: (r) => (
                  <>
                    {labels[r.progress_reason] ?? r.progress_reason}
                    <small className="block text-ink-2">
                      Last balance day: {date(r.last_counted_date)}
                    </small>
                    <small className="block text-ink-2">
                      Evaluated: {date(r.last_evaluated_at)}
                    </small>
                  </>
                ),
              },
              {
                key: "rate",
                header: "Locked share / expiry",
                render: (r) =>
                  r.rate_locked
                    ? `${Number(r.rate_locked) * 100}% · ${date(r.fee_share_until)}`
                    : "Locks at qualification",
              },
            ]}
          />
        </div>
        {!progress.rows.length && (
          <p className="mt-2 text-sm text-ink-2">
            No enrolled referrals in this view. Only referrals made after
            creator approval enter this programme.
          </p>
        )}
        {progress.next_cursor && (
          <Link
            className={`${button} mt-3 inline-block`}
            href={nextLink({ after: progress.next_cursor })}
          >
            Next referrals
          </Link>
        )}
      </section>
      <section
        className="border-t border-hairline py-6"
        aria-labelledby="earnings-heading"
      >
        <h2 id="earnings-heading" className="text-lg font-semibold">
          Earnings and bonuses
        </h2>
        <p className="mt-2 text-sm text-ink-2">
          Creator earnings include the creator’s $10 qualification bonus and
          verified fee share. Referee bonuses are a separate $10 owed to each
          qualified referred user. Creator earnings have a seven-day review
          delay; referee bonuses can be reviewed on qualification.
        </p>
        <div className="mt-4 overflow-x-auto">
          <DataTable
            rows={statement.totals}
            rowKey={(r) => `${r.ledger_kind}:${r.currency}`}
            columns={[
              {
                key: "kind",
                header: "Beneficiary ledger",
                render: (r) =>
                  r.ledger_kind === "creator_earning"
                    ? "Creator earnings"
                    : "Referee bonuses",
              },
              {
                key: "cool",
                header: "Cooling",
                render: (r) => `${money(r.cooling_usd)} ${r.currency}`,
              },
              {
                key: "review",
                header: "Ready for review",
                render: (r) => money(r.reviewable_usd),
              },
              {
                key: "unpaid",
                header: "Approved, unpaid",
                render: (r) => money(r.approved_unpaid_usd),
              },
              {
                key: "paid",
                header: "Recorded paid",
                render: (r) => money(r.paid_usd),
              },
            ]}
          />
        </div>
        <p className="mt-2 text-xs text-ink-2">
          Statement cutoff: {statement.as_of}. Totals include all rows in this
          filter, across pages. Refresh before preparing a payment.
        </p>
        <form action={creatorPayoutAction} className="mt-5 space-y-4">
          <input type="hidden" name="creator_id" value={creatorId ?? ""} />
          <input type="hidden" name="idempotency_key" value={randomUUID()} />
          <div className="overflow-x-auto">
            <DataTable
              rows={statement.rows}
              rowKey={(r) => `${r.ledger_kind}:${r.ledger_id}`}
              columns={[
                {
                  key: "select",
                  header: "Select",
                  render: (r) => (
                    <input
                      aria-label={`Select ${r.obligation_type} ${r.ledger_id}`}
                      type="checkbox"
                      name="row"
                      value={`${r.ledger_kind}:${r.ledger_id}`}
                      disabled={
                        !!r.batch_id ||
                        !["reviewable", "approved_unpaid"].includes(
                          r.payout_stage,
                        )
                      }
                    />
                  ),
                },
                {
                  key: "kind",
                  header: "Obligation",
                  render: (r) => (
                    <>
                      {r.obligation_type.replaceAll("_", " ")}
                      <small className="block text-ink-2">
                        {names.get(r.creator_id) ?? r.creator_id}
                      </small>
                    </>
                  ),
                },
                {
                  key: "beneficiary",
                  header: "Pay this account",
                  render: (r) => (
                    <span className="break-all text-xs">
                      {r.beneficiary_did ??
                        "Missing creator identity — resolve before paying"}
                    </span>
                  ),
                },
                {
                  key: "amount",
                  header: "Exact amount",
                  render: (r) => `${money(r.amount_usd)} ${r.currency}`,
                },
                {
                  key: "stage",
                  header: "Status",
                  render: (r) => (
                    <>
                      {r.payout_stage.replaceAll("_", " ")}
                      {r.flagged && (
                        <small className="block text-alert">
                          Risk flagged — review first
                        </small>
                      )}
                      {r.batch_id && (
                        <Link
                          className="block text-accent underline"
                          href={nextLink({ batch: r.batch_id })}
                        >
                          View payment batch
                        </Link>
                      )}
                    </>
                  ),
                },
                {
                  key: "reference",
                  header: "Payment reference",
                  render: (r) => (
                    <span className="break-all">
                      {r.payout_reference ?? "—"}
                    </span>
                  ),
                },
              ]}
            />
          </div>
          {!statement.rows.length && (
            <p className="text-sm text-ink-2">
              No earnings have been recorded in this view. Qualification must
              complete before either bonus or the six-month fee-share window
              begins.
            </p>
          )}
          {!!statement.rows.length && (
            <>
              <button
                className={button}
                name="operation"
                value="approve"
                formNoValidate
              >
                Approve selected reviewable rows
              </button>
              <fieldset className="border border-hairline p-4">
                <legend className="px-2 text-sm font-medium">
                  Prepare a manual payment
                </legend>
                <p className="mb-3 max-w-[75ch] text-sm text-ink-2">
                  Select approved rows for one beneficiary. Verify their payout
                  destination directly. Preparing a batch reserves these rows
                  and fixes the exact amount; it does not send money.
                </p>
                <div className="flex flex-wrap items-end gap-3">
                  <label className="grid flex-1 gap-1 text-sm">
                    Beneficiary DID
                    <input
                      name="beneficiary_did"
                      required
                      className={input}
                      placeholder="did:privy:…"
                    />
                  </label>
                  <label className="grid flex-1 gap-1 text-sm">
                    Verified destination (include network)
                    <input
                      name="destination"
                      required
                      className={input}
                      placeholder="e.g. Solana USDC · wallet address"
                    />
                  </label>
                  <button className={button} name="operation" value="freeze">
                    Prepare selected payment
                  </button>
                </div>
              </fieldset>
            </>
          )}
        </form>
        {statement.next_cursor && (
          <Link
            className={`${button} mt-3 inline-block`}
            href={nextLink({ cursor: statement.next_cursor })}
          >
            Next statement rows
          </Link>
        )}
      </section>
      <section
        className="border-t border-hairline py-6"
        aria-labelledby="recent-batches-heading"
      >
        <h2 id="recent-batches-heading" className="text-lg font-semibold">
          Payment batches · all creators
        </h2>
        <p className="my-2 text-sm text-ink-2">
          Check this list after a timeout before preparing or sending another
          payment. Paid and cancelled batches remain in the audit trail.
        </p>
        <div className="overflow-x-auto">
          <DataTable
            rows={batches.rows}
            rowKey={(r) => r.id}
            columns={[
              {
                key: "id",
                header: "Batch",
                render: (r) => (
                  <Link
                    className="text-accent underline"
                    href={nextLink({ batch: r.id })}
                  >
                    {r.id}
                  </Link>
                ),
              },
              {
                key: "did",
                header: "Beneficiary",
                render: (r) => (
                  <span className="break-all text-xs">{r.beneficiary_did}</span>
                ),
              },
              {
                key: "total",
                header: "Exact total",
                render: (r) => `${money(r.total_usd)} ${r.currency}`,
              },
              { key: "status", header: "Status", render: (r) => r.status },
              {
                key: "ref",
                header: "Payment reference",
                render: (r) => (
                  <span className="break-all">
                    {r.payout_reference ?? "Not recorded"}
                  </span>
                ),
              },
            ]}
          />
        </div>
        {batches.next_cursor && (
          <Link
            className={`${button} mt-3 inline-block`}
            href={nextLink({ batches_after: batches.next_cursor })}
          >
            Older batches
          </Link>
        )}
      </section>
      {batch && (
        <section
          className="border-t border-hairline py-6"
          aria-labelledby="batch-heading"
        >
          <h2 id="batch-heading" className="text-lg font-semibold">
            Payment batch · {batch.status}
          </h2>
          <dl className="my-4 grid gap-2 text-sm sm:grid-cols-[10rem_1fr]">
            <dt className="text-ink-2">Batch</dt>
            <dd className="break-all">{batch.id}</dd>
            <dt className="text-ink-2">Beneficiary</dt>
            <dd className="break-all">{batch.beneficiary_did}</dd>
            <dt className="text-ink-2">Destination</dt>
            <dd className="break-all">{batch.destination}</dd>
            <dt className="text-ink-2">Exact amount</dt>
            <dd className="numeral">
              {money(batch.total_usd)} {batch.currency}
            </dd>
            <dt className="text-ink-2">Ledger rows</dt>
            <dd>{batch.allocations.length}</dd>
            <dt className="text-ink-2">Payment reference</dt>
            <dd className="break-all">
              {batch.payout_reference ?? "Not recorded"}
            </dd>
          </dl>
          <details className="mb-4 text-sm">
            <summary className="cursor-pointer">
              Show allocated row IDs and amounts
            </summary>
            <ul className="mt-2 space-y-1">
              {batch.allocations.map((a) => (
                <li
                  key={`${a.ledger_kind}:${a.ledger_id}`}
                  className="break-all"
                >
                  {a.ledger_kind} · {a.ledger_id} · {money(a.amount_usd)}
                </li>
              ))}
            </ul>
          </details>
          {batch.status === "frozen" && (
            <form action={creatorPayoutAction} className="space-y-3">
              <input type="hidden" name="batch_id" value={batch.id} />
              <input type="hidden" name="creator_id" value={creatorId ?? ""} />
              <p className="max-w-[75ch] text-sm text-ink-2">
                Pay the beneficiary outside this dashboard, then record the
                payment here. This records your confirmation; it does not verify
                the external transfer.
              </p>
              <label className="grid max-w-xl gap-1 text-sm">
                Transaction hash or payment reference
                <input required name="payout_reference" className={input} />
              </label>
              <label className="flex items-start gap-2 text-sm">
                <input required type="checkbox" name="confirm_paid" />I
                completed the external payment for this exact destination and
                amount.
              </label>
              <div className="flex flex-wrap gap-3">
                <button
                  name="operation"
                  value="paid"
                  className={`${button} bg-accent text-canvas`}
                >
                  Record payment completed
                </button>
              </div>
            </form>
          )}
          {batch.status === "frozen" && (
            <details className="mt-5 text-sm">
              <summary className="cursor-pointer">Cancel this batch</summary>
              <form action={creatorPayoutAction} className="mt-3 space-y-3">
                <input type="hidden" name="batch_id" value={batch.id} />
                <input
                  type="hidden"
                  name="creator_id"
                  value={creatorId ?? ""}
                />
                <label className="flex items-start gap-2">
                  <input required type="checkbox" name="confirm_unpaid" />
                  No external payment was sent for this batch. Release its rows
                  for a new payment.
                </label>
                <button name="operation" value="cancel" className={button}>
                  Cancel unpaid batch
                </button>
              </form>
            </details>
          )}
        </section>
      )}
      <section
        className="border-t border-hairline py-6"
        aria-labelledby="coverage-heading"
      >
        <h2 id="coverage-heading" className="text-lg font-semibold">
          Fee source coverage
        </h2>
        <p className="my-2 max-w-[75ch] text-sm text-ink-2">
          Only proven Freeport-earned USDC fees during a qualified user’s locked
          window can produce creator fee share. Source totals below include
          enrolled users before qualification; they are not payout amounts.
          Missing proof is held for recovery. Other currencies are shown without
          a USD conversion.
        </p>
        <div className="overflow-x-auto">
          <DataTable
            rows={coverage}
            rowKey={(r) => `${r.venue}:${r.fee_token}`}
            columns={[
              {
                key: "source",
                header: "Source",
                render: (r) => `${r.venue} · ${r.fee_token}`,
              },
              {
                key: "fills",
                header: "Raw fee fills",
                render: (r) => r.fill_count,
              },
              {
                key: "proof",
                header: "Proven fills",
                render: (r) => r.verified_fill_count,
              },
              {
                key: "missing",
                header: "Missing proof",
                render: (r) => r.missing_proof_count,
              },
              {
                key: "fees",
                header: "Verified earned fees",
                render: (r) => money(r.verified_fee_usd),
              },
              {
                key: "support",
                header: "Fee-share support",
                render: (r) =>
                  r.supported_for_fee_share
                    ? "Supported"
                    : `Excluded · ${r.raw_fee_token_amount} ${r.fee_token}`,
              },
            ]}
          />
        </div>
      </section>
    </main>
  );
}
