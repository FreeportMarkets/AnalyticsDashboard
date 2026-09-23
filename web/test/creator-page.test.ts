import { describe, it, expect, vi } from "vitest";
import { createRequire } from "node:module";
import fs from "node:fs";
const { renderToStaticMarkup } = createRequire(import.meta.url)(
  "react-dom/server",
);
vi.mock("@/auth", () => ({
  auth: async () => ({ user: { email: "qa@example.test" } }),
}));
vi.mock("@/app/creators/actions", () => ({
  creatorPayoutAction: async () => {},
}));
vi.mock("@/lib/creators", () => ({
  creatorRequest: vi.fn(),
  payoutPath: "creator-payouts",
  creatorQuery: (p: Record<string, string>) =>
    new URLSearchParams(Object.entries(p).filter(([, v]) => !!v)).toString(),
}));
import CreatorsPage from "@/app/creators/page";
import { creatorRequest } from "@/lib/creators";
const statement = {
  as_of: "2026-09-23T00:00:00Z",
  rows: [],
  totals: [],
  next_cursor: null,
  limitations: [],
};
const creator = {
  id: "11111111-1111-4111-8111-111111111111",
  display_name: "Test creator",
  privy_did: "did:privy:creator",
  referral_code: "FREE-TEST",
  status: "active",
};
const progress = {
  referee_did: "did:privy:referee",
  creator_id: creator.id,
  referral_code: "FREE-TEST",
  status: "pending",
  qualifying_deposit_usd: "110",
  qualifying_days: 29,
  trades_opened: 3,
  last_counted_date: "2026-09-22",
  rate_locked: null,
  fee_share_until: null,
  last_evaluated_at: "2026-09-23T00:00:00Z",
  progress_reason: "awaiting_observed_hold_days",
};
function fixture(path: string) {
  if (path === "creators") return { creators: [creator] };
  if (path.includes("/statement")) return statement;
  if (path.includes("/progress"))
    return { rows: [progress], next_cursor: null };
  if (path.includes("/coverage"))
    return {
      rows: [
        {
          venue: "hyperliquid",
          fee_token: "USDC",
          fill_count: 7,
          verified_fill_count: 0,
          missing_proof_count: 7,
          verified_fee_usd: "0",
          raw_fee_token_amount: "0.205765",
          supported_for_fee_share: true,
        },
      ],
    };
  if (path.includes("/batches")) return { rows: [], next_cursor: null };
  throw new Error(path);
}
describe("Creator operator page", () => {
  it("renders the backend object-shaped coverage and separates eligibility from owed money", async () => {
    vi.mocked(creatorRequest).mockImplementation(
      async (path) => fixture(path) as never,
    );
    const html = renderToStaticMarkup(
      await CreatorsPage({ searchParams: Promise.resolve({}) }),
    );
    expect(html).toContain("29 / 30");
    expect(html).toContain("3 / 3");
    expect(html).toContain("Missing proof");
    expect(html).toContain("No earnings have been recorded");
    expect(html).toContain("Payment batches");
  });
  it("renders frozen payment review and separate unpaid cancellation controls", async () => {
    const id = "22222222-2222-4222-8222-222222222222";
    const batch = {
      id,
      status: "frozen",
      currency: "USD",
      beneficiary_did: "did:privy:test-creator",
      destination: "Test destination · Solana USDC",
      total_usd: "10.000000",
      payout_reference: null,
      allocations: [
        {
          ledger_kind: "creator_earning",
          ledger_id: id,
          amount_usd: "10.000000",
        },
      ],
    };
    vi.mocked(creatorRequest).mockImplementation(async (path) => {
      if (path.endsWith(`/batches/${id}`)) return batch as never;
      if (path.includes("/statement"))
        return {
          ...statement,
          rows: [
            {
              ledger_kind: "creator_earning",
              ledger_id: id,
              creator_id: creator.id,
              beneficiary_did: batch.beneficiary_did,
              obligation_type: "qualified_user",
              amount_usd: "10.000000",
              currency: "USD",
              payout_stage: "approved_unpaid",
              batch_id: id,
              flagged: false,
            },
          ],
          totals: [
            {
              ledger_kind: "creator_earning",
              currency: "USD",
              cooling_usd: "0",
              reviewable_usd: "0",
              approved_unpaid_usd: "10",
              paid_usd: "0",
            },
          ],
        } as never;
      if (path.includes("/batches"))
        return { rows: [batch], next_cursor: null } as never;
      return fixture(path) as never;
    });
    const html = renderToStaticMarkup(
      await CreatorsPage({ searchParams: Promise.resolve({ batch: id }) }),
    );
    expect(html).toContain("Record payment completed");
    expect(html).toContain("No external payment was sent");
    expect(html).toContain('name="confirm_unpaid"');
    expect(html).toContain('name="confirm_paid"');
    expect(html).toContain("$10.000000");
    expect(html).toContain('disabled=""');
    if (process.env.CREATOR_QA_HTML)
      fs.writeFileSync(
        process.env.CREATOR_QA_HTML.replace(".html", "-payout.html"),
        `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/qa.css"></head><body><p style="padding:1rem">Synthetic payout fixture · local visual verification · no real payment</p>${html}</body></html>`,
      );
  });
  it("shows limitations and binds each stage to its own action form", async () => {
    const row = {
      ledger_kind: "creator_earning",
      creator_id: creator.id,
      beneficiary_did: "did:privy:test",
      obligation_type: "qualified_user",
      amount_usd: "10.000000",
      currency: "USD",
      batch_id: null,
      flagged: false,
    };
    vi.mocked(creatorRequest).mockImplementation(async (path) => {
      if (path.includes("/statement"))
        return {
          ...statement,
          limitations: ["Unresolved source evidence remains excluded."],
          rows: [
            {
              ...row,
              ledger_id: "33333333-3333-4333-8333-333333333333",
              payout_stage: "reviewable",
            },
            {
              ...row,
              ledger_id: "44444444-4444-4444-8444-444444444444",
              payout_stage: "approved_unpaid",
            },
          ],
        } as never;
      return fixture(path) as never;
    });
    const html = renderToStaticMarkup(
      await CreatorsPage({ searchParams: Promise.resolve({}) }),
    );
    expect(html).toContain("Unresolved source evidence remains excluded.");
    expect(html).toMatch(
      /form="creator-approve-form"[^>]*value="creator_earning:33333333/,
    );
    expect(html).toMatch(
      /form="creator-freeze-form"[^>]*value="creator_earning:44444444/,
    );
  });
  it("shows an unavailable state instead of false zero balances", async () => {
    vi.mocked(creatorRequest).mockRejectedValue(
      new Error("Creator service unavailable."),
    );
    const html = renderToStaticMarkup(
      await CreatorsPage({ searchParams: Promise.resolve({}) }),
    );
    expect(html).toContain('role="alert"');
    expect(html).toContain("Unavailable data is not a zero balance");
  });
  it.skipIf(!process.env.CREATOR_QA_SOURCE)(
    "renders retained actual-source report for visual QA",
    async () => {
      const data = JSON.parse(
        fs.readFileSync(process.env.CREATOR_QA_SOURCE!, "utf8"),
      );
      vi.mocked(creatorRequest).mockImplementation(async (path) => {
        if (path === "creators") return data.creators;
        if (path.includes("/statement")) return data.statement;
        if (path.includes("/progress")) return data.progress;
        if (path.includes("/coverage")) return data.coverage;
        if (path.includes("/batches")) return data.batches;
        throw new Error(path);
      });
      const html = renderToStaticMarkup(
        await CreatorsPage({ searchParams: Promise.resolve({}) }),
      );
      fs.writeFileSync(
        process.env.CREATOR_QA_HTML!,
        `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/qa.css"></head><body><p style="padding:1rem">Captured production data · local read-only visual verification</p>${html}</body></html>`,
      );
      expect(html).toContain("Bryan Reed");
    },
  );
});
