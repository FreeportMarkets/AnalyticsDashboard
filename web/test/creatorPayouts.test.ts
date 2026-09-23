import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
const fakes = vi.hoisted(() => ({
  auth: vi.fn(),
  request: vi.fn(),
  redirect: vi.fn(),
  revalidate: vi.fn(),
}));
vi.mock("server-only", () => ({}));
vi.mock("@/auth", () => ({ auth: fakes.auth }));
vi.mock("next/navigation", () => ({ redirect: fakes.redirect }));
vi.mock("next/cache", () => ({ revalidatePath: fakes.revalidate }));
vi.mock("@/lib/creators", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/creators")>()),
  creatorRequest: fakes.request,
}));
import { creatorPayoutAction } from "@/app/creators/actions";
import { GET } from "@/app/creators/export/route";
import { CreatorApiError } from "@/lib/creators";
const batchId = "12345678-1234-1234-1234-123456789abc",
  rowId = "23456789-1234-1234-1234-123456789abc";
const form = (values: Record<string, string>) => {
  const f = new FormData();
  Object.entries(values).forEach(([k, v]) => f.append(k, v));
  return f;
};
const redirectStop = new Error("test redirect");
async function action(values: Record<string, string>) {
  await expect(creatorPayoutAction(form(values))).rejects.toBe(redirectStop);
  return new URL(
    fakes.redirect.mock.calls.at(-1)![0],
    "https://analytics.example",
  );
}
const row = (id: string) => ({
  ledger_kind: "creator_earning",
  ledger_id: id,
  creator_id: batchId,
  beneficiary_did: "did:privy:creator",
  referee_did: "did:privy:referee",
  obligation_type: "qualified_user",
  amount_usd: "10",
  currency: "USD",
  status: "approved",
  payout_stage: "approved_unpaid",
  created_at: "2026-09-01",
  eligible_at: "2026-09-08",
  payout_reference: null,
  batch_id: null,
  flagged: false,
});
const page = (rows: unknown[], next: string | null) => ({
  as_of: "2026-09-22",
  rows,
  totals: [],
  next_cursor: next,
  limitations: [],
});
beforeEach(() => {
  vi.resetAllMocks();
  fakes.auth.mockResolvedValue({
    user: { email: "founder@freeportmarkets.com" },
  });
  fakes.request.mockResolvedValue({ id: batchId });
  fakes.redirect.mockImplementation(() => {
    throw redirectStop;
  });
});
afterEach(() => vi.restoreAllMocks());
describe("creator finance server actions", () => {
  it("rejects missing sessions before any backend call", async () => {
    fakes.auth.mockResolvedValue(null);
    await expect(
      creatorPayoutAction(
        form({ operation: "approve", row: `creator_earning:${rowId}` }),
      ),
    ).rejects.toThrow("Not authenticated");
    expect(fakes.request).not.toHaveBeenCalled();
  });
  it("requires completed-payment confirmation on the server", async () => {
    const result = await action({
      operation: "paid",
      batch_id: batchId,
      payout_reference: "chain:123",
    });
    expect(fakes.request).not.toHaveBeenCalled();
    expect(result.searchParams.get("message")).toContain("Confirm");
    expect(result.searchParams.get("batch")).toBe(batchId);
    expect(result.searchParams.has("ok")).toBe(false);
  });
  it("requires unpaid confirmation and rejects conflicting paid confirmation on cancellation", async () => {
    await action({ operation: "cancel", batch_id: batchId });
    expect(fakes.request).not.toHaveBeenCalled();
    await action({
      operation: "cancel",
      batch_id: batchId,
      confirm_unpaid: "on",
      confirm_paid: "on",
    });
    expect(fakes.request).not.toHaveBeenCalled();
    await action({
      operation: "cancel",
      batch_id: batchId,
      confirm_unpaid: "on",
    });
    expect(fakes.request).toHaveBeenCalledWith(
      `creator-payouts/batches/${batchId}/cancel`,
      expect.objectContaining({
        method: "POST",
        actor: "founder@freeportmarkets.com",
      }),
    );
  });
  it("uses session email rather than a supplied actor field", async () => {
    await action({
      operation: "approve",
      row: `creator_earning:${rowId}`,
      actor: "forged@attacker.test",
    });
    expect(fakes.request).toHaveBeenCalledWith("creator-payouts/approve", {
      method: "POST",
      actor: "founder@freeportmarkets.com",
      body: { rows: [{ ledger_kind: "creator_earning", ledger_id: rowId }] },
    });
  });
  it("preserves the rendered idempotency key when preparing payment", async () => {
    const result = await action({
      operation: "freeze",
      row: `creator_earning:${rowId}`,
      idempotency_key: "stable-key",
      beneficiary_did: "did:privy:creator",
      destination: "Solana USDC: wallet",
    });
    expect(fakes.request.mock.calls[0]![1].body.idempotency_key).toBe(
      "stable-key",
    );
    expect(result.searchParams.get("batch")).toBe(batchId);
  });
  it("records confirmed references and retains batch recovery on timeout", async () => {
    await action({
      operation: "paid",
      batch_id: batchId,
      payout_reference: " chain:123 ",
      confirm_paid: "on",
    });
    expect(fakes.request).toHaveBeenCalledWith(
      `creator-payouts/batches/${batchId}/paid`,
      expect.objectContaining({ body: { payout_reference: "chain:123" } }),
    );
    fakes.request.mockRejectedValueOnce(
      new CreatorApiError(
        "Refresh to check whether your last action completed before retrying.",
      ),
    );
    const result = await action({
      operation: "paid",
      batch_id: batchId,
      payout_reference: "chain:123",
      confirm_paid: "on",
    });
    expect(result.searchParams.get("batch")).toBe(batchId);
    expect(result.searchParams.has("ok")).toBe(false);
  });
});
describe("creator full statement CSV", () => {
  it("rejects unauthenticated exports before querying data", async () => {
    fakes.auth.mockResolvedValue(null);
    expect(
      (await GET(new Request("https://analytics.example/creators/export")))
        .status,
    ).toBe(401);
    expect(fakes.request).not.toHaveBeenCalled();
  });
  it("exhausts pages, retains filters, emits one header and neutralizes spreadsheet formulas", async () => {
    fakes.request
      .mockResolvedValueOnce(
        page(
          [{ ...row(rowId), payout_reference: '=HYPERLINK("bad")' }],
          "cursor-page-2",
        ),
      )
      .mockResolvedValueOnce(page([row(batchId)], null));
    const response = await GET(
      new Request(
        `https://analytics.example/creators/export?creator_id=${batchId}`,
      ),
    );
    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toContain("no-store");
    const csv = await response.text();
    expect(csv.match(/ledger_kind,ledger_id/g)).toHaveLength(1);
    expect(csv).toContain(rowId);
    expect(csv).toContain('"\'=HYPERLINK(""bad"")"');
    const second = new URL(
      fakes.request.mock.calls[1]![0],
      "https://backend.example/",
    );
    expect(second.searchParams.get("creator_id")).toBe(batchId);
    expect(second.searchParams.get("cursor")).toBe("cursor-page-2");
    expect(second.searchParams.get("limit")).toBe("500");
  });
  it("never serves a partial statement as complete when a later page fails", async () => {
    fakes.request
      .mockResolvedValueOnce(page([row(rowId)], "next"))
      .mockRejectedValueOnce(new Error("timeout"));
    const response = await GET(
      new Request("https://analytics.example/creators/export"),
    );
    expect(response.status).toBe(503);
    expect(await response.text()).not.toContain(rowId);
  });
});
