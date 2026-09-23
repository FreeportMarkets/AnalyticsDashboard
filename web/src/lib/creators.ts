import "server-only";

export type LedgerKey = {
  ledger_kind: "creator_earning" | "referee_bonus";
  ledger_id: string;
};
export type Creator = {
  id: string;
  display_name: string;
  privy_did: string | null;
  referral_code: string | null;
  status: string;
};
export type StatementRow = LedgerKey & {
  creator_id: string;
  beneficiary_did: string | null;
  referee_did: string | null;
  obligation_type: string;
  amount_usd: string;
  currency: string;
  status: string;
  payout_stage: string;
  created_at: string;
  eligible_at: string;
  payout_reference: string | null;
  batch_id: string | null;
  flagged: boolean;
};
export type Statement = {
  as_of: string;
  rows: StatementRow[];
  totals: {
    ledger_kind: string;
    currency: string;
    cooling_usd: string;
    reviewable_usd: string;
    approved_unpaid_usd: string;
    paid_usd: string;
    row_count: number;
  }[];
  next_cursor: string | null;
  limitations: string[];
};
export type Progress = {
  referee_did: string;
  creator_id: string;
  referral_code: string;
  status: string;
  qualifying_deposit_usd: string | null;
  qualifying_days: number;
  trades_opened: number;
  last_counted_date: string | null;
  rate_locked: string | null;
  fee_share_until: string | null;
  last_evaluated_at: string | null;
  progress_reason: string;
  latest_balance_date?: string | null;
  latest_balance_complete?: boolean | null;
  latest_balance_known_usd?: string | null;
};
export type Coverage = {
  venue: string;
  fee_token: string;
  fill_count: number;
  verified_fill_count: number;
  missing_proof_count: number;
  verified_fee_usd: string;
  raw_fee_token_amount: string;
  supported_for_fee_share: boolean;
};
export type Batch = {
  id: string;
  status: "frozen" | "paid" | "cancelled";
  currency: string;
  beneficiary_did: string;
  destination: string;
  total_usd: string;
  payout_reference: string | null;
  paid_at: string | null;
  created_by: string;
  allocations: (LedgerKey & {
    amount_usd: string;
    released_at: string | null;
  })[];
};

export class CreatorApiError extends Error {}
/** Server-only admin credential; callers must independently verify the session. */
export async function creatorRequest<T>(
  path: string,
  opts: { method?: "GET" | "POST"; body?: unknown; actor?: string } = {},
): Promise<T> {
  const key = process.env.ADMIN_API_KEY;
  if (!key)
    throw new CreatorApiError(
      "Creator reports are unavailable: the server admin credential is not configured.",
    );
  const base =
    process.env.POINTS_API_BASE_URL ??
    "https://trading-api.freeportmarkets.com/v1/points";
  let response: Response;
  try {
    response = await fetch(`${base}/admin/${path}`, {
      method: opts.method ?? "GET",
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
      headers: {
        "content-type": "application/json",
        "x-admin-key": key,
        "x-admin-id": opts.actor ?? "analytics-dashboard-web",
      },
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
    });
  } catch {
    throw new CreatorApiError(
      "The creator service did not respond. Refresh to check whether your last action completed before retrying.",
    );
  }
  const data = await response.json().catch(() => null);
  if (!response.ok || !data)
    throw new CreatorApiError(
      typeof data?.error === "string"
        ? data.error
        : `Creator service unavailable (${response.status}).`,
    );
  return data as T;
}
export const payoutPath = "creator-payouts";
export function creatorQuery(values: Record<string, string | undefined>) {
  return new URLSearchParams(
    Object.entries(values).filter((v): v is [string, string] => !!v[1]),
  ).toString();
}
