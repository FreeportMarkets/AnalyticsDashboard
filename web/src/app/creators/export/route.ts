import { auth } from "@/auth";
import {
  creatorRequest,
  creatorQuery,
  payoutPath,
  type Statement,
  type StatementRow,
} from "@/lib/creators";
export const dynamic = "force-dynamic";
export const maxDuration = 60;
const columns: (keyof StatementRow)[] = [
  "ledger_kind",
  "ledger_id",
  "creator_id",
  "beneficiary_did",
  "referee_did",
  "obligation_type",
  "amount_usd",
  "currency",
  "status",
  "payout_stage",
  "created_at",
  "eligible_at",
  "payout_reference",
  "batch_id",
  "flagged",
];
const cell = (v: unknown) => {
  let s = v == null ? "" : String(v);
  if (/^[\s]*[=+\-@]/.test(s)) s = "'" + s;
  return `"${s.replaceAll('"', '""')}"`;
};
export async function GET(request: Request) {
  if (!(await auth())?.user?.email)
    return new Response("Unauthorized", { status: 401 });
  const creator =
    new URL(request.url).searchParams.get("creator_id") ?? undefined;
  const output = [columns.join(",")];
  let cursor: string | undefined;
  const start = Date.now();
  try {
    do {
      if (Date.now() - start > 45_000)
        return new Response(
          "Export exceeded its time budget. Select one creator and retry.",
          { status: 503 },
        );
      const page = await creatorRequest<Statement>(
        `${payoutPath}/statement?${creatorQuery({ creator_id: creator, cursor, limit: "500" })}`,
      );
      output.push(
        ...page.rows.map((row) => columns.map((c) => cell(row[c])).join(",")),
      );
      cursor = page.next_cursor ?? undefined;
    } while (cursor);
    return new Response(output.join("\r\n") + "\r\n", {
      headers: {
        "content-type": "text/csv; charset=utf-8",
        "content-disposition":
          'attachment; filename="freeport-creator-statement.csv"',
        "cache-control": "private, no-store",
      },
    });
  } catch {
    return new Response(
      "Statement unavailable. No partial export was produced.",
      { status: 503 },
    );
  }
}
