"use server";
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import {
  creatorRequest,
  CreatorApiError,
  payoutPath,
  type Batch,
  type LedgerKey,
} from "@/lib/creators";

const field = (f: FormData, name: string) =>
  typeof f.get(name) === "string" ? String(f.get(name)).trim() : "";
function selection(f: FormData): LedgerKey[] {
  const rows = f.getAll("row").map((value) => {
    const [ledger_kind, ledger_id] = String(value).split(":");
    if (
      (ledger_kind !== "creator_earning" && ledger_kind !== "referee_bonus") ||
      !ledger_id ||
      !/^[0-9a-f-]{36}$/i.test(ledger_id)
    )
      throw new CreatorApiError("Select valid statement rows.");
    return { ledger_kind, ledger_id } as LedgerKey;
  });
  if (!rows.length || rows.length > 500)
    throw new CreatorApiError("Select between 1 and 500 rows.");
  return rows;
}
export async function creatorPayoutAction(f: FormData): Promise<void> {
  const session = await auth();
  const actor = session?.user?.email;
  if (!actor) throw new Error("Not authenticated.");
  const operation = field(f, "operation");
  const result = new URLSearchParams();
  const creator = field(f, "creator_id");
  if (creator) result.set("creator", creator);
  try {
    if (operation === "approve") {
      await creatorRequest(`${payoutPath}/approve`, {
        method: "POST",
        actor,
        body: { rows: selection(f) },
      });
      result.set(
        "message",
        "Selected rows approved. They are ready to include in a manual payment batch.",
      );
    } else if (operation === "freeze") {
      const batch = await creatorRequest<Batch>(`${payoutPath}/batches`, {
        method: "POST",
        actor,
        body: {
          rows: selection(f),
          idempotency_key: field(f, "idempotency_key"),
          beneficiary_did: field(f, "beneficiary_did"),
          destination: field(f, "destination"),
        },
      });
      result.set("batch", batch.id);
      result.set(
        "message",
        "Payment batch prepared. Review the exact beneficiary, destination and total below before paying externally.",
      );
    } else if (operation === "paid" || operation === "cancel") {
      const id = field(f, "batch_id");
      if (!/^[0-9a-f-]{36}$/i.test(id))
        throw new CreatorApiError("Invalid batch.");
      if (
        operation === "cancel" &&
        (field(f, "confirm_unpaid") !== "on" ||
          field(f, "confirm_paid") === "on")
      )
        throw new CreatorApiError(
          "Confirm that no external payment was sent before cancelling this batch.",
        );
      if (operation === "paid" && field(f, "confirm_paid") !== "on")
        throw new CreatorApiError(
          "Confirm that the external payment has completed.",
        );
      await creatorRequest(`${payoutPath}/batches/${id}/${operation}`, {
        method: "POST",
        actor,
        body:
          operation === "paid"
            ? { payout_reference: field(f, "payout_reference") }
            : {},
      });
      result.set("batch", id);
      result.set(
        "message",
        operation === "paid"
          ? "Payment recorded. The exact rows in this batch are marked paid."
          : "Batch cancelled. Its rows are available for a new batch.",
      );
    } else throw new CreatorApiError("Invalid action.");
    result.set("ok", "1");
  } catch (error) {
    result.set(
      "message",
      error instanceof CreatorApiError
        ? error.message
        : "The request failed. Refresh the statement before retrying.",
    );
    const id = field(f, "batch_id");
    if (id) result.set("batch", id);
  }
  revalidatePath("/creators");
  redirect(`/creators?${result}`);
}
