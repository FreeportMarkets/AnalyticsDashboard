'use server'

/**
 * Referrals admin Server Actions -- the ONLY code path in this dashboard
 * that mutates production state (promo codes on the trading-backend).
 *
 * Every action here:
 *   1. Re-checks authentication itself via `auth()`. Middleware gates the
 *      /referrals route, but a Server Action is its OWN entry point (it can
 *      be invoked directly, e.g. from a stale client bundle after a
 *      session expires) -- never assume middleware already covered it.
 *   2. Calls the trading-backend admin API via `@/lib/points` (the
 *      x-admin-key never leaves the server -- points.ts only reads
 *      ADMIN_API_KEY server-side and this file has no client boundary).
 *   3. Writes one `audit_log` row per successful mutation: actor email
 *      (from the session, not a client-supplied field), action, target
 *      code, and the exact request payload sent to the backend. Streamlit
 *      has NO audit trail at all -- every write from this dashboard is a
 *      real improvement over the tool it replaces.
 *   4. Redirects back to /referrals with `?ok=` + `?msg=` so the page can
 *      render the backend's error message VERBATIM rather than swallowing
 *      it into a generic failure banner.
 */

import { redirect } from 'next/navigation'
import { revalidatePath } from 'next/cache'
import { auth } from '@/auth'
import { sql } from '@/lib/db'
import {
  createPromoCode,
  disablePromoCode,
  attachPromoBeneficiary,
  PointsApiError,
  type CreatePromoPayload,
} from '@/lib/points'

async function requireActorEmail(): Promise<string> {
  const session = await auth()
  const email = session?.user?.email
  if (!email) {
    // No session at the Server Action boundary -- reject outright rather
    // than trusting that middleware already handled it.
    throw new Error('Not authenticated.')
  }
  return email
}

/**
 * Best-effort audit row. If the insert itself fails, that failure is
 * appended to the user-facing message (never silently dropped) but does
 * NOT roll back or hide the fact that the backend mutation already
 * succeeded -- the promo code really was created/disabled/updated.
 */
async function logAudit(
  actorEmail: string,
  action: string,
  target: string | null,
  payload: unknown
): Promise<string | null> {
  try {
    await sql(
      `INSERT INTO audit_log (actor_email, action, target, payload) VALUES ($1, $2, $3, $4::jsonb)`,
      [actorEmail, action, target, JSON.stringify(payload ?? null)]
    )
    return null
  } catch (e) {
    return `audit log write failed: ${e instanceof Error ? e.message : String(e)}`
  }
}

function redirectWithResult(ok: boolean, message: string): never {
  const params = new URLSearchParams({ ok: ok ? '1' : '0', msg: message })
  redirect(`/referrals?${params.toString()}`)
}

function str(fd: FormData, key: string): string {
  const v = fd.get(key)
  return typeof v === 'string' ? v.trim() : ''
}

function num(fd: FormData, key: string): number {
  const v = str(fd, key)
  const n = Number(v)
  return Number.isFinite(n) ? n : 0
}

// --- Create -----------------------------------------------------------

export async function createPromoCodeAction(formData: FormData): Promise<void> {
  const actorEmail = await requireActorEmail()

  const code = str(formData, 'code').toUpperCase()
  const kind = str(formData, 'reward_kind')

  if (!code) redirectWithResult(false, 'Code is required.')
  if (kind !== 'points' && kind !== 'multiplier_boost' && kind !== 'chest_grant') {
    redirectWithResult(false, `Unknown reward kind: ${kind || '(empty)'}`)
  }

  let reward_value: CreatePromoPayload['reward_value']
  if (kind === 'points') {
    const amount = str(formData, 'amount')
    if (!amount) redirectWithResult(false, 'Points amount is required.')
    reward_value = { amount }
  } else if (kind === 'multiplier_boost') {
    const multiplier = str(formData, 'multiplier')
    const durationHours = num(formData, 'duration_hours')
    if (!multiplier || durationHours <= 0) {
      redirectWithResult(false, 'Multiplier and duration hours are required.')
    }
    reward_value = { multiplier, duration_hours: durationHours }
  } else {
    const tier = str(formData, 'tier')
    if (!tier) redirectWithResult(false, 'Chest tier is required.')
    reward_value = { tier }
  }

  const payload: CreatePromoPayload = { code, reward_kind: kind, reward_value }

  const maxRedemptions = num(formData, 'max_redemptions')
  if (maxRedemptions > 0) payload.max_redemptions = maxRedemptions

  const expiresAt = str(formData, 'expires_at')
  if (expiresAt) payload.expires_at = expiresAt

  if (kind === 'points') {
    const kickbackRate = str(formData, 'kickback_rate')
    if (kickbackRate) payload.kickback_rate = kickbackRate
    const championBonus = num(formData, 'champion_redeem_bonus')
    if (championBonus > 0) payload.champion_redeem_bonus = championBonus
    const kickbackCap = num(formData, 'kickback_cap_per_referee')
    if (kickbackCap > 0) payload.kickback_cap_per_referee = kickbackCap
  }

  let ok = false
  let message: string
  try {
    const res = await createPromoCode(payload)
    ok = true
    message = `Created ${res.code ?? code}.`
    const auditWarning = await logAudit(actorEmail, 'promo.create', res.code ?? code, payload)
    if (auditWarning) message += ` (${auditWarning})`
  } catch (e) {
    message = e instanceof PointsApiError ? e.message : `Unexpected error: ${String(e)}`
  }

  revalidatePath('/referrals')
  redirectWithResult(ok, message)
}

// --- Disable ------------------------------------------------------------

export async function disablePromoCodeAction(formData: FormData): Promise<void> {
  const actorEmail = await requireActorEmail()

  const code = str(formData, 'code').toUpperCase()
  const reason = str(formData, 'reason')
  // Typed re-confirmation: must retype the exact code, not just tick a box.
  // This is enforced server-side (not just an HTML `required` attribute) so
  // a crafted / replayed request can't skip it.
  const confirmCode = str(formData, 'confirm_code').toUpperCase()

  if (!code) redirectWithResult(false, 'Code is required.')
  if (!reason) redirectWithResult(false, 'A reason is required to disable a code (recorded in the audit log).')
  if (confirmCode !== code) {
    redirectWithResult(false, `Confirmation text "${confirmCode}" did not match code "${code}" -- nothing was disabled.`)
  }

  let ok = false
  let message: string
  try {
    await disablePromoCode(code, reason)
    ok = true
    message = `Disabled ${code}.`
    const auditWarning = await logAudit(actorEmail, 'promo.disable', code, { reason })
    if (auditWarning) message += ` (${auditWarning})`
  } catch (e) {
    message = e instanceof PointsApiError ? e.message : `Unexpected error: ${String(e)}`
  }

  revalidatePath('/referrals')
  redirectWithResult(ok, message)
}

// --- Attach beneficiary ---------------------------------------------------

export async function attachBeneficiaryAction(formData: FormData): Promise<void> {
  const actorEmail = await requireActorEmail()

  const code = str(formData, 'code').toUpperCase()
  const did = str(formData, 'did') || null
  const wallet = str(formData, 'wallet') || null
  const force = formData.get('force') === 'on'

  if (!code) redirectWithResult(false, 'Code is required.')
  if (!did && !wallet) redirectWithResult(false, 'Provide either a DID or a wallet address.')

  const payload = { did, wallet, force }

  let ok = false
  let message: string
  try {
    await attachPromoBeneficiary(code, payload)
    ok = true
    message = `Beneficiary attached to ${code}.`
    const auditWarning = await logAudit(actorEmail, 'promo.beneficiary.attach', code, payload)
    if (auditWarning) message += ` (${auditWarning})`
  } catch (e) {
    message = e instanceof PointsApiError ? e.message : `Unexpected error: ${String(e)}`
  }

  revalidatePath('/referrals')
  redirectWithResult(ok, message)
}
