import type { neon } from '@neondatabase/serverless'
import { fetchPrivyUsers, type PrivyWalletMap, type DidIdentity, type LoginType } from './privy'

type SqlTag = ReturnType<typeof neon>

/**
 * Postgres mirror of Privy identities (see db/migrations/0004_privy_identities.sql).
 *
 * WHY THIS FILE EXISTS: `fetchPrivyUsers()` in `./privy.ts` is a live,
 * paginated Privy REST call that measured 14.4s cold and indexes ~5,830
 * wallets. It was being awaited INSIDE the render path of /trades, /users,
 * and /referrals -- on Vercel, every cold serverless instance re-runs it, so
 * clicking around repeatedly hits fresh instances and stalls for up to a
 * minute. This module replaces that render-path call with a single indexed
 * Postgres query scoped to only the wallets/DIDs a given page actually needs,
 * backed by a table refreshed nightly (see `refreshPrivyIdentities`, wired
 * into the existing /api/cron/reconcile cron).
 *
 * `./privy.ts` itself is untouched -- `fetchPrivyUsers`/`fetchPrivyUserByDid`
 * still talk to the live Privy API and are reused here (by the nightly
 * refresh) and by /referrals (as the capped per-DID fallback for identities
 * this mirror hasn't caught up to yet, e.g. a user created since last
 * night's refresh).
 */

const UPSERT_CHUNK = 1000

export interface RefreshResult {
  /** Wallets returned by the live Privy bulk listing this run. */
  fetchedWallets: number
  /** Rows actually inserted/updated in privy_identities. */
  upserted: number
}

/**
 * Refresh the privy_identities mirror from the live Privy API. Intended to
 * be called once per night from /api/cron/reconcile, not from page render.
 *
 * Reuses `fetchPrivyUsers()` (the same paginated bulk listing pages used to
 * call directly) rather than hitting Privy per-wallet -- one 14s-ish fetch a
 * night is the entire cost this table exists to avoid paying on every page
 * render.
 *
 * Batched as UNNEST + ON CONFLICT upserts (one Postgres statement per batch
 * of up to `UPSERT_CHUNK` rows) -- the Neon HTTP driver sends exactly one
 * statement per call, so a 5,830-row table cannot be upserted as individual
 * INSERTs without 5,830 round trips.
 *
 * Does NOT catch its own errors -- a genuine DB failure (e.g. the migration
 * hasn't been applied yet) should surface to the caller loudly, same as
 * reconcileEvents/reconcileTrades. `fetchPrivyUsers()` itself already fails
 * soft to an empty map on any Privy-side failure, so a Privy outage alone
 * degrades this to a zero-row no-op rather than an error.
 */
export async function refreshPrivyIdentities(sql: SqlTag): Promise<RefreshResult> {
  const map: PrivyWalletMap = await fetchPrivyUsers()
  const entries = Array.from(map.entries())
  const now = new Date().toISOString()

  let upserted = 0
  for (let i = 0; i < entries.length; i += UPSERT_CHUNK) {
    const chunk = entries.slice(i, i + UPSERT_CHUNK)
    const result = await sql(
      `INSERT INTO privy_identities (wallet_address, did, label, login_type, contact, synced_at)
       SELECT * FROM UNNEST(
         $1::text[], $2::text[], $3::text[], $4::text[], $5::text[], $6::timestamptz[]
       )
       ON CONFLICT (wallet_address) DO UPDATE SET
         did        = excluded.did,
         label      = excluded.label,
         login_type = excluded.login_type,
         contact    = excluded.contact,
         synced_at  = excluded.synced_at`,
      [
        chunk.map(([wallet]) => wallet),
        chunk.map(([, ident]) => ident.privyDid || null),
        chunk.map(([, ident]) => ident.label),
        chunk.map(([, ident]) => ident.loginType),
        chunk.map(([, ident]) => ident.contact),
        chunk.map(() => now),
      ],
      { fullResults: true }
    )
    upserted += result.rowCount
  }

  return { fetchedWallets: entries.length, upserted }
}

/**
 * Page-scoped identity lookup: reads privy_identities for exactly the
 * wallets a page needs (deduped, lowercased), not the entire table. Returns
 * a `PrivyWalletMap` -- the same shape `fetchPrivyUsers()` used to return --
 * so `labelForWallet` / `TraderCell` need no changes at all; they don't know
 * or care whether the map came from a live bulk fetch or this mirror.
 *
 * FAILS SOFT: an empty table (migration not yet run/populated) or any query
 * error (DB hiccup) resolves to an empty map, never throws. `labelForWallet`
 * already falls back to a truncated address when a wallet has no entry, so
 * this degrades gracefully to "no labels today", not a broken page.
 */
export async function fetchWalletIdentities(sql: SqlTag, wallets: Iterable<string>): Promise<PrivyWalletMap> {
  const map: PrivyWalletMap = new Map()
  const addrs = Array.from(new Set(Array.from(wallets, w => (w ?? '').toLowerCase()).filter(Boolean)))
  if (addrs.length === 0) return map

  try {
    const rows = (await sql(
      `SELECT wallet_address, did, label, login_type, contact
         FROM privy_identities
        WHERE wallet_address = ANY($1::text[])`,
      [addrs]
    )) as Array<{
      wallet_address: string
      did: string | null
      label: string | null
      login_type: string | null
      contact: string | null
    }>

    for (const r of rows) {
      map.set(r.wallet_address, {
        privyDid: r.did ?? '',
        label: r.label,
        loginType: (r.login_type as LoginType | null) ?? 'wallet_only',
        contact: r.contact,
        // The migration schema has no separate email column -- `contact` is
        // already the email/phone/etc. contact string extractIdentity()
        // derived at write time, so it doubles as `email` on read.
        email: r.contact,
      })
    }
  } catch {
    // Fail soft -- see doc comment above.
  }

  return map
}

/**
 * DID-scoped identity lookup for /referrals (top referrers, promo code
 * beneficiaries, and redemption rows are keyed by Privy DID, not wallet
 * address). Reads privy_identities via the `did` index for exactly the DIDs
 * requested. Same fail-soft contract as `fetchWalletIdentities`.
 */
export async function fetchIdentitiesByDid(sql: SqlTag, dids: Iterable<string>): Promise<Map<string, DidIdentity>> {
  const out = new Map<string, DidIdentity>()
  const list = Array.from(new Set(Array.from(dids))).filter((d): d is string => !!d && d.startsWith('did:privy:'))
  if (list.length === 0) return out

  try {
    const rows = (await sql(
      `SELECT did, label, login_type, contact
         FROM privy_identities
        WHERE did = ANY($1::text[])`,
      [list]
    )) as Array<{ did: string | null; label: string | null; login_type: string | null; contact: string | null }>

    for (const r of rows) {
      if (!r.did) continue
      out.set(r.did, {
        label: r.label ?? '',
        email: r.contact ?? '',
        loginType: (r.login_type as LoginType | null) ?? '',
      })
    }
  } catch {
    // Fail soft -- see fetchWalletIdentities' doc comment.
  }

  return out
}
