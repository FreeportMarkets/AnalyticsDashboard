import { auth, signOut } from '@/auth'
import {
  fetchPromoCodes,
  fetchPromoCodeDetail,
  fetchTopReferrers,
  type PromoCode,
  type RewardValue,
} from '@/lib/points'
import { fetchPrivyUserByDid, type DidIdentity } from '@/lib/privy'
import { sql } from '@/lib/db'
import { fetchIdentitiesByDid } from '@/lib/privyIdentities'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { StatTile } from '@/components/StatTile'
import { DataTable } from '@/components/DataTable'
import { createPromoCodeAction, disablePromoCodeAction, attachBeneficiaryAction } from './actions'

export const dynamic = 'force-dynamic'

const compact = (n: number) => Math.round(n).toLocaleString('en-US')

// --- Formatting helpers (ported from app.py's tab_referrals section) ------

function titleCase(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

/** Truncate a long ISO-ish timestamp string to `YYYY-MM-DD HH:MM:SS`. */
function fmtTs(s: string | null | undefined): string {
  return s ? s.slice(0, 19) : '—'
}

function rewardValueLabel(kind: string | undefined, rv: RewardValue | null | undefined): string {
  const v = rv ?? {}
  switch (kind) {
    case 'points':
      return `${v.amount ?? '?'} pts`
    case 'multiplier_boost':
      return `${v.multiplier ?? '?'}× / ${v.duration_hours ?? '?'}h`
    case 'chest_grant':
      return `${titleCase(String(v.tier ?? '?'))} chest`
    default:
      return kind ?? '?'
  }
}

function progressLabel(c: PromoCode): string {
  const current = c.current_redemptions ?? 0
  return c.max_redemptions != null ? `${current}/${c.max_redemptions}` : `${current} (unlimited)`
}

function shortDid(did: string): string {
  return did.length > 24 ? `${did.slice(0, 24)}…` : did
}

/** "label (email)" when both present and distinct, else whichever exists, else a truncated DID. */
function formatIdentity(did: string, label?: string | null, email?: string | null): string {
  if (label && email && label !== email) return `${label} (${email})`
  return label || email || shortDid(did)
}

// Hard cap on live per-DID Privy fallback calls per page render. The
// privy_identities table (refreshed nightly, see src/lib/privyIdentities.ts)
// covers the vast majority of DIDs; this fallback exists only for identities
// created since the last refresh. Without a cap, a page with hundreds of
// uncovered DIDs (e.g. right after a promo-code campaign) could fire
// hundreds of sequential Privy network calls in one render.
const MAX_DID_FALLBACK = 25

/**
 * Resolve a set of Privy DIDs to display labels. Two-tier, matching
 * app.py's `_beneficiary_display` / `enrich_did_df`: the privy_identities
 * mirror first (a single indexed Postgres query, see `fetchIdentitiesByDid`
 * in src/lib/privyIdentities.ts), then a capped parallel per-DID live fetch
 * for anything the mirror doesn't have yet.
 * FAILS SOFT: fetchPrivyUserByDid already never throws, so any resolution
 * failure just falls back to the truncated DID string here.
 */
async function resolveDidLabels(dids: Iterable<string>, didMap: Map<string, DidIdentity>): Promise<Map<string, string>> {
  const out = new Map<string, string>()
  const needFallback: string[] = []
  for (const did of new Set(dids)) {
    if (!did) continue
    const ident = didMap.get(did)
    if (ident && (ident.label || ident.email)) {
      out.set(did, formatIdentity(did, ident.label, ident.email))
    } else if (did.startsWith('did:privy:')) {
      needFallback.push(did)
    } else {
      out.set(did, shortDid(did))
    }
  }
  if (needFallback.length > 0) {
    const capped = needFallback.slice(0, MAX_DID_FALLBACK)
    const overflow = needFallback.slice(MAX_DID_FALLBACK)
    const fetched = await Promise.all(capped.map(d => fetchPrivyUserByDid(d)))
    capped.forEach((did, i) => {
      const f = fetched[i]
      out.set(did, f ? formatIdentity(did, f.label, f.email) : shortDid(did))
    })
    // Past the cap: truncated DID rather than an unbounded fetch burst.
    for (const did of overflow) out.set(did, shortDid(did))
  }
  return out
}

const inputCls =
  'w-full rounded-sm border border-hairline bg-surface px-2 py-1.5 text-sm text-ink-1 outline-none transition-colors focus:border-accent'
const labelCls = 'block text-xs uppercase tracking-wide text-ink-2'
const fieldCls = 'space-y-1'

export default async function ReferralsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const params = await searchParams
  const resultOk = params.ok === '1'
  const resultMsg = typeof params.msg === 'string' ? params.msg : undefined
  const inspectCode = typeof params.inspect === 'string' ? params.inspect.toUpperCase() : undefined

  const [session, codes, topReferrers] = await Promise.all([
    auth(),
    fetchPromoCodes(),
    fetchTopReferrers(50),
  ])

  const detail = inspectCode ? await fetchPromoCodeDetail(inspectCode) : null
  const redemptions = detail?.redemptions ?? []

  const allDids = [
    ...codes.map(c => c.beneficiary_did).filter((d): d is string => !!d),
    ...topReferrers.map(r => r.referrer_did).filter(Boolean),
    ...redemptions.map(r => r.privy_did).filter((d): d is string => !!d),
  ]
  // DID-scoped lookup against the privy_identities mirror -- a single
  // indexed query for exactly the DIDs on this page, not a 14s live Privy
  // fetch of the whole user base. See src/lib/privyIdentities.ts.
  const didMap = await fetchIdentitiesByDid(sql, allDids)
  const didLabels = await resolveDidLabels(allDids, didMap)

  // --- KPIs (client-side over the full list, mirrors app.py) --------------
  const totalCodes = codes.length
  const activeCount = codes.filter(c => c.status === 'active').length
  const totalRedemptions = codes.reduce((sum, c) => sum + (c.current_redemptions ?? 0), 0)
  const totalPoints = codes.reduce((sum, c) => {
    if (c.reward_kind !== 'points') return sum
    const amt = Number(c.reward_value?.amount)
    if (!Number.isFinite(amt)) return sum
    return sum + amt * (c.current_redemptions ?? 0)
  }, 0)

  const activeCodes = codes.filter(c => c.status === 'active').map(c => c.code)
  const inspectable = codes
    .filter(c => (c.current_redemptions ?? 0) > 0)
    .sort((a, b) => (b.current_redemptions ?? 0) - (a.current_redemptions ?? 0))

  return (
    <main className="mx-auto w-full max-w-[1600px] px-8 py-8">
      <PageHeader
        title="Referrals — Promo Codes"
        subtitle={
          <span className="text-alert">
            This tab writes to production. Every mutation below is recorded in the audit log.
          </span>
        }
        right={
          <div className="flex items-center gap-4">
            <form action={async () => { 'use server'; await signOut({ redirectTo: '/login' }) }}>
              <button className="text-xs text-ink-2 outline-none transition-colors hover:text-ink-1 focus-visible:ring-2 focus-visible:ring-accent">
                {session?.user?.email} · sign out
              </button>
            </form>
          </div>
        }
      />

      <div className="animate-content-fade">
        {resultMsg && (
          <div
            role="alert"
            className={`mt-6 rounded-sm border px-3 py-2 text-sm ${
              resultOk
                ? 'border-positive/40 bg-positive/10 text-positive'
                : 'border-negative/40 bg-negative/10 text-negative'
            }`}
          >
            {resultMsg}
          </div>
        )}

        {/* --- KPIs --- */}
        <section aria-label="Promo code KPIs" className="mt-8 grid gap-x-6 divide-y divide-hairline sm:grid-cols-4 sm:divide-x sm:divide-y-0">
          <StatTile label="Total codes" value={totalCodes} format={compact} />
          <div className="sm:pl-6">
            <StatTile label="Active" value={activeCount} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Total redemptions" value={totalRedemptions} format={compact} />
          </div>
          <div className="sm:pl-6">
            <StatTile label="Points distributed" value={totalPoints} format={compact} />
          </div>
        </section>

        {/* --- Codes table --- */}
        <section aria-label="Promo codes" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Promo codes</SectionHeading>
          <div className="mt-3">
            <DataTable
              rowKey={row => row.code}
              rows={codes}
              columns={[
                { key: 'code', header: 'Code', render: r => <span className="numeral">{r.code}</span> },
                {
                  key: 'status',
                  header: 'Status',
                  render: r => (
                    <span className={r.status === 'active' ? 'text-positive' : 'text-ink-3'}>{r.status}</span>
                  ),
                },
                { key: 'type', header: 'Type', render: r => r.reward_kind },
                { key: 'value', header: 'Value', render: r => rewardValueLabel(r.reward_kind, r.reward_value) },
                { key: 'progress', header: 'Uses / limit', align: 'right', render: r => <span className="numeral">{progressLabel(r)}</span> },
                { key: 'created', header: 'Created', align: 'right', render: r => <span className="numeral">{fmtTs(r.created_at)}</span> },
                {
                  key: 'beneficiary',
                  header: 'Beneficiary',
                  render: r =>
                    r.beneficiary_did ? (
                      <span title={r.beneficiary_did}>{didLabels.get(r.beneficiary_did) ?? shortDid(r.beneficiary_did)}</span>
                    ) : (
                      '—'
                    ),
                },
              ]}
            />
          </div>
          <p className="mt-2 text-xs text-ink-3">{compact(totalCodes)} codes total.</p>
        </section>

        {/* --- Inspect redemptions --- */}
        <section aria-label="Inspect redemptions" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Inspect redemptions</SectionHeading>
          {inspectable.length === 0 ? (
            <p className="mt-2 text-xs text-ink-3">No redemptions to inspect yet.</p>
          ) : (
            <>
              <form method="GET" action="/referrals" className="mt-3 flex flex-wrap items-end gap-2">
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="inspect-select">Pick a code to see who redeemed it</label>
                  <select id="inspect-select" name="inspect" defaultValue={inspectCode ?? ''} className={inputCls}>
                    <option value="" disabled>
                      Select a code…
                    </option>
                    {inspectable.map(c => (
                      <option key={c.code} value={c.code}>
                        {c.code} ({compact(c.current_redemptions ?? 0)} redemptions)
                      </option>
                    ))}
                  </select>
                </div>
                <button type="submit" className="rounded-sm border border-hairline px-3 py-1.5 text-xs text-ink-1 hover:bg-surface">
                  Inspect
                </button>
              </form>

              {inspectCode && (
                <div className="mt-4">
                  {!detail ? (
                    <p className="text-xs text-negative">Couldn&apos;t load detail for {inspectCode} — check the API / network.</p>
                  ) : redemptions.length === 0 ? (
                    <p className="text-xs text-ink-3">No redemption rows on {inspectCode} yet.</p>
                  ) : (
                    <>
                      <DataTable
                        rowKey={row => row.id ?? `${row.privy_did}-${row.redeemed_at}`}
                        rows={redemptions}
                        columns={[
                          { key: 'redeemed_at', header: 'Redeemed', align: 'right', render: r => <span className="numeral">{fmtTs(r.redeemed_at)}</span> },
                          {
                            key: 'user',
                            header: 'User',
                            render: r =>
                              r.privy_did ? (
                                <span title={r.privy_did}>{didLabels.get(r.privy_did) ?? shortDid(r.privy_did)}</span>
                              ) : (
                                '—'
                              ),
                          },
                          { key: 'reward', header: 'Reward', render: r => rewardValueLabel(r.reward_kind, r.reward_value) },
                          { key: 'trace', header: 'Trace ID', render: r => <span className="numeral text-ink-3">{r.trace_id ?? '—'}</span> },
                        ]}
                      />
                      <p className="mt-2 text-xs text-ink-3">{compact(redemptions.length)} redemption rows on {inspectCode}.</p>
                    </>
                  )}
                </div>
              )}
            </>
          )}
        </section>

        {/* --- Create code --- */}
        <section aria-label="Create a promo code" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Create a new promo code</SectionHeading>
          <form action={createPromoCodeAction} className="mt-3 max-w-xl space-y-4">
            <div className={fieldCls}>
              <label className={labelCls} htmlFor="c-code">Code (alphanumeric + _-, 2-64 chars)</label>
              <input id="c-code" name="code" required minLength={2} maxLength={64} placeholder="e.g. FREE-IMC2026" className={inputCls} />
              <p className="text-xs text-ink-3">Will be uppercased on save.</p>
            </div>

            <div className={fieldCls}>
              <label className={labelCls} htmlFor="c-kind">Reward kind</label>
              <select id="c-kind" name="reward_kind" required defaultValue="points" className={inputCls}>
                <option value="points">points</option>
                <option value="multiplier_boost">multiplier_boost</option>
                <option value="chest_grant">chest_grant</option>
              </select>
            </div>

            <fieldset className="rounded-sm border border-hairline p-3">
              <legend className="px-1 text-xs uppercase tracking-wide text-ink-2">Reward value — fill only the fields for the selected kind</legend>
              <div className="grid grid-cols-2 gap-3">
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="c-amount">Points amount</label>
                  <input id="c-amount" name="amount" type="number" min={1} step={1} defaultValue={10000} className={inputCls} />
                </div>
                <div />
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="c-mult">Multiplier (e.g. 2.0)</label>
                  <input id="c-mult" name="multiplier" defaultValue="2.0" className={inputCls} />
                </div>
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="c-hours">Duration hours</label>
                  <input id="c-hours" name="duration_hours" type="number" min={1} max={720} defaultValue={24} className={inputCls} />
                </div>
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="c-tier">Chest tier</label>
                  <select id="c-tier" name="tier" defaultValue="gold" className={inputCls}>
                    <option value="wood">wood</option>
                    <option value="silver">silver</option>
                    <option value="gold">gold</option>
                  </select>
                </div>
              </div>
            </fieldset>

            <div className="grid grid-cols-2 gap-3">
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="c-max">Max redemptions (0 = unlimited)</label>
                <input id="c-max" name="max_redemptions" type="number" min={0} defaultValue={50} className={inputCls} />
              </div>
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="c-expires">Expires at (ISO 8601, blank = never)</label>
                <input id="c-expires" name="expires_at" placeholder="2026-12-31T23:59:59Z" className={inputCls} />
              </div>
            </div>

            <fieldset className="rounded-sm border border-hairline p-3">
              <legend className="px-1 text-xs uppercase tracking-wide text-ink-2">Champion kickback (optional, points-only)</legend>
              <div className="grid grid-cols-3 gap-3">
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="c-kickback">Kickback rate (e.g. 0.10)</label>
                  <input id="c-kickback" name="kickback_rate" placeholder="0.10" className={inputCls} />
                </div>
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="c-champion-bonus">Champion redeem bonus (pts)</label>
                  <input id="c-champion-bonus" name="champion_redeem_bonus" type="number" min={0} defaultValue={0} className={inputCls} />
                </div>
                <div className={fieldCls}>
                  <label className={labelCls} htmlFor="c-kickback-cap">Kickback cap / referee (pts)</label>
                  <input id="c-kickback-cap" name="kickback_cap_per_referee" type="number" min={0} defaultValue={0} className={inputCls} />
                </div>
              </div>
            </fieldset>

            <button type="submit" className="rounded-sm bg-accent px-3 py-1.5 text-xs font-medium text-canvas hover:opacity-90">
              Create code
            </button>
          </form>
        </section>

        {/* --- Disable code --- */}
        <section aria-label="Disable a promo code" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Disable / revoke an active code</SectionHeading>
          {activeCodes.length === 0 ? (
            <p className="mt-2 text-xs text-ink-3">No active codes to disable.</p>
          ) : (
            <form action={disablePromoCodeAction} className="mt-3 max-w-xl space-y-4">
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="d-code">Code to disable</label>
                <select id="d-code" name="code" required defaultValue="" className={inputCls}>
                  <option value="" disabled>
                    Select a code…
                  </option>
                  {activeCodes.map(code => (
                    <option key={code} value={code}>{code}</option>
                  ))}
                </select>
              </div>
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="d-reason">Reason (required, recorded in the audit log)</label>
                <input id="d-reason" name="reason" required placeholder="e.g. campaign ended, abuse detected" className={inputCls} />
              </div>
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="d-confirm">
                  Type the code again to confirm — this disables it immediately. Users mid-redeem will get a 409.
                </label>
                <input id="d-confirm" name="confirm_code" required placeholder="Retype the exact code" className={inputCls} />
              </div>
              <button type="submit" className="rounded-sm bg-negative px-3 py-1.5 text-xs font-medium text-canvas hover:opacity-90">
                Disable code
              </button>
            </form>
          )}
        </section>

        {/* --- Attach beneficiary --- */}
        <section aria-label="Attach champion beneficiary" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Attach champion beneficiary to a code</SectionHeading>
          {codes.length === 0 ? (
            <p className="mt-2 text-xs text-ink-3">No codes available.</p>
          ) : (
            <form action={attachBeneficiaryAction} className="mt-3 max-w-xl space-y-4">
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="b-code">Code</label>
                <select id="b-code" name="code" required defaultValue="" className={inputCls}>
                  <option value="" disabled>
                    Select a code…
                  </option>
                  {codes.map(c => (
                    <option key={c.code} value={c.code}>{c.code}</option>
                  ))}
                </select>
              </div>
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="b-did">Beneficiary DID (preferred)</label>
                <input id="b-did" name="did" placeholder="did:privy:..." className={inputCls} />
              </div>
              <div className={fieldCls}>
                <label className={labelCls} htmlFor="b-wallet">OR beneficiary wallet (resolves to DID via Privy)</label>
                <input id="b-wallet" name="wallet" placeholder="0x…" className={inputCls} />
              </div>
              <label className="flex items-center gap-2 text-xs text-ink-2">
                <input type="checkbox" name="force" className="rounded-sm" />
                Force-overwrite an existing beneficiary
              </label>
              <button type="submit" className="rounded-sm border border-hairline px-3 py-1.5 text-xs text-ink-1 hover:bg-surface">
                Attach beneficiary
              </button>
            </form>
          )}
        </section>

        {/* --- Top personal referrers --- */}
        <section aria-label="Top personal referrers" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Top personal referrers</SectionHeading>
          <p className="mt-1 text-xs text-ink-3">
            Users who have referred the most other accounts via their personal FREE-XXXXXX code.
          </p>
          {topReferrers.length === 0 ? (
            <p className="mt-2 text-xs text-ink-3">No personal referral data yet.</p>
          ) : (
            <div className="mt-3">
              <DataTable
                rowKey={row => row.code}
                rows={topReferrers}
                columns={[
                  { key: 'code', header: 'Code', render: r => <span className="numeral">{r.code}</span> },
                  {
                    key: 'referrer',
                    header: 'Referrer',
                    render: r => <span title={r.referrer_did}>{didLabels.get(r.referrer_did) ?? shortDid(r.referrer_did)}</span>,
                  },
                  { key: 'referees', header: 'Referees', align: 'right', render: r => <span className="numeral">{compact(r.referee_count ?? 0)}</span> },
                  { key: 'first', header: 'First referral', align: 'right', render: r => <span className="numeral">{fmtTs(r.first_referral_at)}</span> },
                  { key: 'recent', header: 'Most recent', align: 'right', render: r => <span className="numeral">{fmtTs(r.most_recent_referral_at)}</span> },
                ]}
              />
            </div>
          )}
        </section>
      </div>
    </main>
  )
}
