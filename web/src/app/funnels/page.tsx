import { auth, signOut } from '@/auth'
import {
  availableEvents,
  computeFunnel,
  featureEngagement,
  type BreakdownDim,
} from '@/lib/metrics/funnels'
import { watermarkAge } from '@/lib/metrics/staleness'
import { NY_TZ } from '@/lib/time'
import { RANGES, isRangeKey, rangeStart, todayNy, type RangeKey } from '@/lib/ranges'
import { PageHeader } from '@/components/PageHeader'
import { SectionHeading } from '@/components/SectionHeading'
import { BarList } from '@/components/BarList'
import { DataTable } from '@/components/DataTable'
import { StalenessBadge } from '@/components/StalenessBadge'

export const dynamic = 'force-dynamic'

const WINDOWS = { '1h': 3600, '24h': 86400, '7d': 604800 } as const
type WindowKey = keyof typeof WINDOWS

function isWindowKey(v: string | undefined): v is WindowKey {
  return v === '1h' || v === '24h' || v === '7d'
}

function isBreakdownKey(v: string | undefined): v is BreakdownDim {
  return v === 'platform' || v === 'app_version'
}

/**
 * The three funnels the Streamlit original hardcoded (app.py ~1959-2022),
 * ported as one-click presets over the general composable funnel below.
 */
const PRESETS: Array<{ label: string; steps: string[] }> = [
  { label: 'Notification → Trade', steps: ['notification_received', 'notification_tap', 'trade_initiated', 'trade_success'] },
  { label: 'Trade Funnel', steps: ['buy_button_tap', 'swap_modal_open', 'trade_initiated', 'trade_success'] },
  { label: 'Deposit Funnel', steps: ['deposit_modal_open', 'deposit_initiated', 'deposit_success'] },
]

const STEP_SLOTS = 5

function formatDateRange(start: string, end: string): string {
  const fmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' })
  const startLabel = fmt.format(new Date(`${start}T12:00:00Z`))
  const endLabel = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    .format(new Date(`${end}T12:00:00Z`))
  return `${startLabel} – ${endLabel}`
}

const compact = (n: number) => Math.round(n).toLocaleString('en-US')
const pct = (fraction: number) => `${(fraction * 100).toFixed(1)}%`

function first(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v
}

/** Build an href to /funnels preserving current selection, with overrides. */
function hrefWith(
  current: { range: RangeKey; steps: string[]; windowKey: WindowKey; by: BreakdownDim | undefined },
  overrides: Partial<{ range: RangeKey; steps: string[]; windowKey: WindowKey; by: BreakdownDim | undefined }>
): string {
  const merged = { ...current, ...overrides }
  const params = new URLSearchParams()
  if (merged.range !== '7d') params.set('range', merged.range)
  if (merged.steps.length > 0) params.set('steps', merged.steps.join(','))
  if (merged.windowKey !== '24h') params.set('window', merged.windowKey)
  if (merged.by) params.set('by', merged.by)
  const qs = params.toString()
  return qs ? `/funnels?${qs}` : '/funnels'
}

export default async function FunnelsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}) {
  const params = await searchParams
  const range: RangeKey = isRangeKey(first(params.range)) ? (first(params.range) as RangeKey) : '7d'
  const windowKey: WindowKey = isWindowKey(first(params.window)) ? (first(params.window) as WindowKey) : '24h'
  const by: BreakdownDim | undefined = isBreakdownKey(first(params.by)) ? (first(params.by) as BreakdownDim) : undefined

  const [session, end] = [await auth(), todayNy()]
  const start = rangeStart(end, range)

  // Requested steps: either a `steps=a,b,c` query param (preset links, and
  // the shareable/bookmarkable form) or s1..s5 from the builder form below,
  // which avoids needing any client-side JS to assemble an ordered list.
  const stepsParam = first(params.steps)
  const requestedSteps = stepsParam
    ? stepsParam.split(',').map(s => s.trim()).filter(Boolean)
    : Array.from({ length: STEP_SLOTS }, (_, i) => first(params[`s${i + 1}`]))
        .map(s => (s ?? '').trim())
        .filter(Boolean)

  const [events, ages] = await Promise.all([
    availableEvents(start, end),
    watermarkAge(),
  ])
  const eventNames = events.map(e => e.event)
  const eventSet = new Set(eventNames)

  const validSteps = requestedSteps.filter(s => eventSet.has(s))
  const invalidSteps = requestedSteps.filter(s => !eventSet.has(s))
  const canRunFunnel = validSteps.length >= 2 && validSteps.length <= 6

  const current = { range, steps: requestedSteps, windowKey, by }

  const [funnel, engagement] = await Promise.all([
    canRunFunnel ? computeFunnel(start, end, validSteps, WINDOWS[windowKey], by) : Promise.resolve(null),
    featureEngagement(start, end),
  ])

  const maxUsers = Math.max(funnel?.steps[0]?.users ?? 1, 1)

  return (
    <main className="mx-auto w-full max-w-[1600px] px-8 py-8">
      <PageHeader
        title="Funnels"
        subtitle={
          <div className="flex flex-wrap items-center gap-3">
            <span>{formatDateRange(start, end)} · {NY_TZ}</span>
            <nav aria-label="Date range" className="flex items-center gap-1">
              {(Object.keys(RANGES) as RangeKey[]).map(key => (
                <a
                  key={key}
                  href={hrefWith(current, { range: key })}
                  aria-current={key === range ? 'true' : undefined}
                  className={`numeral rounded-sm px-2 py-0.5 text-xs outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent ${
                    key === range ? 'bg-raised font-medium text-ink-1' : 'text-ink-3 hover:text-ink-1'
                  }`}
                >
                  {RANGES[key].label}
                </a>
              ))}
            </nav>
          </div>
        }
        right={
          <div className="flex items-center gap-4">
            <StalenessBadge ages={ages} />
            <form action={async () => { 'use server'; await signOut({ redirectTo: '/login' }) }}>
              <button className="text-xs text-ink-2 outline-none transition-colors hover:text-ink-1 focus-visible:ring-2 focus-visible:ring-accent">
                {session?.user?.email} · sign out
              </button>
            </form>
          </div>
        }
      />

      <div key={`${range}-${requestedSteps.join(',')}-${windowKey}-${by ?? ''}`} className="animate-content-fade">
        <section aria-label="Presets" className="mt-8">
          <SectionHeading>Presets</SectionHeading>
          <div className="mt-3 flex flex-wrap gap-2">
            {PRESETS.map(preset => {
              const active = preset.steps.join(',') === validSteps.join(',') && requestedSteps.length === preset.steps.length
              return (
                <a
                  key={preset.label}
                  href={hrefWith(current, { steps: preset.steps })}
                  aria-current={active ? 'true' : undefined}
                  className={`rounded-sm border px-3 py-1.5 text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent ${
                    active
                      ? 'border-accent bg-surface text-ink-1'
                      : 'border-hairline text-ink-2 hover:bg-surface hover:text-ink-1'
                  }`}
                >
                  {preset.label}
                </a>
              )
            })}
          </div>
        </section>

        <section aria-label="Build a funnel" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Build a funnel</SectionHeading>
          <form method="GET" action="/funnels" className="mt-3 flex flex-wrap items-end gap-4">
            <input type="hidden" name="range" value={range} />
            {Array.from({ length: STEP_SLOTS }, (_, i) => {
              const slotValue = requestedSteps[i] ?? ''
              return (
                <label key={i} className="flex flex-col gap-1 text-xs text-ink-2">
                  Step {i + 1}
                  <select
                    name={`s${i + 1}`}
                    defaultValue={slotValue}
                    className="numeral rounded-sm border border-hairline bg-canvas px-2 py-1.5 text-sm text-ink-1 outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    <option value="">{'—'}</option>
                    {eventNames.map(name => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                </label>
              )
            })}
            <label className="flex flex-col gap-1 text-xs text-ink-2">
              Window
              <select
                name="window"
                defaultValue={windowKey}
                className="numeral rounded-sm border border-hairline bg-canvas px-2 py-1.5 text-sm text-ink-1 outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                {(Object.keys(WINDOWS) as WindowKey[]).map(key => (
                  <option key={key} value={key}>{key}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-2">
              Breakdown
              <select
                name="by"
                defaultValue={by ?? ''}
                className="numeral rounded-sm border border-hairline bg-canvas px-2 py-1.5 text-sm text-ink-1 outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <option value="">None</option>
                <option value="platform">Platform</option>
                <option value="app_version">App version</option>
              </select>
            </label>
            <button
              type="submit"
              className="rounded-sm border border-hairline bg-surface px-3 py-1.5 text-sm text-ink-1 outline-none transition-colors hover:border-accent focus-visible:ring-2 focus-visible:ring-accent"
            >
              Run funnel
            </button>
          </form>
        </section>

        <section aria-label="Funnel result" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>
            {requestedSteps.length > 0 ? requestedSteps.join(' → ') : 'Funnel'}
          </SectionHeading>

          {invalidSteps.length > 0 && (
            <p className="mt-3 text-sm text-alert">
              Unknown event{invalidSteps.length > 1 ? 's' : ''} dropped from this range: {invalidSteps.join(', ')}
            </p>
          )}

          {!canRunFunnel && (
            <p className="mt-3 text-sm text-ink-3">
              Pick a preset or select at least 2 steps below to compute a funnel.
            </p>
          )}

          {funnel && (
            <div className="mt-4 space-y-3">
              {funnel.steps.map((step, i) => {
                const width = Math.max((step.users / maxUsers) * 100, step.users > 0 ? 2 : 0)
                return (
                  <div key={`${step.event}-${i}`} className="flex items-center gap-3">
                    <div className="w-40 shrink-0 truncate text-sm text-ink-2" title={step.event}>
                      {i + 1}. {step.event}
                    </div>
                    <div className="relative h-6 min-w-0 flex-1 overflow-hidden rounded-sm bg-surface">
                      <div className="h-full rounded-sm bg-accent-bar" style={{ width: `${width}%` }} />
                    </div>
                    <div className="numeral w-16 shrink-0 text-right text-sm text-ink-1">
                      {compact(step.users)}
                    </div>
                    <div className="numeral w-32 shrink-0 text-right text-xs text-ink-3">
                      {i === 0 ? '—' : `${pct(step.convFromPrev)} of prev`}
                    </div>
                    <div className="numeral w-24 shrink-0 text-right text-xs text-ink-3">
                      {pct(step.convFromStart)} of start
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {funnel?.breakdown && funnel.breakdown.length > 0 && (
            <div className="mt-6">
              <SectionHeading as="h3">
                Breakdown by {by === 'app_version' ? 'app version' : by}
              </SectionHeading>
              <div className="mt-3">
                <DataTable
                  rowKey={row => `${row.event}-${row.dimValue}`}
                  rows={funnel.breakdown}
                  columns={[
                    { key: 'event', header: 'Step', render: r => r.event },
                    { key: 'dim', header: by === 'app_version' ? 'App version' : 'Platform', render: r => r.dimValue },
                    { key: 'users', header: 'Users', align: 'right', render: r => compact(r.users) },
                  ]}
                />
              </div>
            </div>
          )}
        </section>

        <section aria-label="Feature engagement" className="mt-8 border-t border-hairline pt-8">
          <SectionHeading>Feature engagement</SectionHeading>
          <p className="mt-1 text-xs text-ink-3">Total events and distinct users per event type in this range.</p>
          <div className="mt-3">
            <BarList
              items={engagement.map(e => ({ label: e.event, value: e.total, sublabel: `${compact(e.users)}u` }))}
              formatValue={compact}
            />
          </div>
        </section>
      </div>
    </main>
  )
}
