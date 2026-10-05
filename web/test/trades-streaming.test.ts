import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createElement, isValidElement, Suspense, type ReactElement, type ReactNode } from 'react'
import { createRequire } from 'node:module'
const { renderToStaticMarkup } = createRequire(import.meta.url)('react-dom/server')

vi.mock('@/auth', () => ({ auth: async () => ({ user: { email: 'qa@example.test' } }), signOut: vi.fn() }))
vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
vi.mock('@/lib/metrics/trades', () => ({ volumeSummary: vi.fn(), dailyVolume: vi.fn(), topAssets: vi.fn(), venueSplit: vi.fn(), recentTrades: vi.fn(), recentlyFundedWallets: vi.fn(async () => new Set()), depositSummary: vi.fn() }))
vi.mock('@/lib/metrics/staleness', () => ({ watermarkAge: async () => [], formatAge: vi.fn(), isStale: vi.fn() }))
vi.mock('@/lib/metrics/hlVolumeRead', () => ({ hlVolumeTotal: vi.fn(), hlVolumeDaily: vi.fn() }))
vi.mock('@/lib/privyIdentities', () => ({ fetchWalletIdentities: vi.fn(async () => new Map()) }))
vi.mock('next/link', () => ({ default: 'a' }))
vi.mock('@/components/AutoRefresh', () => ({ AutoRefresh: () => null }))

import TradesPage from '@/app/trades/page'
import * as metrics from '@/lib/metrics/trades'
import { hlVolumeDaily, hlVolumeTotal } from '@/lib/metrics/hlVolumeRead'
import { fetchWalletIdentities } from '@/lib/privyIdentities'
import { DepositsSection } from '@/components/DepositsSection'
import { TimeSeriesLine } from '@/components/TimeSeriesLine'

const volume = { totalVolumeUsd: 125, totalTrades: 2, uniqueTraders: 1, avgTradeSize: 62.5, byType: [], perpsByClient: [] }
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (reason: Error) => void; const promise = new Promise<T>((a, b) => { resolve = a; reject = b }); return { promise, resolve, reject } }
function find(node: ReactNode, type: unknown): ReactElement | undefined {
  if (Array.isArray(node)) return node.map(child => find(child, type)).find(Boolean)
  if (!isValidElement<{ children?: ReactNode }>(node)) return undefined
  return node.type === type ? node : find(node.props.children, type)
}
function linkTo(node: ReactNode, fragment: string): ReactElement<{ href: string }> | undefined {
  if (Array.isArray(node)) return node.map(child => linkTo(child, fragment)).find(Boolean)
  if (!isValidElement<{ href?: string; children?: ReactNode }>(node)) return undefined
  if (node.type === 'a' && node.props.href?.includes(fragment)) return node as ReactElement<{ href: string }>
  return linkTo(node.props.children, fragment)
}
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(metrics.volumeSummary).mockResolvedValue(volume)
  for (const fn of [metrics.dailyVolume, metrics.topAssets, metrics.venueSplit, metrics.recentTrades]) vi.mocked(fn).mockResolvedValue([])
  vi.mocked(hlVolumeTotal).mockResolvedValue({ notionalUsd: 125, builderFeeUsd: 1, fillCount: 2 })
  vi.mocked(hlVolumeDaily).mockResolvedValue([])
})

describe('Trades dependency isolation', () => {
  it('renders recorded fill sizes and wallets without estimated volume labels', async () => {
    vi.mocked(metrics.recentTrades).mockResolvedValue([
      { ts: '2026-10-05T17:20:48.587Z', timestamp: 'hyperliquid:875727468560255', type: 'perps',
        asset: 'BTC', side: 'long', size: 0.00053, price: 85291, leverage: null, client: 'unknown',
        venue: 'hyperliquid', status: 'filled', volumeUsd: 45.204230, isClose: true, action: 'Close', recordedFill: true,
        walletAddress: '0xf21aec8af0a4dae86a0fa579bc60effd43c1d087' },
      { ts: '2026-10-05T17:20:34.030Z', timestamp: 'hyperliquid:67933775089653', type: 'perps',
        asset: 'BTC', side: 'long', size: 0.00053, price: 85302, leverage: null, client: 'unknown',
        venue: 'hyperliquid', status: 'filled', volumeUsd: 45.210060, isClose: false, action: 'Open', recordedFill: true,
        walletAddress: '0xf21aec8af0a4dae86a0fa579bc60effd43c1d087' },
    ])
    const html = renderToStaticMarkup(await TradesPage({ searchParams: Promise.resolve({}) }))
    const recentSection = html.split('aria-label="Recent trades"')[1]!.split('</section>')[0]!
    expect(recentSection).toContain('0.00053')
    expect(recentSection).toContain('0xf21aec8af0a4dae86a0fa579bc60effd43c1d087')
    expect(recentSection).toContain('unknown')
    expect(recentSection).not.toContain('title="Perps stores')
  })

  it('keeps the page available with an explicit recent-fill read failure', async () => {
    vi.mocked(metrics.recentTrades).mockRejectedValue(new Error('contract mismatch'))
    const html = renderToStaticMarkup(await TradesPage({ searchParams: Promise.resolve({}) }))
    expect(html).toContain('Recent trades are unavailable')
    const recentSection = html.split('aria-label="Recent trades"')[1]!.split('</section>')[0]!
    expect(recentSection).not.toContain('No data in this range')
    expect(recentSection).not.toContain('Showing 0 trades')
  });
  it('renders a day present only in the backend fill ledger', async () => {
    vi.mocked(hlVolumeTotal).mockResolvedValue({ notionalUsd: 150, builderFeeUsd: 1, fillCount: 2, source: 'backend' })
    vi.mocked(hlVolumeDaily).mockResolvedValue([{ day: '2026-09-21', notionalUsd: 150, fillCount: 2 }])
    vi.mocked(metrics.dailyVolume).mockResolvedValue([{ day: '2026-09-22', swapVolumeUsd: 20, perpsVolumeUsd: 80, tradeCount: 3 }])
    const tree = await TradesPage({ searchParams: Promise.resolve({}) })
    const chart = find(tree, TimeSeriesLine) as ReactElement<{ data: Array<{ day: string; value: number }> }>
    expect(chart.props.data.map(({ day, value }) => ({ day, value }))).toEqual([
      { day: '2026-09-21', value: 150 },
      { day: '2026-09-22', value: 20 },
    ])
  })

  it('keeps estimated daily perps when the backend report is empty', async () => {
    vi.mocked(hlVolumeTotal).mockResolvedValue({ notionalUsd: 0, builderFeeUsd: 0, fillCount: 0, source: 'backend' })
    vi.mocked(metrics.dailyVolume).mockResolvedValue([{ day: '2026-09-22', swapVolumeUsd: 20, perpsVolumeUsd: 80, tradeCount: 3 }])
    const tree = await TradesPage({ searchParams: Promise.resolve({}) })
    const chart = find(tree, TimeSeriesLine) as ReactElement<{ data: Array<{ day: string; value: number }> }>
    expect(chart.props.data.map(({ day, value }) => ({ day, value }))).toEqual([
      { day: '2026-09-22', value: 100 },
    ])
  })

  it('returns trading content with an independent deposits Suspense boundary', async () => {
    const deposits = deferred<Awaited<ReturnType<typeof metrics.depositSummary>>>()
    vi.mocked(metrics.depositSummary).mockReturnValue(deposits.promise)
    const tree = await TradesPage({ searchParams: Promise.resolve({}) })
    expect(metrics.depositSummary).not.toHaveBeenCalled()
    expect(find(tree, Suspense)).toBeTruthy()
    const section = find(tree, DepositsSection) as ReactElement<{ start: string; end: string }>
    const pendingSection = DepositsSection(section.props)
    expect(metrics.depositSummary).toHaveBeenCalledOnce()
    // The page already resolved while this dependency is still pending.
    expect(tree.type).toBe('main')
    deposits.reject(new Error('slow backend unavailable'))
    const fallback = renderToStaticMarkup(await pendingSection)
    expect(fallback).toContain('Deposit events are unavailable')
    expect(fallback).not.toContain('Success events')
  })

  it('begins identities as soon as recent trades return, before unrelated aggregates', async () => {
    const slow = deferred<typeof volume>()
    vi.mocked(metrics.volumeSummary).mockReturnValue(slow.promise)
    const pending = TradesPage({ searchParams: Promise.resolve({}) })
    for (let i = 0; i < 8; i++) await Promise.resolve()
    expect(fetchWalletIdentities).toHaveBeenCalledOnce()
    slow.resolve(volume)
    await pending
  })

  it('offers the next 50 trades and returns to the last visible row', async () => {
    vi.mocked(metrics.recentTrades).mockResolvedValue(
      Array.from({ length: 51 }, (_, index) => ({ walletAddress: `wallet-${index}` })) as Awaited<ReturnType<typeof metrics.recentTrades>>
    )
    const tree = await TradesPage({ searchParams: Promise.resolve({ range: '30d' }) })
    expect(metrics.recentTrades).toHaveBeenCalledWith(expect.any(String), expect.any(String), 51)
    expect(fetchWalletIdentities).toHaveBeenCalledWith(expect.anything(), expect.arrayContaining(['wallet-0', 'wallet-49']))
    expect(linkTo(tree, 'trades=100')?.props.href).toBe('/trades?range=30d&trades=100#recent-trade-50')
  })

  it('bounds the requested trade count and loads one extra row to detect more', async () => {
    await TradesPage({ searchParams: Promise.resolve({ trades: '100' }) })
    expect(metrics.recentTrades).toHaveBeenCalledWith(expect.any(String), expect.any(String), 101)
    await TradesPage({ searchParams: Promise.resolve({ trades: '999999' }) })
    expect(metrics.recentTrades).toHaveBeenLastCalledWith(expect.any(String), expect.any(String), 1001)
  })

  it('renders successful deposit values as event diagnostics, never funded-account conversion', async () => {
    vi.mocked(metrics.depositSummary).mockResolvedValue({ initiated: 4, success: 6, error: 1, conversionRate: 1.5, byProvider: [{ provider: 'meld', initiated: 4, success: 6, error: 1 }] })
    const html = renderToStaticMarkup(await DepositsSection({ start: '2026-09-15', end: '2026-09-21' }))
    expect(html).toContain('not unique orders or funded accounts')
    expect(html).toContain('150.0%')
    expect(html).toContain('it is not a conversion rate')
  })
})
