import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createElement, isValidElement, Suspense, type ReactElement, type ReactNode } from 'react'
import { createRequire } from 'node:module'
const { renderToStaticMarkup } = createRequire(import.meta.url)('react-dom/server')

vi.mock('@/auth', () => ({ auth: async () => ({ user: { email: 'qa@example.test' } }), signOut: vi.fn() }))
vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
vi.mock('@/lib/metrics/trades', () => ({ volumeSummary: vi.fn(), dailyVolume: vi.fn(), topAssets: vi.fn(), venueSplit: vi.fn(), recentTrades: vi.fn(), recentlyFundedWallets: vi.fn(async () => new Set()), depositSummary: vi.fn() }))
vi.mock('@/lib/metrics/staleness', () => ({ watermarkAge: async () => [], formatAge: vi.fn(), isStale: vi.fn() }))
vi.mock('@/lib/metrics/hlVolumeRead', () => ({ hlVolumeTotal: async () => ({ notionalUsd: 125, builderFeeUsd: 1, fillCount: 2 }), hlVolumeDaily: async () => [] }))
vi.mock('@/lib/privyIdentities', () => ({ fetchWalletIdentities: vi.fn(async () => new Map()) }))
vi.mock('next/link', () => ({ default: 'a' }))
vi.mock('@/components/AutoRefresh', () => ({ AutoRefresh: () => null }))

import TradesPage from '@/app/trades/page'
import * as metrics from '@/lib/metrics/trades'
import { fetchWalletIdentities } from '@/lib/privyIdentities'
import { DepositsSection } from '@/components/DepositsSection'

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
})

describe('Trades dependency isolation', () => {
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
