import { afterEach, describe, expect, it, vi } from 'vitest'
import { isValidElement, Suspense, type ReactElement, type ReactNode } from 'react'
import { createRequire } from 'node:module'
const { renderToStaticMarkup } = createRequire(import.meta.url)('react-dom/server')
vi.mock('@/auth', () => ({ auth: async () => ({ user: { email: 'qa@example.test' } }), signOut: vi.fn() }))
vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
vi.mock('@/lib/metrics/users', () => ({ activeUsers: vi.fn(), sessionStats: vi.fn(), topUsersByActivity: vi.fn(), topTraders: vi.fn(), activityHeatmap: vi.fn() }))
vi.mock('@/lib/metrics/funnels', () => ({ availableEvents: vi.fn(), computeFunnel: vi.fn(), featureEngagement: vi.fn() }))
vi.mock('@/lib/metrics/staleness', () => ({ watermarkAge: async () => [], formatAge: vi.fn(), isStale: vi.fn() }))
vi.mock('@/components/AutoRefresh', () => ({ AutoRefresh: () => null }))
vi.mock('@/lib/accountMetricsApi', () => ({ fetchAccountMetrics: vi.fn() }))
vi.mock('@/lib/funnelApi', () => ({ fetchFunnel: vi.fn(async () => ({ ok: false, reason: 'unreachable' })), formatCohortRange: () => '', daysBehind: () => null }))
import UsersPage from '@/app/users/page'
import FunnelsPage from '@/app/funnels/page'
import JourneyPage from '@/app/journey/page'
import { activeUsers } from '@/lib/metrics/users'
import { availableEvents } from '@/lib/metrics/funnels'
import { fetchAccountMetrics } from '@/lib/accountMetricsApi'
import { fetchFunnel } from '@/lib/funnelApi'
afterEach(() => vi.useRealTimers())
function boundaries(node: ReactNode): ReactElement<{children: ReactElement}>[] {
  if (Array.isArray(node)) return node.flatMap(boundaries)
  if (!isValidElement<{children?:ReactNode}>(node)) return []
  return node.type === Suspense ? [node as ReactElement<{children:ReactElement}>] : boundaries(node.props.children)
}
describe('Account page source isolation', () => {
  it('routes independent NY/UTC dates and preserves account bounds through diagnostic controls', async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date('2026-09-22T01:00:00Z'))
    const params = { from: '2026-09-09', to: '2026-09-22', accountFrom: '2026-09-08', accountTo: '2026-09-21', population: 'mobile_linked' }
    const sections = boundaries(await JourneyPage({ searchParams: Promise.resolve(params) }))
    expect(sections[0]!.props.children.props).toMatchObject({ from: params.accountFrom, to: params.accountTo })
    const intro = sections[1]!.props.children
    const render = intro.type as (props: unknown) => Promise<ReactNode>
    const html = renderToStaticMarkup(await render(intro.props)).replaceAll('&amp;', '&')
    expect(fetchFunnel).toHaveBeenLastCalledWith(expect.objectContaining({ from: params.from, to: params.to }))
    const presetHref = html.match(/href="([^"]+)"[^>]*>14d<\/a>/)?.[1]
    expect(presetHref).toBeDefined()
    expect(Object.fromEntries(new URL(presetHref!, 'https://example.test').searchParams)).toMatchObject(params)
    const links = [...html.matchAll(/href="(\/journey\?[^"]+)"/g)].map(match => new URL(match[1]!, 'https://example.test').searchParams)
    expect(links.some(q => q.get('from') === '2026-09-09' && q.get('to') === '2026-09-22' && q.get('accountFrom') === '2026-09-08' && q.get('accountTo') === '2026-09-21')).toBe(true)
    expect(links.filter(q => q.get('window') === '24h')).toHaveLength(1)
    expect(links.find(q => q.get('window') === '24h')!.get('accountFrom')).toBe(params.accountFrom)
    const legacy = boundaries(await JourneyPage({ searchParams: Promise.resolve({ from: params.from, to: params.to }) }))
    expect(legacy[0]!.props.children.props).toEqual({ from: undefined, to: undefined, population: 'all' })
  })
  it.each([['Users', UsersPage, activeUsers], ['Funnels', FunnelsPage, availableEvents]] as const)('%s returns account and legacy source boundaries without awaiting either source', async (_name, Page, query) => {
    vi.clearAllMocks()
    const page = await Page({ searchParams: Promise.resolve({}) })
    const sections = boundaries(page)
    expect(sections).toHaveLength(2)
    expect(fetchAccountMetrics).not.toHaveBeenCalled()
    expect(query).not.toHaveBeenCalled()
    vi.mocked(query).mockRejectedValueOnce(new Error('legacy source offline'))
    const child = sections[1]!.props.children
    const render = child.type as (props: unknown) => Promise<ReactNode>
    const html = renderToStaticMarkup(await render(child.props))
    expect(html).toContain('unavailable')
    expect(html).toContain('independent source')
    expect(fetchAccountMetrics).not.toHaveBeenCalled()
  })
})
