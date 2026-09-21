import { describe, expect, it, vi } from 'vitest'
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
import UsersPage from '@/app/users/page'
import FunnelsPage from '@/app/funnels/page'
import { activeUsers } from '@/lib/metrics/users'
import { availableEvents } from '@/lib/metrics/funnels'
import { fetchAccountMetrics } from '@/lib/accountMetricsApi'
function boundaries(node: ReactNode): ReactElement<{children: ReactElement}>[] {
  if (Array.isArray(node)) return node.flatMap(boundaries)
  if (!isValidElement<{children?:ReactNode}>(node)) return []
  return node.type === Suspense ? [node as ReactElement<{children:ReactElement}>] : boundaries(node.props.children)
}
describe('Account page source isolation', () => {
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
