import { describe, expect, it, vi } from 'vitest'
vi.mock('@/lib/db', () => ({ sql: vi.fn() }))
import { sql } from '@/lib/db'
import { activeUsers, activityHeatmap, topUsersByActivity } from '@/lib/metrics/users'
import { availableEvents, computeFunnel, featureEngagement } from '@/lib/metrics/funnels'

describe('legacy web diagnostics retain an explicit source boundary', () => {
  it('filters activity and every custom funnel step to web instead of claiming mobile coverage', async () => {
    vi.mocked(sql).mockResolvedValue([] as never)
    await availableEvents('2026-09-15', '2026-09-21', 'web')
    await featureEngagement('2026-09-15', '2026-09-21', 'web')
    await activityHeatmap('2026-09-15', '2026-09-21', 'web')
    await topUsersByActivity('2026-09-15', '2026-09-21', 20, 'web')
    await computeFunnel('2026-09-15', '2026-09-21', ['start', 'trade'], 3600, undefined, 'web')
    const queries = vi.mocked(sql).mock.calls.map(call => String(call[0]))
    expect(queries).toHaveLength(5)
    for (const query of queries) expect(query).toContain("platform = 'web'")
    expect(queries[4]).toContain("e.platform = 'web'")
    expect(queries.join(' ')).not.toContain("platform <> 'server'")
  })
})
