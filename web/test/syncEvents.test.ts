import { describe, it, expect, vi } from 'vitest'
import { syncEvents, EVENTS_SOURCE } from '@/lib/sync/syncEvents'

function item(overrides: Record<string, unknown> = {}) {
  return {
    date: '2026-07-20',
    sk: '2026-07-20T12:00:00.000Z#0xabc#aaaa1111',
    event: 'session_start',
    wallet_address: '0xabc',
    session_id: 's1',
    timestamp: '2026-07-20T12:00:00.000Z',
    ...overrides,
  }
}

function deps(items: unknown[]) {
  const inserted: unknown[][] = []
  const quarantined: Array<{ raw: unknown; reason: string }> = []
  return {
    inserted,
    quarantined,
    args: {
      now: new Date('2026-07-20T14:00:00.000Z'),
      readWatermark: vi.fn(async () => new Date('2026-07-20T00:00:00.000Z')),
      advanceWatermark: vi.fn(async () => {}),
      ensurePartitions: vi.fn(async () => []),
      fetchEvents: vi.fn(async () => items),
      insertEvents: vi.fn(async (rows: unknown[]) => { inserted.push(rows); return rows.length }),
      quarantine: vi.fn(async (raw: unknown, reason: string) => { quarantined.push({ raw, reason }) }),
    },
  }
}

describe('syncEvents', () => {
  it('inserts mapped rows and reports counts', async () => {
    const d = deps([item(), item({ sk: 'b', event: 'trade_success' })])
    const r = await syncEvents(d.args as never)
    // fetchEvents is called once per scanned date (previous + current UTC day)
    expect(r.scanned).toBe(4)
    expect(r.inserted).toBe(4)
    expect(r.quarantined).toBe(0)
  })

  it('quarantines bad rows without failing the batch', async () => {
    const d = deps([item(), { garbage: true }])
    const r = await syncEvents(d.args as never)
    expect(r.quarantined).toBe(2)
    expect(r.inserted).toBe(2)
    expect(d.quarantined[0]!.reason).toMatch(/missing/)
  })

  it('advances the watermark to now minus the lag, not to now', async () => {
    const d = deps([item()])
    await syncEvents(d.args as never)
    const arg = (d.args.advanceWatermark as ReturnType<typeof vi.fn>).mock.calls[0]![1] as Date
    expect(arg.toISOString()).toBe('2026-07-20T13:50:00.000Z')
  })

  it('ensures partitions before inserting', async () => {
    const d = deps([item()])
    await syncEvents(d.args as never)
    // Partitions come from the MAPPED ROW'S OWN `date` field ('2026-07-20',
    // item()'s default), not the scan window ['2026-07-19', '2026-07-20'].
    // This is the corrected contract: ensuring the scan window instead of the
    // row's actual date is exactly the bug (see the regression test below).
    expect(d.args.ensurePartitions).toHaveBeenCalledWith(['2026-07-20'])
    // Argument-only assertions pass even if a mutant reordered these calls
    // (e.g. ran ensurePartitions after the insert loop). Assert the real
    // invocation order via each mock's own call-order sequence number.
    const ensurePartitionsOrder = (d.args.ensurePartitions as ReturnType<typeof vi.fn>).mock
      .invocationCallOrder[0]!
    const insertEventsOrder = (d.args.insertEvents as ReturnType<typeof vi.fn>).mock
      .invocationCallOrder[0]!
    expect(ensurePartitionsOrder).toBeLessThan(insertEventsOrder)
  })

  it(
    'REGRESSION: ensures a partition for a row date far outside the scan window, ' +
      'not the scan window itself -- without the fix, a row dated 2031-03-15 would ' +
      'only get partitions ensured for the scan dates (e.g. 2026-07-19/20), landing ' +
      'in events_default and permanently blocking that month\'s partition',
    async () => {
      const d = deps([
        item({
          date: '2031-03-15',
          sk: '2031-03-15T12:00:00.000Z#0xabc#bbbb2222',
          timestamp: '2031-03-15T12:00:00.000Z',
        }),
      ])
      await syncEvents(d.args as never)
      expect(d.args.ensurePartitions).toHaveBeenCalledWith(['2031-03-15'])
      expect(d.args.ensurePartitions).not.toHaveBeenCalledWith(['2026-07-19', '2026-07-20'])
    }
  )

  it('advances the watermark for the events source', async () => {
    const d = deps([item()])
    await syncEvents(d.args as never)
    const arg = (d.args.advanceWatermark as ReturnType<typeof vi.fn>).mock.calls[0]![0]
    expect(arg).toBe(EVENTS_SOURCE)
  })

  it('does not advance the watermark when an insert throws', async () => {
    const d = deps([item()])
    d.args.insertEvents = vi.fn(async () => { throw new Error('neon down') })
    await expect(syncEvents(d.args as never)).rejects.toThrow('neon down')
    expect(d.args.advanceWatermark).not.toHaveBeenCalled()
  })

  it('is a no-op when there is nothing new', async () => {
    const d = deps([])
    const r = await syncEvents(d.args as never)
    expect(r.inserted).toBe(0)
    // The watermark still advances -- an empty window is a successfully synced window.
    expect(d.args.advanceWatermark).toHaveBeenCalled()
  })
})
