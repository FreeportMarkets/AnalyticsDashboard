import { describe, expect, it } from 'vitest'
import { syncWalletVolume, syncWalletsBatch, type SyncDeps } from '../src/lib/volume/syncVolume'
import type { HlFill } from '../src/lib/hl/volume'

const BASE = Date.UTC(2026, 6, 15, 13, 0, 0)
const fill = (o: Partial<HlFill>): HlFill => ({
  sz: '1', px: '1000', dir: 'Open Long', tid: Math.random(), time: BASE, builderFee: '0.65', ...o,
})

/** A fake store that records what was written, so writes are assertable. */
function fakeDeps(fillsByWallet: Record<string, HlFill[]>, watermarks: Record<string, number> = {}) {
  const writes: Array<{ address: string; buckets: unknown[]; wm: number; additive: boolean }> = []
  const fetchCalls: Array<{ address: string; sinceMs: number }> = []
  const deps: SyncDeps = {
    readWatermark: async a => watermarks[a] ?? 0,
    fetchFills: async (a, sinceMs) => {
      fetchCalls.push({ address: a, sinceMs })
      return fillsByWallet[a] ?? []
    },
    writeBuckets: async (address, buckets, wm, additive) => {
      writes.push({ address, buckets, wm, additive })
    },
  }
  return { deps, writes, fetchCalls }
}

describe('syncWalletVolume', () => {
  it('backfill (additive=false) fetches from 0 and writes absolute buckets', async () => {
    const { deps, writes, fetchCalls } = fakeDeps({
      w: [fill({ tid: 1, sz: '2', px: '1000', time: BASE })],
    })
    const r = await syncWalletVolume('w', deps, { additive: false })
    expect(fetchCalls[0]!.sinceMs).toBe(0)
    expect(r.notionalUsd).toBe(2000)
    expect(r.daysTouched).toBe(1)
    expect(writes[0]!.additive).toBe(false)
    expect(writes[0]!.wm).toBe(BASE)
  })

  it('incremental (additive=true) fetches from the watermark and writes deltas', async () => {
    const { deps, writes, fetchCalls } = fakeDeps(
      { w: [fill({ tid: 2, sz: '1', px: '3000', time: BASE + 5_000 })] },
      { w: BASE }
    )
    const r = await syncWalletVolume('w', deps, { additive: true })
    expect(fetchCalls[0]!.sinceMs).toBe(BASE)
    expect(r.notionalUsd).toBe(3000)
    expect(writes[0]!.additive).toBe(true)
  })

  it('skips the write when there is nothing new', async () => {
    const { deps, writes } = fakeDeps({ w: [] }, { w: BASE })
    const r = await syncWalletVolume('w', deps, { additive: true })
    expect(writes).toHaveLength(0)
    expect(r.attributedFills).toBe(0)
    expect(r.watermarkMs).toBe(BASE)
  })

  it('advances the watermark past non-builder fills without writing volume', async () => {
    const { deps, writes } = fakeDeps({
      w: [fill({ tid: 3, sz: '9', px: '1000', builderFee: '0', time: BASE + 9_000 })],
    })
    const r = await syncWalletVolume('w', deps, { additive: false })
    // A write happens (watermark must advance) but with zero buckets.
    expect(writes).toHaveLength(1)
    expect(writes[0]!.buckets).toHaveLength(0)
    expect(writes[0]!.wm).toBe(BASE + 9_000)
    expect(r.notionalUsd).toBe(0)
  })

  it('captures a fetch error per-wallet instead of throwing', async () => {
    const deps: SyncDeps = {
      readWatermark: async () => 0,
      fetchFills: async () => { throw new Error('HL 429') },
      writeBuckets: async () => {},
    }
    const r = await syncWalletVolume('w', deps, { additive: false })
    expect(r.error).toBe('HL 429')
    expect(r.notionalUsd).toBe(0)
  })
})

describe('syncWalletsBatch', () => {
  it('processes every wallet and isolates a single failure', async () => {
    const deps: SyncDeps = {
      readWatermark: async () => 0,
      fetchFills: async a => {
        if (a === 'bad') throw new Error('boom')
        return [fill({ tid: a, sz: '1', px: '1000', time: BASE })]
      },
      writeBuckets: async () => {},
    }
    const { results, processed } = await syncWalletsBatch(['a', 'bad', 'b'], deps, {
      additive: false, concurrency: 2,
    })
    expect(processed).toBe(3)
    expect(results.find(r => r.address === 'bad')!.error).toBe('boom')
    expect(results.filter(r => r.notionalUsd === 1000)).toHaveLength(2)
  })

  it('stops at the deadline and reports the remainder for the next run', async () => {
    let clock = 0
    const deps: SyncDeps = {
      readWatermark: async () => 0,
      fetchFills: async () => { clock += 100; return [] }, // each fetch "takes" 100ms
      writeBuckets: async () => {},
    }
    const { processed, skippedForTime } = await syncWalletsBatch(
      Array.from({ length: 100 }, (_, i) => `w${i}`),
      deps,
      { additive: true, concurrency: 1, deadlineMs: 250, now: () => clock }
    )
    expect(processed).toBeLessThan(100)
    expect(processed + skippedForTime).toBe(100)
  })
})
