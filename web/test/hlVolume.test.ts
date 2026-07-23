import { describe, expect, it } from 'vitest'
import {
  HL_PAGE_CAP,
  computePerpVolume,
  fillNotional,
  hlFillsFetcher,
  isPerpFill,
  perpVolumeFromFills,
  type HlFill,
} from '../src/lib/hl/volume'

/**
 * Mirror of `test_hl_volume.py`, case for case. Both dashboards must compute
 * the same volume; these tests are what keeps the TS port and the Python
 * original from drifting. If you change one, change the other.
 */

const fill = (over: Partial<HlFill> = {}): HlFill => ({
  sz: '1', px: '100', dir: 'Open Long', tid: 1, time: 1000, ...over,
})

describe('fillNotional', () => {
  it('is size * price', () => {
    expect(fillNotional(fill({ sz: '2', px: '50' }))).toBe(100)
  })

  it('uses the absolute size (shorts report negative)', () => {
    expect(fillNotional(fill({ sz: '-2', px: '50' }))).toBe(100)
  })

  it('handles fractional sizes and prices', () => {
    expect(fillNotional(fill({ sz: '0.5', px: '2500.5' }))).toBeCloseTo(1250.25, 6)
  })
})

describe('isPerpFill', () => {
  it.each(['Open Long', 'Close Long', 'Open Short', 'Close Short'])('accepts %s', dir => {
    expect(isPerpFill(fill({ dir }))).toBe(true)
  })

  it.each(['Buy', 'Sell', 'Settlement'])('excludes %s', dir => {
    expect(isPerpFill(fill({ dir }))).toBe(false)
  })

  it('excludes a fill with no direction', () => {
    expect(isPerpFill({ sz: '1', px: '1' })).toBe(false)
  })
})

describe('perpVolumeFromFills', () => {
  it('sums opens and closes', () => {
    const v = perpVolumeFromFills([
      fill({ tid: 1, dir: 'Open Long', sz: '1', px: '100' }),
      fill({ tid: 2, dir: 'Close Long', sz: '1', px: '110' }),
    ])
    expect(v).toBe(210)
  })

  it('counts shorts', () => {
    const v = perpVolumeFromFills([
      fill({ tid: 1, dir: 'Open Short', sz: '-2', px: '100' }),
      fill({ tid: 2, dir: 'Close Short', sz: '2', px: '90' }),
    ])
    expect(v).toBe(380)
  })

  it('excludes spot and settlement', () => {
    const v = perpVolumeFromFills([
      fill({ tid: 1, dir: 'Open Long', sz: '1', px: '100' }),
      fill({ tid: 2, dir: 'Buy', sz: '1', px: '999' }),
      fill({ tid: 3, dir: 'Sell', sz: '1', px: '999' }),
      fill({ tid: 4, dir: 'Settlement', sz: '1', px: '999' }),
    ])
    expect(v).toBe(100)
  })

  it('dedups by tid', () => {
    const v = perpVolumeFromFills([
      fill({ tid: 7, sz: '1', px: '100' }),
      fill({ tid: 7, sz: '1', px: '100' }),
    ])
    expect(v).toBe(100)
  })

  it('is zero for no fills', () => {
    expect(perpVolumeFromFills([])).toBe(0)
  })
})

describe('computePerpVolume', () => {
  const fetcherFor = (byWallet: Record<string, HlFill[]>) =>
    async (w: string) => byWallet[w] ?? []

  it('sums across wallets', async () => {
    const v = await computePerpVolume(['a', 'b'], 0, 10_000, fetcherFor({
      a: [fill({ tid: 1, sz: '1', px: '100', time: 1 })],
      b: [fill({ tid: 2, sz: '1', px: '250', time: 2 })],
    }))
    expect(v).toBe(350)
  })

  it('keeps only fills within [startMs, endMs)', async () => {
    const v = await computePerpVolume(['a'], 1000, 2000, fetcherFor({
      a: [
        fill({ tid: 1, sz: '1', px: '100', time: 999 }),   // before
        fill({ tid: 2, sz: '1', px: '100', time: 1000 }),  // inclusive start
        fill({ tid: 3, sz: '1', px: '100', time: 1999 }),  // inside
        fill({ tid: 4, sz: '1', px: '100', time: 2000 }),  // exclusive end
      ],
    }))
    expect(v).toBe(200)
  })

  it('dedups the same fill seen via two wallets', async () => {
    const shared = fill({ tid: 42, sz: '1', px: '100', time: 5 })
    const v = await computePerpVolume(['a', 'b'], 0, 10_000, fetcherFor({
      a: [shared],
      b: [shared],
    }))
    expect(v).toBe(100)
  })

  it('is zero when there are no wallets', async () => {
    expect(await computePerpVolume([], 0, 1, async () => [])).toBe(0)
  })
})

describe('hlFillsFetcher pagination', () => {
  it('does not paginate when the batch is under the cap', async () => {
    let calls = 0
    const out = await hlFillsFetcher('w', 0, async () => {
      calls++
      return [fill({ tid: 1, time: 1 })]
    })
    expect(calls).toBe(1)
    expect(out).toHaveLength(1)
  })

  it('pages past the 2000-fill cap', async () => {
    const page1 = Array.from({ length: HL_PAGE_CAP }, (_, i) =>
      fill({ tid: i, time: 1000 + i })
    )
    const page2 = [fill({ tid: 999_999, time: 99_999 })]
    let calls = 0
    const out = await hlFillsFetcher('w', 0, async () => {
      calls++
      return calls === 1 ? page1 : page2
    })
    expect(calls).toBe(2)
    expect(out).toHaveLength(HL_PAGE_CAP + 1)
  })

  it('terminates on a degenerate same-timestamp boundary page', async () => {
    // Every fill shares a timestamp AND a tid set, so advancing startTime
    // yields nothing new. Must stop rather than loop forever.
    const page = Array.from({ length: HL_PAGE_CAP }, (_, i) =>
      fill({ tid: i, time: 5000 })
    )
    let calls = 0
    const out = await hlFillsFetcher('w', 0, async () => {
      calls++
      if (calls > 10) throw new Error('did not terminate')
      return page
    })
    expect(calls).toBe(2)
    expect(out).toHaveLength(HL_PAGE_CAP)
  })

  it('stops on an empty first page', async () => {
    let calls = 0
    const out = await hlFillsFetcher('w', 0, async () => {
      calls++
      return []
    })
    expect(calls).toBe(1)
    expect(out).toEqual([])
  })
})
