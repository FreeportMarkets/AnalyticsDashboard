import { describe, expect, it } from 'vitest'
import { builderFeeTotals } from '../src/lib/hl/builderFees'

describe('builderFeeTotals', () => {
  it('total = builderRewards (cumulative), NOT claimed + builderRewards', async () => {
    // The live-collector shape. builderRewards is the cumulative total; claimed
    // and unclaimed are its two halves (18818.22 + 2126.98 = 20945.20). Adding
    // claimed on top would double-count — the bug that made a complete backfill
    // read as 53%.
    const post = async () => ({
      builderRewards: '20945.20',
      tokenToState: [
        [0, {
          cumVlm: '90.75',
          claimedRewards: '18818.22',
          unclaimedRewards: '2126.98',
          builderRewards: '20945.20',
        }],
        [360, { claimedRewards: '113.47', builderRewards: '113.47' }],
      ],
    })
    const t = await builderFeeTotals('0xbuilder', post)
    expect(t.totalUsdc).toBeCloseTo(20945.20, 2)
    expect(t.claimedUsdc).toBeCloseTo(18818.22, 2)
    expect(t.pendingUsdc).toBeCloseTo(2126.98, 2)
    // The invariant that proves we read it right:
    expect(t.claimedUsdc + t.pendingUsdc).toBeCloseTo(t.totalUsdc, 2)
  })

  it('falls back to top-level builderRewards when tokenToState is absent', async () => {
    const post = async () => ({ builderRewards: '5000' })
    const t = await builderFeeTotals('0xbuilder', post)
    expect(t.pendingUsdc).toBe(5000)
    expect(t.claimedUsdc).toBe(0)
    expect(t.totalUsdc).toBe(5000)
  })

  it('returns zeros for an empty response', async () => {
    const t = await builderFeeTotals('0xbuilder', async () => ({}))
    expect(t.totalUsdc).toBe(0)
  })
})
