import { describe, expect, it } from 'vitest'
import { builderFeeTotals } from '../src/lib/hl/builderFees'

describe('builderFeeTotals', () => {
  it('reads USDC (token 0) builder pending + claimed from tokenToState', async () => {
    const post = async () => ({
      builderRewards: '20945.19',
      tokenToState: [
        [0, { cumVlm: '90.75', claimedRewards: '18818.22', builderRewards: '20945.19' }],
        [360, { claimedRewards: '113.47', builderRewards: '113.47' }],
      ],
    })
    const t = await builderFeeTotals('0xbuilder', post)
    expect(t.pendingUsdc).toBeCloseTo(20945.19, 2)
    expect(t.claimedUsdc).toBeCloseTo(18818.22, 2)
    expect(t.totalUsdc).toBeCloseTo(39763.41, 2)
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
