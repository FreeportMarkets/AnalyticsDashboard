import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

/**
 * The Deposits page read the right event names with the right query against a
 * table those events no longer reach (Neon `events` stopped receiving mobile
 * events on 2026-08-08), and answered 0 for weeks without looking broken.
 *
 * These tests pin the two things that decide whether that can recur: the wire
 * contract with the backend, and the rule that a FAILURE IS NEVER A ZERO.
 */

const BODY = {
  trace_id: 'trace-abc',
  range: { from: '2026-08-16', to: '2026-09-14' },
  totals: { initiated: 1666, success: 34, error: 170, conversion: 0.0204 },
  by_provider: [
    { provider: 'crossmint', initiated: 1331, success: 20, error: 55, conversion: 0.015 },
    { provider: 'applepay', initiated: 214, success: 14, error: 42, conversion: 0.065 },
  ],
}

function jsonRes(status: number, body: unknown = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  vi.resetModules()
  process.env.ANALYTICS_FUNNEL_READ_SECRET = 's3cret'
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  delete process.env.ANALYTICS_FUNNEL_READ_SECRET
  delete process.env.TRADING_API_BASE_URL
})

describe('fetchDeposits — request', () => {
  it('sends the range and the shared funnel secret', async () => {
    const { fetchDeposits } = await import('@/lib/depositsApi')
    fetchMock.mockResolvedValue(jsonRes(200, BODY))

    await fetchDeposits({ from: '2026-08-16', to: '2026-09-14' })

    const call = fetchMock.mock.calls[0]!
    const [url, init] = call as [string, RequestInit]
    const parsed = new URL(url)
    expect(parsed.pathname).toBe('/v1/analytics/deposits')
    expect(parsed.searchParams.get('from')).toBe('2026-08-16')
    expect(parsed.searchParams.get('to')).toBe('2026-09-14')
    // Shared with the funnel read on purpose: one caller, one class of data,
    // one credential to rotate.
    expect(init.headers).toMatchObject({ 'x-funnel-secret': 's3cret' })
  })

  it('omits absent bounds rather than sending empty ones', async () => {
    const { fetchDeposits } = await import('@/lib/depositsApi')
    fetchMock.mockResolvedValue(jsonRes(200, BODY))

    await fetchDeposits({ days: 7 })

    const parsed = new URL(fetchMock.mock.calls[0]![0] as string)
    expect(parsed.searchParams.get('days')).toBe('7')
    // `from=` with an empty value is a 400 from the backend, which validates
    // rather than coercing. Absent must mean absent.
    expect(parsed.searchParams.has('from')).toBe(false)
    expect(parsed.searchParams.has('to')).toBe(false)
  })
})

describe('fetchDeposits — classification', () => {
  it('fails closed when the secret is not configured, without calling out', async () => {
    delete process.env.ANALYTICS_FUNNEL_READ_SECRET
    const { fetchDeposits } = await import('@/lib/depositsApi')

    const res = await fetchDeposits({ days: 30 })

    expect(res).toEqual({ ok: false, reason: 'not_configured' })
    // An unset secret must not produce an unauthenticated request whose 401 is
    // then reported as a backend fault.
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('distinguishes a bad credential from a bad range from an outage', async () => {
    const { fetchDeposits } = await import('@/lib/depositsApi')

    fetchMock.mockResolvedValueOnce(jsonRes(401))
    expect(await fetchDeposits({ days: 30 })).toEqual({ ok: false, reason: 'unauthorized' })

    // 400 is the backend REJECTING an impossible range rather than widening it,
    // so it is a bad question, not a broken backend. Blaming the backend for a
    // typo in the address bar sends the next person debugging the wrong thing.
    fetchMock.mockResolvedValueOnce(jsonRes(400))
    expect(await fetchDeposits({ from: '2026-02-30' })).toEqual({ ok: false, reason: 'bad_range' })

    fetchMock.mockResolvedValueOnce(jsonRes(503))
    expect(await fetchDeposits({ days: 30 })).toEqual({ ok: false, reason: 'unreachable' })
  })

  it('treats a thrown fetch (timeout, DNS, abort) as unreachable, never as zero', async () => {
    const { fetchDeposits } = await import('@/lib/depositsApi')
    fetchMock.mockRejectedValue(new Error('The operation was aborted'))

    expect(await fetchDeposits({ days: 30 })).toEqual({ ok: false, reason: 'unreachable' })
  })

  it("carries the backend's trace id out of a failure", async () => {
    const { fetchDeposits } = await import('@/lib/depositsApi')
    fetchMock.mockResolvedValue(jsonRes(401, { error: 'unauthorized', trace_id: 'trace-xyz' }))

    const res = await fetchDeposits({ days: 30 })

    // A loud failure with nothing to grep by is only half an improvement over
    // a silent zero.
    expect(res).toEqual({ ok: false, reason: 'unauthorized', traceId: 'trace-xyz' })
  })

  it('survives an error body that is not JSON', async () => {
    const { fetchDeposits } = await import('@/lib/depositsApi')
    fetchMock.mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new SyntaxError('Unexpected token < in JSON')
      },
    } as unknown as Response)

    // A gateway's own HTML 502 must still classify, not crash the page with a
    // parse error that hides what actually happened.
    expect(await fetchDeposits({ days: 30 })).toEqual({
      ok: false,
      reason: 'unreachable',
      traceId: undefined,
    })
  })

  it('returns the payload on success', async () => {
    const { fetchDeposits } = await import('@/lib/depositsApi')
    fetchMock.mockResolvedValue(jsonRes(200, BODY))

    const res = await fetchDeposits({ days: 30 })

    expect(res.ok).toBe(true)
    if (res.ok) expect(res.data.totals.initiated).toBe(1666)
  })
})

describe('depositSummary — mapping and fail-loud', () => {
  // trades.ts pulls in @/lib/db, which constructs a Neon client at import
  // time and validates the URL's shape. Nothing on this path issues a query —
  // depositSummary no longer touches Neon at all, which is the point of the
  // change — so this only has to parse.
  beforeEach(() => {
    process.env.DATABASE_URL =
      process.env.DATABASE_URL ?? 'postgresql://unused:unused@unused.example.com/unused'
  })

  it('maps backend totals and the provider split onto DepositSummary', async () => {
    fetchMock.mockResolvedValue(jsonRes(200, BODY))
    const { depositSummary } = await import('@/lib/metrics/trades')

    const summary = await depositSummary('2026-08-16', '2026-09-14')

    expect(summary.initiated).toBe(1666)
    expect(summary.success).toBe(34)
    expect(summary.error).toBe(170)
    // Conversion comes from the backend so the page and the API can never
    // disagree about what "conversion" divides by.
    expect(summary.conversionRate).toBe(0.0204)
    expect(summary.byProvider).toEqual([
      { provider: 'crossmint', initiated: 1331, success: 20, error: 55 },
      { provider: 'applepay', initiated: 214, success: 14, error: 42 },
    ])
  })

  it('THROWS instead of rendering zeros when the backend cannot answer', async () => {
    // This is the whole point. "Zero deposits" and "we could not ask" looked
    // identical on this page, which is exactly what let a dead data source hide
    // behind a plausible screen for weeks.
    fetchMock.mockResolvedValue(jsonRes(503))
    const { depositSummary } = await import('@/lib/metrics/trades')

    await expect(depositSummary('2026-08-16', '2026-09-14')).rejects.toThrow(/unreachable/)
  })

  it('puts the trace id in the thrown message', async () => {
    fetchMock.mockResolvedValue(jsonRes(401, { error: 'unauthorized', trace_id: 'trace-xyz' }))
    const { depositSummary } = await import('@/lib/metrics/trades')

    await expect(depositSummary('2026-08-16', '2026-09-14')).rejects.toThrow(/trace trace-xyz/)
  })

  it('names the reason in the error, so the screen is diagnosable', async () => {
    delete process.env.ANALYTICS_FUNNEL_READ_SECRET
    const { depositSummary } = await import('@/lib/metrics/trades')

    await expect(depositSummary('2026-08-16', '2026-09-14')).rejects.toThrow(/not_configured/)
  })
})
