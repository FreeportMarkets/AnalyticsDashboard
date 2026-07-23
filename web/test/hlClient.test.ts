import { describe, expect, it, vi } from 'vitest'
import { hlInfoPost } from '../src/lib/hl/client'

/**
 * The retry policy is the single most reliability-critical path in the volume
 * system: an un-retried 429 dropping a wallet is what once made a full
 * backfill reconcile at 1.5% instead of ~100%. These tests pin that behavior.
 */

const noSleep = async () => {}
function jsonResponse(status: number, body: unknown = [], headers: Record<string, string> = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: String(status),
    headers: { get: (k: string) => headers[k.toLowerCase()] ?? null },
    json: async () => body,
  } as unknown as Response
}

describe('hlInfoPost retry policy', () => {
  it('returns the body on a first-try 200', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(200, [{ tid: 1 }]))
    const out = await hlInfoPost<{ tid: number }[]>({}, { fetchImpl, sleepImpl: noSleep })
    expect(out).toEqual([{ tid: 1 }])
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('retries a 429 and then succeeds (no wallet dropped)', async () => {
    let call = 0
    const fetchImpl = vi.fn(async () => (++call < 3 ? jsonResponse(429) : jsonResponse(200, [{ tid: 9 }])))
    const out = await hlInfoPost<{ tid: number }[]>({}, { fetchImpl, sleepImpl: noSleep })
    expect(out).toEqual([{ tid: 9 }])
    expect(fetchImpl).toHaveBeenCalledTimes(3)
  })

  it('retries a 5xx and then succeeds', async () => {
    let call = 0
    const fetchImpl = vi.fn(async () => (++call < 2 ? jsonResponse(502) : jsonResponse(200, [])))
    await hlInfoPost({}, { fetchImpl, sleepImpl: noSleep })
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('honors Retry-After (seconds) for the sleep duration', async () => {
    let call = 0
    const fetchImpl = vi.fn(async () =>
      ++call < 2 ? jsonResponse(429, [], { 'retry-after': '3' }) : jsonResponse(200, [])
    )
    const sleeps: number[] = []
    await hlInfoPost({}, { fetchImpl, sleepImpl: async ms => { sleeps.push(ms) } })
    expect(sleeps[0]).toBe(3000)
  })

  it('does NOT retry a genuine 4xx (e.g. 400) -- throws immediately', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(400))
    await expect(hlInfoPost({}, { fetchImpl, sleepImpl: noSleep })).rejects.toThrow(/HL 400/)
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('retries a network error/timeout', async () => {
    let call = 0
    const fetchImpl = vi.fn(async () => {
      if (++call < 2) throw new Error('ETIMEDOUT')
      return jsonResponse(200, [])
    })
    await hlInfoPost({}, { fetchImpl, sleepImpl: noSleep })
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('throws after exhausting the retry budget (never returns a partial)', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse(429))
    await expect(
      hlInfoPost({}, { fetchImpl, sleepImpl: noSleep, maxRetries: 4 })
    ).rejects.toThrow(/after retries/)
    expect(fetchImpl).toHaveBeenCalledTimes(4)
  })
})
