/**
 * The one HTTP path to Hyperliquid's info endpoint, with the retry policy that
 * every HL call in this project depends on.
 *
 * HL's public endpoint rate-limits per IP. A single un-retried 429 dropping a
 * whole wallet from a sweep is how a full backfill once reconciled at 1.5%
 * instead of ~100% -- most wallets 429'd and silently vanished. So this
 * retries 429 and 5xx with exponential backoff (honoring Retry-After) and
 * network errors/timeouts too; only a genuine 4xx or an exhausted budget
 * throws. Callers must let it throw rather than swallow a partial result:
 * a short page is silent truncation, strictly worse than a clean
 * drop-and-retry-next-run.
 *
 * `fetchImpl` and `sleepImpl` are injectable so the retry policy itself is
 * unit-testable without real network or real time.
 */

const HL_INFO_URL = 'https://api.hyperliquid.xyz/info'

export interface HlPostOptions {
  maxRetries?: number
  timeoutMs?: number
  fetchImpl?: typeof fetch
  sleepImpl?: (ms: number) => Promise<void>
}

const defaultSleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms))

export async function hlInfoPost<T>(body: unknown, opts: HlPostOptions = {}): Promise<T> {
  const maxRetries = opts.maxRetries ?? 8
  const timeoutMs = opts.timeoutMs ?? 20_000
  const doFetch = opts.fetchImpl ?? fetch
  const sleep = opts.sleepImpl ?? defaultSleep

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    let res: Response
    try {
      res = await doFetch(HL_INFO_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
        cache: 'no-store',
      })
    } catch (err) {
      // Network error / timeout -- transient, retry unless budget exhausted.
      if (attempt === maxRetries - 1) throw err
      await sleep(backoffMs(attempt))
      continue
    }
    if (res.ok) return (await res.json()) as T
    if (res.status !== 429 && res.status < 500) {
      throw new Error(`HL ${res.status} ${res.statusText}`)
    }
    const retryAfter = Number(res.headers.get('retry-after'))
    await sleep(retryAfter > 0 ? retryAfter * 1000 : backoffMs(attempt))
  }
  throw new Error('HL info request failed after retries (rate limited)')
}

function backoffMs(attempt: number): number {
  return Math.min(30_000, 1_000 * 2 ** attempt)
}
