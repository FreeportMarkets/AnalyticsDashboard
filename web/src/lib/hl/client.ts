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
  /**
   * Absolute epoch-ms deadline. Past it, no new attempt or backoff sleep is
   * started and the call throws immediately. This is what keeps a serverless
   * cron from blowing its function timeout: without it, a wallet stuck in
   * rate-limit backoff (up to 8 retries × tens of seconds) runs for minutes,
   * and a between-wallet deadline check can't stop an in-flight fetch. With
   * it, every HL call returns or throws by the deadline.
   */
  deadlineMs?: number
}

const defaultSleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms))

export async function hlInfoPost<T>(body: unknown, opts: HlPostOptions = {}): Promise<T> {
  const maxRetries = opts.maxRetries ?? 8
  const timeoutMs = opts.timeoutMs ?? 20_000
  const doFetch = opts.fetchImpl ?? fetch
  const sleep = opts.sleepImpl ?? defaultSleep
  const deadlineMs = opts.deadlineMs ?? Infinity
  const remaining = () => deadlineMs - Date.now()

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    if (remaining() <= 0) throw new Error('HL info request deadline exceeded')
    let res: Response
    try {
      // Per-request timeout is the min of the configured timeout and the time
      // left until the deadline, so a single hung request can't overshoot.
      res = await doFetch(HL_INFO_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(Math.max(1, Math.min(timeoutMs, remaining()))),
        cache: 'no-store',
      })
    } catch (err) {
      if (attempt === maxRetries - 1 || remaining() <= 0) throw err
      const wait = backoffMs(attempt)
      if (wait >= remaining()) throw err // no point sleeping past the deadline
      await sleep(wait)
      continue
    }
    if (res.ok) return (await res.json()) as T
    if (res.status !== 429 && res.status < 500) {
      throw new Error(`HL ${res.status} ${res.statusText}`)
    }
    const retryAfter = Number(res.headers.get('retry-after'))
    const wait = retryAfter > 0 ? retryAfter * 1000 : backoffMs(attempt)
    if (wait >= remaining()) throw new Error('HL rate-limited, deadline exceeded')
    await sleep(wait)
  }
  throw new Error('HL info request failed after retries (rate limited)')
}

function backoffMs(attempt: number): number {
  return Math.min(30_000, 1_000 * 2 ** attempt)
}
