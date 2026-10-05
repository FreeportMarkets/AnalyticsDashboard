'use client'

import { useRef, useState } from 'react'

/** Full, selectable address with a clipboard action and manual-copy fallback. */
export function WalletAddress({ wallet }: { wallet: string }) {
  const input = useRef<HTMLInputElement>(null)
  const [status, setStatus] = useState<'idle' | 'copied' | 'error'>('idle')

  async function copy() {
    try {
      await navigator.clipboard.writeText(wallet)
      setStatus('copied')
    } catch {
      input.current?.focus()
      input.current?.select()
      setStatus('error')
    }
  }

  return (
    <div className="min-w-[23rem]">
      <div className="flex items-center gap-2">
        <input
          ref={input}
          aria-label="Trader wallet address"
          readOnly
          value={wallet}
          size={Math.max(wallet.length, 1)}
          onFocus={event => event.currentTarget.select()}
          className="numeral min-w-0 flex-1 rounded-sm bg-transparent text-[11px] text-ink-2 outline-none focus:ring-1 focus:ring-accent"
        />
        <button
          type="button"
          onClick={copy}
          onBlur={() => setStatus('idle')}
          aria-label={`Copy wallet address ${wallet}`}
          disabled={!wallet}
          className="shrink-0 rounded border border-hairline px-2 py-1 text-xs text-ink-2 hover:bg-raised hover:text-ink-1 focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50"
        >
          {status === 'copied' ? 'Copied' : 'Copy'}
        </button>
      </div>
      <span role="status" className="text-xs text-ink-2">
        {status === 'error' ? 'Copy unavailable. Address selected; press Ctrl/Cmd+C.' : ''}
      </span>
    </div>
  )
}
