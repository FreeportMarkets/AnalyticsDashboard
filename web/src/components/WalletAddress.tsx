'use client'

import { useRef, useState } from 'react'

/** Full, selectable address with a clipboard action and manual-copy fallback. */
export function WalletAddress({ wallet }: { wallet: string }) {
  const address = useRef<HTMLSpanElement>(null)
  const [status, setStatus] = useState<'idle' | 'copied' | 'error'>('idle')

  async function copy() {
    try {
      await navigator.clipboard.writeText(wallet)
      setStatus('copied')
    } catch {
      if (address.current) {
        address.current.focus()
        const range = document.createRange()
        range.selectNodeContents(address.current)
        const selection = window.getSelection()
        selection?.removeAllRanges()
        selection?.addRange(range)
      }
      setStatus('error')
    }
  }

  return (
    <div className="max-w-[18rem]">
      <div className="flex items-start gap-2">
        <span
          ref={address}
          aria-label="Trader wallet address"
          tabIndex={0}
          className="numeral min-w-0 flex-1 select-text break-all rounded-sm text-[11px] text-ink-2 outline-none focus:ring-1 focus:ring-accent"
        >{wallet}</span>
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
      <span role="status" className={status === 'copied' ? 'sr-only' : 'text-xs text-ink-2'}>
        {status === 'copied' ? 'Wallet address copied.' : status === 'error' ? 'Copy unavailable. Address selected; press Ctrl/Cmd+C.' : ''}
      </span>
    </div>
  )
}
