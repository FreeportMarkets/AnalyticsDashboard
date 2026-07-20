import { labelForWallet, shortWallet, type PrivyWalletMap } from '@/lib/privy'

/**
 * Trader identity cell: human label as primary text, truncated wallet
 * address as dim secondary text -- so an operator can recognize the person
 * *and* still copy/identify the address. Falls back to just the truncated
 * address (no duplicate secondary line) when Privy has no identity for the
 * wallet.
 */
export function TraderCell({ wallet, privyMap }: { wallet: string; privyMap: PrivyWalletMap }) {
  const label = labelForWallet(wallet, privyMap)
  const truncated = shortWallet(wallet)
  const hasIdentity = label !== truncated

  return (
    <span title={wallet}>
      <span className="block truncate text-ink-1">{label}</span>
      {hasIdentity && <span className="numeral block text-[10px] text-ink-3">{truncated}</span>}
    </span>
  )
}
