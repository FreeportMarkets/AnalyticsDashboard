import { labelForWallet, shortWallet, type PrivyWalletMap } from '@/lib/privy'
import { WalletAddress } from './WalletAddress'

/**
 * Keep identity enrichment on the server; only the address copy control
 * needs client-side JavaScript. The address is always visible, even when
 * the cached Privy lookup has no label for this wallet.
 */
export function TraderCell({ wallet, privyMap }: { wallet: string; privyMap: PrivyWalletMap }) {
  const label = labelForWallet(wallet, privyMap)
  const truncated = shortWallet(wallet)
  const hasIdentity = label !== truncated

  return (
    <div>
      {hasIdentity && <span className="block text-ink-1">{label}</span>}
      <WalletAddress wallet={wallet} />
    </div>
  )
}
