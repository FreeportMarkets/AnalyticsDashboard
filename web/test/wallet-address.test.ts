// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { WalletAddress } from '@/components/WalletAddress'

const wallet = '0x1D4c5c27a2a85630674f59Bb8Da9B0d3517C81d9'
let container: HTMLDivElement
let root: Root

beforeEach(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
  await act(async () => root.render(createElement(WalletAddress, { wallet })))
})

afterEach(async () => {
  await act(async () => root.unmount())
  container.remove()
  window.getSelection()?.removeAllRanges()
  vi.unstubAllGlobals()
})

async function clickCopy() {
  await act(async () => container.querySelector('button')!.click())
}

describe('Wallet address clipboard control', () => {
  it('copies the exact full address and confirms success', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    await clickCopy()
    expect(writeText).toHaveBeenCalledWith(wallet)
    expect(container.querySelector('button')!.textContent).toBe('Copied')
    expect(container.querySelector('[role="status"]')!.textContent).toBe('Wallet address copied.')
  })

  it.each(['rejected', 'unavailable'])('selects the full address when the API is %s', async mode => {
    vi.stubGlobal('navigator', mode === 'rejected'
      ? { clipboard: { writeText: vi.fn().mockRejectedValue(new Error('Permission denied')) } }
      : {})
    await clickCopy()
    expect(window.getSelection()!.toString()).toBe(wallet)
    expect(document.activeElement).toBe(container.querySelector('[aria-label="Trader wallet address"]'))
    expect(container.querySelector('[role="status"]')!.textContent).toContain('Ctrl/Cmd+C')
    expect(container.querySelector('button')!.textContent).toBe('Copy')
  })
})
