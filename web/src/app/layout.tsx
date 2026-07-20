import type { Metadata } from 'next'
import './globals.css'
import { auth } from '@/auth'
import { Nav } from '@/components/Nav'

export const metadata: Metadata = {
  title: 'Freeport Analytics',
  description: 'Internal analytics dashboard',
  robots: { index: false, follow: false },
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth()
  const authenticated = Boolean(session?.user)

  return (
    <html lang="en">
      <body className="min-h-screen bg-canvas text-ink-1">
        {authenticated ? (
          <div className="flex min-h-screen">
            <Nav />
            <div className="min-w-0 flex-1">{children}</div>
          </div>
        ) : (
          children
        )}
      </body>
    </html>
  )
}
