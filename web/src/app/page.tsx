import { auth, signOut } from '@/auth'
import { dailyEventCounts, dailyActiveUsers } from '@/lib/metrics/queries'
import { watermarkAge, formatAge, isStale } from '@/lib/metrics/staleness'
import { NY_TZ } from '@/lib/time'

export const dynamic = 'force-dynamic'

function today(): string {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: NY_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date())
}

export default async function OverviewPage() {
  const session = await auth()
  const end = today()
  const startDate = new Date(`${end}T00:00:00Z`)
  startDate.setUTCDate(startDate.getUTCDate() - 6)
  const start = startDate.toISOString().slice(0, 10)

  const [counts, users, ages] = await Promise.all([
    dailyEventCounts(start, end),
    dailyActiveUsers(start, end),
    watermarkAge(),
  ])
  const stale = ages.filter(a => isStale(a.ageSeconds))

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Freeport Analytics</h1>
          <p className="mt-1 text-sm text-neutral-400">Last 7 days · {NY_TZ}</p>
        </div>
        <form action={async () => { 'use server'; await signOut({ redirectTo: '/login' }) }}>
          <button className="text-xs text-neutral-400 hover:text-neutral-200">
            {session?.user?.email} · sign out
          </button>
        </form>
      </header>

      {stale.length > 0 && (
        <div className="mt-6 rounded-md border border-amber-700/50 bg-amber-950/40 px-4 py-3 text-sm text-amber-200">
          Data is stale — {stale.map(s => `${s.source} ${formatAge(s.ageSeconds)} behind`).join(', ')}.
          The sync cron may not be running.
        </div>
      )}

      <section className="mt-8 grid gap-4 sm:grid-cols-2">
        <div className="rounded-lg border border-neutral-800 p-5">
          <div className="text-xs uppercase tracking-wide text-neutral-500">Events (7d)</div>
          <div className="mt-1 text-3xl font-semibold tabular-nums">
            {counts.reduce((a, r) => a + r.count, 0).toLocaleString()}
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 p-5">
          <div className="text-xs uppercase tracking-wide text-neutral-500">Peak DAU (7d)</div>
          <div className="mt-1 text-3xl font-semibold tabular-nums">
            {users.length ? Math.max(...users.map(u => u.users)).toLocaleString() : '0'}
          </div>
        </div>
      </section>

      <section className="mt-8 rounded-lg border border-neutral-800">
        <table className="w-full text-sm">
          <thead className="text-neutral-500">
            <tr className="border-b border-neutral-800">
              <th className="px-5 py-3 text-left font-medium">Day</th>
              <th className="px-5 py-3 text-right font-medium">Events</th>
              <th className="px-5 py-3 text-right font-medium">Active users</th>
            </tr>
          </thead>
          <tbody>
            {counts.map(row => (
              <tr key={row.day} className="border-b border-neutral-900 last:border-0">
                <td className="px-5 py-2.5 tabular-nums">{row.day}</td>
                <td className="px-5 py-2.5 text-right tabular-nums">{row.count.toLocaleString()}</td>
                <td className="px-5 py-2.5 text-right tabular-nums">
                  {(users.find(u => u.day === row.day)?.users ?? 0).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  )
}
