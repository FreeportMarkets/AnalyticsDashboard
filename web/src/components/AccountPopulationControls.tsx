import { ACCOUNT_POPULATIONS, type AccountPopulation } from '@/lib/accountMetrics'

export function populationHref(params: Record<string, string | string[] | undefined>, population: AccountPopulation) {
  const qs = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    const item = Array.isArray(value) ? value[0] : value
    if (item !== undefined) qs.set(key, item)
  }
  qs.set('population', population)
  return `/journey?${qs}`
}

export function AccountPopulationControls({ params, population }: {
  params: Record<string, string | string[] | undefined>; population: string
}) {
  return <nav aria-label="Account population" className="space-y-2">
    <p className="text-xs font-medium text-ink-2">Account population</p>
    <div className="flex flex-wrap gap-1">
      {(Object.entries(ACCOUNT_POPULATIONS) as [AccountPopulation, string][]).map(([key, label]) =>
        <a key={key} href={populationHref(params, key)} aria-current={population === key ? 'true' : undefined}
          className={`inline-flex min-h-11 items-center rounded-md px-3 py-2 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${population === key ? 'bg-raised text-ink-1' : 'text-ink-2 hover:bg-surface'}`}>{label}</a>)}
    </div>
    <p className="max-w-3xl text-xs leading-relaxed text-ink-2">Mobile links reflect verified app use observed so far. Later links can move accounts between these groups. This filter applies to account cohorts; device intro diagnostics below keep their own population.</p>
  </nav>
}
