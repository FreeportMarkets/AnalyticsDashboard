export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string
  subtitle?: React.ReactNode
  right?: React.ReactNode
}) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-4 border-b border-hairline/60 pb-5">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-ink-1">{title}</h1>
        {subtitle && <div className="mt-1 text-sm text-ink-2">{subtitle}</div>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </header>
  )
}
