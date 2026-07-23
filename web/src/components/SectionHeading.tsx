/**
 * The one section heading for the whole console.
 *
 * This replaced 28 copies of
 *   `text-xs font-semibold uppercase tracking-wide text-ink-2`
 * spread across six pages. That style was the problem, not just the
 * duplication: at 12px, uppercase, in secondary gray, a section heading
 * carried *less* visual weight than the table rows underneath it, so the
 * page had exactly two levels -- the KPI numerals and everything else --
 * and no scan path between them.
 *
 * Now: 16px, semibold, primary ink, sentence case. It reads as a heading
 * because it is one. `meta` carries the qualifier that used to be jammed
 * into the heading text itself ("Daily events · America/New_York", the
 * `est.` tag) as dimmer trailing text, so the heading stays a short noun
 * phrase you can find by shape.
 */
export function SectionHeading({
  children,
  meta,
  as: Tag = 'h2',
  right,
}: {
  children: React.ReactNode
  /** Dimmer qualifier trailing the title: timezone, units, an `est.` tag. */
  meta?: React.ReactNode
  as?: 'h2' | 'h3'
  /** Right-aligned slot for a section-level control or total. */
  right?: React.ReactNode
}) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
      <Tag className={`font-semibold text-ink-1 ${Tag === 'h2' ? 'text-base' : 'text-sm'}`}>
        {children}
        {meta && <span className="ml-2 text-xs font-normal text-ink-3">{meta}</span>}
      </Tag>
      {right && <div className="text-xs text-ink-3">{right}</div>}
    </div>
  )
}
