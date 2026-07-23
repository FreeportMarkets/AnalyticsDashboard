export interface DataTableColumn<T> {
  key: string
  header: string
  align?: 'left' | 'right'
  render: (row: T) => React.ReactNode
}

/**
 * Dense operator table: right-aligned numerals, sticky header, hover row
 * reveal. Density over padding -- this is for scanning, not marketing.
 *
 * `maxHeight` is opt-in and defaults to none. It used to be an unconditional
 * `max-h-[28rem]`, which put a second scroll region inside the page scroll
 * on every table in the app -- a trackpad scroll that reaches the table's
 * bottom stops dead instead of continuing down the page. A table short
 * enough not to need it paid the cost anyway. Pass it only where a table is
 * genuinely unbounded (top-N-of-thousands), never for a 7-row daily detail.
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  maxHeight,
}: {
  columns: Array<DataTableColumn<T>>
  rows: T[]
  rowKey: (row: T) => string
  /** e.g. '28rem'. Omit for a table that scrolls with the page. */
  maxHeight?: string
}) {
  return (
    <div className={maxHeight ? 'overflow-auto' : undefined} style={maxHeight ? { maxHeight } : undefined}>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            {columns.map(col => (
              <th
                key={col.key}
                scope="col"
                className={`sticky top-0 z-10 border-b border-hairline bg-raised px-3 py-2 text-xs font-medium text-ink-2 ${
                  col.align === 'right' ? 'text-right' : 'text-left'
                }`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-3 py-6 text-center text-sm text-ink-3">
                No data in this range
              </td>
            </tr>
          ) : (
            rows.map(row => (
              <tr key={rowKey(row)} className="row-hover border-b border-hairline last:border-0">
                {columns.map(col => (
                  <td
                    key={col.key}
                    className={`px-3 py-2 text-ink-1 ${
                      col.align === 'right' ? 'numeral text-right' : ''
                    }`}
                  >
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}
