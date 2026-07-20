export interface DataTableColumn<T> {
  key: string
  header: string
  align?: 'left' | 'right'
  render: (row: T) => React.ReactNode
}

/**
 * Dense operator table: right-aligned numerals, sticky header, hover row
 * reveal. Density over padding -- this is for scanning, not marketing.
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
}: {
  columns: Array<DataTableColumn<T>>
  rows: T[]
  rowKey: (row: T) => string
}) {
  return (
    <div className="max-h-[28rem] overflow-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            {columns.map(col => (
              <th
                key={col.key}
                scope="col"
                className={`sticky top-0 z-10 border-b border-hairline/60 bg-canvas px-3 py-2 text-xs font-medium uppercase tracking-wide text-ink-2 ${
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
              <tr key={rowKey(row)} className="row-hover border-b border-hairline/60 last:border-0">
                {columns.map(col => (
                  <td
                    key={col.key}
                    className={`px-3 py-1.5 text-ink-1 ${
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
