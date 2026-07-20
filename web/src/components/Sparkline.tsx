/**
 * Pure inline SVG sparkline. No charting library, no axes -- just the shape
 * of a series of numbers, cyan stroke on the accent color.
 */
export function Sparkline({
  values,
  width = 80,
  height = 24,
  strokeWidth = 1.5,
  className,
}: {
  values: number[]
  width?: number
  height?: number
  strokeWidth?: number
  className?: string
}) {
  if (values.length === 0) return null

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const pad = strokeWidth
  const innerW = width - pad * 2
  const innerH = height - pad * 2

  const points = values.map((v, i) => {
    const x = values.length === 1 ? pad : pad + (i / (values.length - 1)) * innerW
    const y = pad + innerH - ((v - min) / range) * innerH
    return `${x.toFixed(2)},${y.toFixed(2)}`
  })

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      role="img"
      aria-label={`trend over ${values.length} points, from ${min} to ${max}`}
    >
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}
