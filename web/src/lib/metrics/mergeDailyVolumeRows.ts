import type { DailyVolumeRow } from './trades'

/** Include fill-only days while retaining swap volume and trade-log counts. */
export function mergeDailyVolumeRows(
  tradeDays: DailyVolumeRow[],
  fillDays: Array<{ day: string; notionalUsd: number; fillCount: number }>,
  useFills: boolean,
): DailyVolumeRow[] {
  if (!useFills) return tradeDays
  const rows = new Map(tradeDays.map(row => [row.day, { ...row, perpsVolumeUsd: 0 }]))
  for (const fill of fillDays) {
    const row = rows.get(fill.day)
    rows.set(fill.day, {
      day: fill.day,
      swapVolumeUsd: row?.swapVolumeUsd ?? 0,
      perpsVolumeUsd: fill.notionalUsd,
      tradeCount: row?.tradeCount ?? 0,
    })
  }
  return [...rows.values()].sort((a, b) => a.day.localeCompare(b.day))
}
