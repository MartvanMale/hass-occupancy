/** Turning the API's timestamps and seconds into something a person reads.
 *
 *  The page these replaced printed `2026-08-30T02:11:00+02:00` on screen. It is
 *  the right instant and the wrong unit: nobody wants a timestamp, they want to
 *  know whether it was recent.
 */

/** "13 hours ago", "just now", "in 6 hours". */
export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return 'never'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return 'never'

  const seconds = Math.round((then - Date.now()) / 1000)
  const magnitude = Math.abs(seconds)
  if (magnitude < 45) return 'just now'

  // Intl does the grammar and the plurals, in whatever locale the browser is
  // set to -- which is not something to hand-roll for the sake of "1 day ago".
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ['second', 60],
    ['minute', 60],
    ['hour', 24],
    ['day', 7],
    ['week', 4.35],
    ['month', 12],
    ['year', Infinity],
  ]
  const format = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
  let value = seconds
  for (const [unit, per] of units) {
    if (Math.abs(value) < per) return format.format(Math.round(value), unit)
    value /= per
  }
  return format.format(Math.round(value), 'year')
}

/** The absolute instant, for the secondary line, in the browser's locale. */
export function absoluteTime(iso: string | null | undefined): string {
  if (!iso) return ''
  const at = new Date(iso)
  if (Number.isNaN(at.getTime())) return ''
  return at.toLocaleString(undefined, {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
  })
}

/** "4m 12s", "52s", "1h 03m". */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return ''
  const whole = Math.max(0, Math.round(seconds))
  if (whole < 60) return `${whole}s`
  const minutes = Math.floor(whole / 60)
  if (minutes < 60) return `${minutes}m ${String(whole % 60).padStart(2, '0')}s`
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}

/** Thousands separators, in the browser's locale. */
export const count = (n: number) => n.toLocaleString()

/** "12.3 kB", "1.5 MB". */
export function bytes(n: number): string {
  if (n < 1024) return `${n} B`
  const units = ['kB', 'MB', 'GB']
  let value = n / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1 }
  return `${value.toFixed(1)} ${units[i]}`
}

/** `alice` -> `Alice`. The slug is derived from the person's entity id, so it
 *  is the closest thing to a name the add-on has without asking Home Assistant. */
export const pretty = (slug: string) => slug.charAt(0).toUpperCase() + slug.slice(1)

/** A 0-1 fraction as a percentage with a fixed number of decimals. */
export const percent = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`

/** A nullable 0-1 share: whole percents, two decimals when it is tiny but not
 *  zero (a 0.3% null fraction must not read as "0%"), "unknown" when null. */
export const share = (frac: number | null) =>
  frac === null ? 'unknown' : `${(frac * 100).toFixed(frac > 0 && frac < 0.01 ? 2 : 0)}%`

/** The "how far back" choices the archive and feature cards offer. */
export const DAY_OPTIONS = [
  { value: '1', label: 'last 24 hours' },
  { value: '7', label: 'last 7 days' },
  { value: '30', label: 'last 30 days' },
  { value: '90', label: 'last 90 days' },
]

/** The same list without 90 days: the forecast table is pruned to
 *  `config.FORECAST_RETENTION_DAYS` (30), so the verification card cannot
 *  honestly offer more. Derived from the list above so the two cannot drift. */
export const DAY_OPTIONS_RECENT = DAY_OPTIONS.filter((o) => Number(o.value) <= 30)

/** `last_error` is stored as `"<iso>: <message>"`. Split it so the message
 *  leads and the instant is said the way every other time on this page is.  */
export function splitError(
  text: string | null | undefined,
): { when: string | null; message: string } | null {
  if (!text) return null
  const match = /^(\d{4}-\d{2}-\d{2}T[\d:.+\-Z]+): ([\s\S]*)$/.exec(text)
  if (!match) return { when: null, message: text }
  return { when: match[1] ?? null, message: match[2] ?? text }
}
