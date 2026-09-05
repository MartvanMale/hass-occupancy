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
