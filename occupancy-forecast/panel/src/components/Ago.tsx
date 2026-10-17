import { useNow } from '../hooks'
import { relativeTime } from '../format'

/** "5 minutes ago" that keeps itself current. Its own component, so the tick
 *  re-renders this text and not the card around it. */
export function Ago({ iso }: { iso: string | null | undefined }) {
  const now = useNow(30_000)
  return <>{relativeTime(iso, now)}</>
}
