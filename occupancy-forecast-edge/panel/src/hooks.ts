import { useEffect, useRef, useState } from 'react'
import { openEvents } from './api'

/**
 * `value`, but only after it has held still for `ms`. For a control that fires
 * continuously and drives a fetch -- dragging the horizon slider end to end is
 * otherwise 48 reads of `metrics.json`, which is not cached.
 */
export function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setSettled(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return settled
}

/**
 * The time, re-read every `everyMs`, so relative times move between fetches.
 * Paused in a background tab and caught up as soon as it is shown again.
 */
export function useNow(everyMs: number): number {
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    let id: ReturnType<typeof setInterval> | undefined
    const start = () => {
      setNow(Date.now())
      id = setInterval(() => setNow(Date.now()), everyMs)
    }
    const onVisibility = () => {
      clearInterval(id)
      if (!document.hidden) start()
    }
    if (!document.hidden) start()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [everyMs])
  return now
}

/** No event for this long, pings included, and the stream counts as held back. */
const EVENTS_SILENT_MS = 45_000
const EVENTS_RETRY_MS = 30_000

/**
 * Calls back when the server says something changed, and reports whether the
 * stream is live. A proxy that buffers it drops `live`, so the caller's polls
 * take over. Retried here: `EventSource` gives up for good after a 401 or 404.
 */
export function useServerEvents(onStatus: () => void, onForecast: () => void): boolean {
  const [live, setLive] = useState(false)
  // Refs, so a re-render with new callbacks does not reopen the stream.
  const status = useRef(onStatus)
  const forecast = useRef(onForecast)
  status.current = onStatus
  forecast.current = onForecast

  useEffect(() => {
    let source: EventSource | null = null
    let silent: ReturnType<typeof setTimeout> | undefined
    let retry: ReturnType<typeof setTimeout> | undefined

    const drop = () => {
      if (!source) return
      source.close()
      source = null
      clearTimeout(silent)
      setLive(false)
      retry = setTimeout(open, EVENTS_RETRY_MS)
    }
    const heard = () => {
      clearTimeout(silent)
      silent = setTimeout(drop, EVENTS_SILENT_MS)
    }
    function open() {
      source = openEvents()
      heard()
      source.addEventListener('hello', () => {
        heard()
        // No status fetch here: `live` turning true restarts the caller's poll, which fetches.
        setLive(true)
        forecast.current()
      })
      source.addEventListener('status', () => { heard(); status.current() })
      source.addEventListener('forecast', () => { heard(); forecast.current() })
      source.addEventListener('ping', heard)
      source.onerror = drop
    }

    open()
    return () => {
      source?.close()
      clearTimeout(silent)
      clearTimeout(retry)
    }
  }, [])
  return live
}
