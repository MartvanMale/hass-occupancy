import { useEffect, useState } from 'react'

/**
 * `value`, but only after it has held still for `ms`.
 *
 * For a control that fires continuously and drives a fetch. Note what this is
 * NOT for: the effects that read it already guard with a `let live = true`
 * cleanup, so an out-of-order response was never the problem. The problem is
 * asking the server for every intermediate value -- dragging the horizon slider
 * end to end is 48 reads of `metrics.json`, which is not behind the explore
 * cache.
 */
export function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setSettled(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return settled
}
