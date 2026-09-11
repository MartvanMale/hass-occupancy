import { useEffect, useState } from 'react'

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
