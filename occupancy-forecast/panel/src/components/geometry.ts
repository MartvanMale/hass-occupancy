/**
 * The one bit of chart arithmetic worth sharing: a null point lifts the pen,
 * and the caller decides what a null is. All three charts must break rather
 * than bridge -- a joined line is confident through the region with no evidence.
 */
export interface Pt {
  x: number
  y: number
}

/** `M`/`L` through the points, restarting after every gap. */
export function linePath(points: (Pt | null)[]): string {
  const out: string[] = []
  let pen = false
  for (const p of points) {
    if (p === null) { pen = false; continue }
    out.push(`${pen ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    pen = true
  }
  return out.join(' ')
}

/** A filled band between two lines, one closed sub-path per unbroken run: one
 *  `M ... Z` would fill across a gap. */
export function bandPath(lo: (Pt | null)[], hi: (Pt | null)[]): string {
  const out: string[] = []
  let run: { lo: Pt; hi: Pt }[] = []
  const flush = () => {
    if (run.length < 2) { run = []; return }
    const top = run.map((r) => r.hi)
    const bottom = run.map((r) => r.lo).reverse()
    out.push(
      `${linePath(top)} ${bottom.map((p) => `L${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} Z`,
    )
    run = []
  }
  for (let i = 0; i < lo.length; i += 1) {
    const a = lo[i]
    const b = hi[i]
    if (a && b) run.push({ lo: a, hi: b })
    else flush()
  }
  flush()
  return out.join(' ')
}

/** Clamp, which every chart here needs and none of them should re-derive. */
export const clamp = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v))

/** Contiguous runs of `true`, as inclusive index pairs. The accumulator behind
 *  every gap band. */
export function runs(flags: boolean[]): [number, number][] {
  const out: [number, number][] = []
  flags.forEach((on, i) => {
    if (!on) return
    const last = out[out.length - 1]
    if (last && last[1] === i - 1) last[1] = i
    else out.push([i, i])
  })
  return out
}

/** Contiguous runs of consecutive integers in a sorted list, as [first, last]
 *  VALUES rather than indices -- so 1..9 is one range and not nine. */
export function integerRuns(nums: number[]): [number, number][] {
  const out: [number, number][] = []
  for (const n of nums) {
    const last = out[out.length - 1]
    if (last && n === last[1] + 1) last[1] = n
    else out.push([n, n])
  }
  return out
}
