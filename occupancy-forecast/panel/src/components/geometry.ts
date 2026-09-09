/**
 * The one bit of chart arithmetic worth sharing.
 *
 * `TimeSeries`, `Curves` and `ScoreByHorizon` all draw a line through points
 * some of which are not there, and all three have to break rather than bridge:
 * `slot_fraction` returns NaN for a slot nobody observed, and
 * `explore.metrics_summary` skips a horizon whose training produced no metrics
 * and nulls a field that is missing from `metrics.json`. Joining across either
 * one draws a straight, confident line through the exact region where there is
 * no evidence at all -- which is the failure this project has already fixed once
 * (see "Stop the Data tab blacking out on a fold with nothing to score").
 *
 * So: a null point lifts the pen, and the caller decides what a null is.
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

/**
 * A filled band between two lines, one closed sub-path per unbroken run.
 *
 * One `M ... Z` for the whole thing would close the shape straight across any
 * gap, filling a region the data says nothing about -- the same mistake as
 * bridging, in two dimensions instead of one.
 */
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

/**
 * Contiguous runs of `true`, as [start, end] index pairs, inclusive.
 *
 * The accumulator behind every gap band: `TimeSeries` shades unobserved slots,
 * `Verification` shades unpublished ones, `Curves` shades hours no model
 * forecast. Each had its own copy of this loop.
 */
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
