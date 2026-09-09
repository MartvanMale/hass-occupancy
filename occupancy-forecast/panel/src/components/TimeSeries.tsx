import type { CSSProperties } from 'react'

import type { Accent } from './Icon'
import { ChartTip, TipRow } from './ChartTip'
import { runs as contiguous } from './geometry'
import { useChartPointer } from './useChartPointer'

/**
 * A line over time, in inline SVG.
 *
 * No charting library, for the reason `Icon.tsx` gives about `@mdi/js`: the
 * whole of one would arrive in the bundle to draw two shapes, and an Ingress
 * panel cannot assume the browser can reach a CDN to fetch it. Both scales here
 * are linear, so the entire chart is `Array.map` into a `<path d>`.
 *
 * `preserveAspectRatio="none"` is what makes it responsive without measuring
 * anything: the viewBox stretches to whatever width the card gives it. The cost
 * is that the transform is anisotropic, so nothing inside may be a shape whose
 * proportions matter -- no circles, no text. Labels live in the `.scale` row
 * below, as they do under the horizon strip, and `vector-effect:
 * non-scaling-stroke` keeps the line 2px however far the box is stretched.
 */

export interface Point {
  t: string
  v: number | null
}

const W = 1000
const H = 140

function pathFor(points: Point[], lo: number, hi: number): string {
  const span = hi - lo || 1
  const n = points.length
  const out: string[] = []
  let pen = false
  points.forEach((p, i) => {
    if (p.v === null) {
      // A break, not a bridge. `slot_fraction` returns NaN for a slot it did
      // not observe, and joining across it would draw a straight line through
      // hours nobody has any evidence about.
      pen = false
      return
    }
    const x = n === 1 ? W / 2 : (i / (n - 1)) * W
    const y = H - ((p.v - lo) / span) * H
    out.push(`${pen ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`)
    pen = true
  })
  return out.join(' ')
}

export interface TimeSeriesProps {
  points: Point[]
  /** The full sentence a screen reader gets. Never let the encoding rest on the
   *  pixels alone: this is the whole chart, for the one reader who cannot see
   *  it, so it says everything the marks say. */
  summary: string
  /** The line printed under the chart, when it should be shorter than the
   *  sentence above. A sighted reader already has the marks and the legend and
   *  wants the numbers; a screen reader has neither and needs the prose. They
   *  were one string, which meant trimming the visible one silently trimmed the
   *  accessible one. Defaults to `summary` where the two are the same. */
  caption?: string
  startLabel: string
  endLabel: string
  /** What the line is, for the legend swatch. */
  label: string
  accent?: Accent
  unit?: string | null
  /** Fixed for a fraction, so two charts of `home_frac` are comparable; left
   *  open for anything whose range is its own business. */
  domain?: [number, number] | undefined
  compact?: boolean
}

export function TimeSeries({
  points, summary, caption = summary, startLabel, endLabel, label,
  accent = 'aqua', unit = null, domain, compact = false,
}: TimeSeriesProps) {
  // Above the early return: a hook may not be called conditionally, and the
  // "nothing observed" branch below is a return.
  const pointer = useChartPointer(points.length)

  const values = points.map((p) => p.v).filter((v): v is number => v !== null)
  if (values.length === 0) {
    return <p className="empty">Nothing was observed in this window.</p>
  }

  const [lo, hi] = domain ?? [Math.min(...values, 0), Math.max(...values)]
  const span = hi - lo || 1
  // Contiguous runs of unobserved slots, as [startIndex, endIndex] inclusive.
  const runs = contiguous(points.map((p) => p.v === null))
  const n = points.length
  const height = compact ? 48 : H

  const fmt = (v: number) => `${v.toFixed(span >= 10 ? 0 : 2)}${unit ? ` ${unit}` : ''}`

  const hover = points[pointer.index ?? -1]

  return (
    <>
      <div
        className="chart"
        style={{ '--c': `var(--rgb-${accent})`, height } as CSSProperties}
        ref={pointer.ref}
        {...pointer.handlers}
      >
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
             aria-label={summary}>
          <title>{summary}</title>

          {/* Midline only. A full grid needs axis labels to mean anything, and
              those cannot live in a box with a non-uniform transform. */}
          <line className="grid" x1="0" y1={H / 2} x2={W} y2={H / 2} />

          {/* Unobserved stretches, drawn before the line so it sits on top.
              The band is the second channel: the break in the line says it, the
              shading says it again, and the summary says it in words. */}
          {runs.map(([a, b]) => {
            const x1 = n === 1 ? 0 : (a / (n - 1)) * W
            const x2 = n === 1 ? W : ((b + 1) / (n - 1)) * W
            return <rect key={a} className="gap" x={x1} y="0"
                         width={Math.max(x2 - x1, 1)} height={H} />
          })}

          <path className="line" d={pathFor(points, lo, hi)}
                vectorEffect="non-scaling-stroke" />

          {/* A vertical line is the one primitive whose meaning survives an
              anisotropic transform, which is why the crosshair is one and the
              readout beside it is HTML. */}
          {pointer.index !== null && (
            <line className="crosshair" vectorEffect="non-scaling-stroke"
                  x1={(pointer.index / Math.max(1, n - 1)) * W}
                  x2={(pointer.index / Math.max(1, n - 1)) * W}
                  y1={0} y2={H} />
          )}
        </svg>

        {hover && (
          <ChartTip fraction={pointer.fraction}>
            <div className="tiphead">{new Date(hover.t).toLocaleString()}</div>
            <TipRow accent={accent} label={label}
                    value={hover.v === null ? 'not observed' : fmt(hover.v)} />
          </ChartTip>
        )}
      </div>

      {/* Scale at the ends, legend in the middle -- the same three-part footer
          the horizon strip uses, and for the same reason: identity must never
          rest on colour alone. */}
      <div className="scale">
        <span className="num">{startLabel}</span>
        <span className="legend" aria-hidden="true">
          <i style={{ '--c': `var(--rgb-${accent})` } as CSSProperties} />
          <span>{label}</span>
          {runs.length > 0 && (
            <>
              <i className="gapkey" />
              <span>not observed</span>
            </>
          )}
        </span>
        <span className="num">{endLabel}</span>
      </div>

      <p className="secondary chart-summary">{caption}</p>
    </>
  )
}
