import type { CSSProperties } from 'react'

import type { VerificationPoint } from '../types'
import { ChartTip, TipRow } from './ChartTip'
import { linePath, type Pt } from './geometry'
import { useChartPointer } from './useChartPointer'

/**
 * What was forecast for each slot, against what actually happened.
 *
 * Neither existing chart fits, which is why this is a third one rather than a
 * prop on either. `TimeSeries` has exactly the right semantics -- real
 * timestamps, `null` lifts the pen, shaded gap bands, a prose summary -- and is
 * architecturally single-series. `Curves` is multi-series but its x axis is
 * "hours ahead" rather than clock time, its domain is welded to 0-1 for a
 * probability, and it bridges. So this is `TimeSeries`' null handling with
 * `ScoreByHorizon`'s two-`linePath` structure, both of them the shared
 * `geometry.linePath` that already knows a null means pen up.
 *
 * **Two different kinds of hole, and they mean opposite things.** A missing
 * `actual` is the trackers not seeing the house -- the same unobserved slot
 * every chart here already draws as a break. A missing `forecast` is the
 * add-on deliberately saying nothing, because the model did not earn that
 * horizon. Only the second gets a shaded band: it is the serving rule made
 * visible over time, and it is the thing this card was built to show.
 *
 * The dash on the second line is not decoration. Under `forced-colors` both
 * strokes collapse to CanvasText and the dash is the only thing left telling
 * them apart -- the same note `ScoreByHorizon` carries, and the reason the
 * legend keys are words rather than swatches alone.
 */

const W = 1000
const H = 160

/** Contiguous runs where nothing was published, as [start, end] inclusive. */
function unpublished(points: VerificationPoint[]): [number, number][] {
  const out: [number, number][] = []
  points.forEach((p, i) => {
    if (p.forecast !== null) return
    const last = out[out.length - 1]
    if (last && last[1] === i - 1) last[1] = i
    else out.push([i, i])
  })
  return out
}

export function Verification({ points, summary, startLabel, endLabel, horizon }: {
  points: VerificationPoint[]
  summary: string
  startLabel: string
  endLabel: string
  horizon: number
}) {
  // Before any early return: a hook may not be called conditionally.
  const pointer = useChartPointer(points.length)

  const n = points.length
  const X = (i: number) => (n === 1 ? W / 2 : (i / (n - 1)) * W)
  // Both series are probabilities, so the domain is fixed at 0-1 rather than
  // fitted. A fitted axis would rescale the whole card whenever the forecast
  // happened to be confident, and make two days incomparable side by side.
  const Y = (v: number) => H - v * H

  const at = (pick: (p: VerificationPoint) => number | null) =>
    points.map((p, i): Pt | null => {
      const v = pick(p)
      return v === null ? null : { x: X(i), y: Y(v) }
    })

  const forecast = at((p) => p.forecast)
  const actual = at((p) => p.actual)
  const holes = unpublished(points)
  const hover = points[pointer.index ?? -1]

  const pct = (v: number) => `${(100 * v).toFixed(0)}%`

  return (
    <>
      <div className="chart" style={{ height: H } as CSSProperties}
           ref={pointer.ref} {...pointer.handlers}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
             aria-label={summary}>
          <title>{summary}</title>

          {[0.25, 0.5, 0.75].map((g) => (
            <line key={g} className="grid" x1={0} x2={W} y1={H * g} y2={H * g}
                  vectorEffect="non-scaling-stroke" />
          ))}

          {/* Drawn first, so both lines sit on top of the shading. */}
          {holes.map(([a, b]) => {
            // Anchored on the POINTS, exactly as `TimeSeries` bands its
            // unobserved runs: the shading has to start where the line stops or
            // the two channels disagree about where the hole is.
            const x1 = n === 1 ? 0 : (a / (n - 1)) * W
            const x2 = n === 1 ? W : ((b + 1) / (n - 1)) * W
            return <rect key={a} className="gap" x={x1} y={0}
                         width={Math.max(x2 - x1, 1)} height={H} />
          })}

          <path className="line base" d={linePath(actual)}
                vectorEffect="non-scaling-stroke" />
          <path className="line model" d={linePath(forecast)}
                vectorEffect="non-scaling-stroke" />

          {pointer.index !== null && (
            <line className="crosshair" vectorEffect="non-scaling-stroke"
                  x1={X(pointer.index)} x2={X(pointer.index)} y1={0} y2={H} />
          )}
        </svg>
        <span className="chart-max num">100%</span>
        <span className="chart-min num">0%</span>

        {hover && (
          <ChartTip fraction={pointer.fraction}>
            <div className="tiphead">{new Date(hover.t).toLocaleString()}</div>
            <TipRow accent="aqua" label={`forecast +${horizon} h ahead`}
                    value={hover.forecast === null
                      ? 'not forecast' : pct(hover.forecast)} />
            <TipRow label="actually home"
                    value={hover.actual === null
                      ? 'not observed' : pct(hover.actual)} />
          </ChartTip>
        )}
      </div>

      <div className="scale">
        <span className="num">{startLabel}</span>
        <span className="legend" aria-hidden="true">
          <span className="key">
            <i style={{ '--c': 'var(--rgb-aqua)' } as CSSProperties} />
            <span>forecast</span>
          </span>
          <span className="key"><i className="dashkey" /><span>actually home</span></span>
          {holes.length > 0 && (
            <span className="key"><i className="gapkey" /><span>not forecast</span></span>
          )}
        </span>
        <span className="num">{endLabel}</span>
      </div>

      <p className="secondary chart-summary">{summary}</p>
    </>
  )
}
