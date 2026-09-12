import { useState, type CSSProperties } from 'react'
import type { NightBand } from '../types'
import { ChartTip, TipRow } from './ChartTip'
import { clamp, linePath, runs } from './geometry'
import { useChartPointer } from './useChartPointer'

/**
 * Several 0-100% curves over the next 48 hours, in inline SVG. A sibling of
 * `TimeSeries`, not an option on it: three curves read against each other is a
 * different question from one series read off a line.
 * **A forecast has holes.** A horizon nothing serves is not in the curve; the
 * pen lifts rather than plotting `?? 0`, which reads as "certainly away".
 * `preserveAspectRatio="none"`, so nothing inside may have proportions.
 */

const W = 1000
const H = 220

export interface Curve {
  key: string
  label: string
  /** A palette token name, not a colour: light and dark are one stylesheet. */
  accent: string
  /** One entry per horizon, 0-1, index 0 being +1 h. `null` is a horizon no
   *  model serves -- a hole to draw around, never a zero to plot. */
  values: (number | null)[]
}

function pathFor(values: (number | null)[]): string {
  const n = values.length
  return linePath(values.map((v, i) => (v === null ? null : {
    x: n === 1 ? W / 2 : (i / (n - 1)) * W,
    y: H - clamp(v, 0, 1) * H,
  })))
}

/** Contiguous runs where EVERY drawn curve is null. `ships` is a property of the
 *  horizon, not the subject, so in practice they all hole at the same hour. */
function holes(curves: Curve[], n: number): [number, number][] {
  return runs(Array.from({ length: n }, (_, i) =>
    curves.length > 0 && curves.every((c) => c.values[i] == null)))
}

export function Curves({ curves, hours, night = [], label, at }: {
  curves: Curve[]
  /** Horizon of each sample, in hours ahead. Used for the axis and tooltips. */
  hours: number[]
  night?: NightBand[]
  label: string
  /** The slot the horizons are measured from. `observed_at`, never
   *  `predicted_at` -- that can be half an hour later. */
  at?: string | undefined
}) {
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  const pointer = useChartPointer(hours.length)

  if (curves.length === 0 || hours.length === 0) {
    return <p className="empty">No forecast yet.</p>
  }

  const lo = hours[0]!
  const hi = hours[hours.length - 1]!
  const span = hi - lo || 1
  const xOf = (h: number) => ((h - lo) / span) * W

  const shown = curves.filter((c) => !hidden[c.key])
  const gaps = holes(shown, hours.length)
  const holeHours = gaps.reduce((n, [a, b]) => n + (b - a + 1), 0)

  // Not the caller's sentence when some of it is switched off, and the holes go in it too.
  const base = shown.length === curves.length
    ? label
    : shown.length === 0
      ? 'Every curve is hidden. Use the legend to bring one back.'
      : `${label} Showing ${shown.map((c) => c.label).join(', ')} only.`
  const drawn = holeHours
    ? `${base} Nothing is forecast at ${holeHours} of the ${hours.length} horizons.`
    : base

  const hoursAhead = pointer.index === null ? null : hours[pointer.index]

  // Clock time `h` hours after the anchor, or '' when there is no anchor.
  // `anchor`, not `base` -- that name is already the aria-label sentence below.
  const anchor = at ? new Date(at) : null
  const clockAt = (h: number | null): string => {
    if (anchor === null || h === null || Number.isNaN(anchor.getTime())) return ''
    const when = new Date(anchor.getTime() + h * 3_600_000)
    return when.toLocaleString(undefined,
      { weekday: 'short', hour: '2-digit', minute: '2-digit' })
  }

  return (
    <>
      <div className="curves" ref={pointer.ref} {...pointer.handlers}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
             role="img" aria-label={drawn}>
          <title>{drawn}</title>
          {/* Night first, so every line draws over it. CLAMPED: a band opening
              at 0 on an axis starting at +1 h is a negative x, and this
              `<svg>` is `overflow: visible`, so it painted out over the card. */}
          {night.map((b, i) => {
            const from = clamp(b.from, lo, hi)
            const to = clamp(b.to, lo, hi)
            if (to <= from) return null
            return (
              <rect key={i} className="night" x={xOf(from)} y={0}
                    width={xOf(to) - xOf(from)} height={H} />
            )
          })}
          {[0.25, 0.5, 0.75].map((g) => (
            <line key={g} className="grid" x1={0} x2={W} y1={H * g} y2={H * g} />
          ))}
          {shown.map((c) => (
            <path key={c.key} d={pathFor(c.values)} className="curve"
                  style={{ '--c': `var(--rgb-${c.accent})` } as CSSProperties} />
          ))}
          {pointer.index !== null && (
            <line className="crosshair" vectorEffect="non-scaling-stroke"
                  x1={xOf(hours[pointer.index]!)} x2={xOf(hours[pointer.index]!)}
                  y1={0} y2={H} />
          )}
        </svg>
        <span className="curves-max">100%</span>
        <span className="curves-min">0%</span>

        {hoursAhead !== undefined && hoursAhead !== null && shown.length > 0 && (
          <ChartTip fraction={pointer.fraction}>
            {/* Hours ahead AND the clock time: "+31 h" is not a thing anybody
                can act on, "Fri 19:30" is. */}
            <div className="tiphead">
              {hoursAhead === 0 ? 'now' : `+${hoursAhead} h`}
              {clockAt(hoursAhead) && <span className="tipwhen">{clockAt(hoursAhead)}</span>}
            </div>
            {shown.map((c) => {
              const v = c.values[pointer.index!]
              return (
                <TipRow key={c.key} accent={c.accent} label={c.label}
                        value={v == null ? 'not forecast'
                          : `${(v * 100).toFixed(1)}%`} />
              )
            })}
          </ChartTip>
        )}
      </div>

      <div className="scale">
        <span>now</span>
        <span className="legend">
          {curves.map((c) => (
            <button
              key={c.key}
              type="button"
              className="key"
              aria-pressed={!hidden[c.key]}
              aria-label={`${hidden[c.key] ? 'Show' : 'Hide'} ${c.label}`}
              onClick={() => setHidden((h) => ({ ...h, [c.key]: !h[c.key] }))}
            >
              <i style={{ '--c': `var(--rgb-${c.accent})` } as CSSProperties} />
              <span>{c.label}</span>
            </button>
          ))}
          {/* A span, not a button: the curves can be switched off, this is a
              key. The legend is not `aria-hidden` because the buttons are focusable. */}
          {night.length > 0 && (
            <span className="key"><i className="nightkey" /><span>asleep</span></span>
          )}
        </span>
        <span>+{hours[hours.length - 1]} h</span>
      </div>

      {/* Real clock times under the axis. Five evenly spaced marks line up with
          `space-between` because the axis is linear in hourly horizons. */}
      {anchor !== null && !Number.isNaN(anchor.getTime()) && (
        <div className="ticks num" aria-hidden="true">
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <span key={f}>{clockAt(Math.round(lo + f * span))}</span>
          ))}
        </div>
      )}
    </>
  )
}
