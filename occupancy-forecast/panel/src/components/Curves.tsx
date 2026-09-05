import { useState, type CSSProperties } from 'react'
import type { NightBand } from '../types'
import { ChartTip, TipRow } from './ChartTip'
import { clamp, linePath } from './geometry'
import { useChartPointer } from './useChartPointer'

/**
 * Several 0-100% curves over the next 48 hours, in inline SVG.
 *
 * A sibling of `TimeSeries` rather than an option on it: that one draws one
 * observed series and its whole hover layer is built around reading a single
 * value off a single line. This draws three forecast curves that share an axis
 * and are read against each other, which is a different question.
 *
 * **A forecast has holes.** It did not use to -- every horizon got an answer,
 * because a horizon the model had not earned was served its baseline instead.
 * Now nothing is published there, and this chart used to draw that hole as
 * `?? 0`: a hard 0%, which reads as "certainly away" rather than "no answer",
 * on the one part of the curve where the add-on has least to say. The pen
 * lifts instead, as it does in `TimeSeries` for a slot nobody observed.
 *
 * That hole used to carry a dashed outline as well. It was removed on request:
 * a dashed box in the corner of a chart reads as a rendering fault rather than
 * as meaning, and the absence is still said four other ways -- the pen lifts,
 * the hover tip reads "not forecast", the `aria-label` counts the missing
 * horizons, and the horizon strip on the same tab greys them. `holes()` stays
 * for that count; only the mark is gone.
 *
 * `preserveAspectRatio="none"`, as in `TimeSeries`: the viewBox stretches to
 * the card's width, so nothing inside may be a shape whose proportions matter.
 * Labels live in the rows outside the SVG.
 *
 * The legend is a row of buttons, because with three curves over one axis the
 * question is often about one of them -- "is that dip one person or the house" is
 * answered by turning the other two off. Which is why the legend can no longer
 * be `aria-hidden`: focusable content inside an `aria-hidden` subtree is an ARIA
 * violation, and every other legend in this panel still carries that attribute
 * precisely because none of them is focusable. The `aria-label` is regenerated
 * from what is actually drawn, so it never describes a hidden line.
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

/** Contiguous runs where EVERY drawn curve is null, as [start, end] indices.
 *
 *  Per-curve bands would be undrawable on a shared axis, and they would also
 *  be dishonest about what a hole is: `ships` is a property of the horizon,
 *  not of the subject, so in practice every curve holes at the same hour. A
 *  band is drawn only where nothing at all is forecast. */
function holes(curves: Curve[], n: number): [number, number][] {
  const out: [number, number][] = []
  for (let i = 0; i < n; i += 1) {
    if (curves.length === 0 || !curves.every((c) => c.values[i] == null)) continue
    const last = out[out.length - 1]
    if (last && last[1] === i - 1) last[1] = i
    else out.push([i, i])
  }
  return out
}

export function Curves({ curves, hours, night = [], label, at }: {
  curves: Curve[]
  /** Horizon of each sample, in hours ahead. Used for the axis and tooltips. */
  hours: number[]
  night?: NightBand[]
  label: string
  /** The slot the horizons are measured from, so the axis can carry real clock
   *  times. `observed_at`, never `predicted_at`: a horizon is h hours after the
   *  feature row's slot, and that slot can be half an hour older than the
   *  moment the arithmetic ran. Omitted, the chart falls back to hours only. */
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

  // Not the caller's sentence when some of it is switched off: an aria-label
  // naming three people over a chart drawing one is worse than no label. The
  // holes go in it too -- a screen reader must not have to infer an absence
  // from a shape it cannot see.
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
          {/* Night first, so every line draws over it.

              CLAMPED to the drawn range, which is not defensive tidying: the
              bands are offsets from NOW (`night.bands`), so a household asleep
              at this moment opens one at 0 while the axis starts at +1 h. That
              is a negative x, and this `<svg>` is `overflow: visible` -- which
              it has to be, or a 2px stroke sitting on the top or bottom edge
              is sliced down the middle -- so the rect was not clipped, it was
              painted out over the card. Clamp the band rather than drop the
              overflow rule: the band really does extend past the axis, and the
              axis is the thing that has to win. */}
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
            {/* Hours ahead AND the clock time. "+31 h" is not a thing anybody
                can act on; "Fri 19:30" is, and reading a dip off the chart is
                exactly the moment somebody wants it. */}
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
          {/* The two marks the chart draws that are not curves. They were
              unlabelled, which made the dashed box at the right read as a
              glitch rather than as the one thing it means: the add-on has
              nothing to say there. Spans and not buttons -- the curves can be
              switched off, these are keys. */}
          {night.length > 0 && (
            <span className="key"><i className="nightkey" /><span>asleep</span></span>
          )}
        </span>
        <span>+{hours[hours.length - 1]} h</span>
      </div>

      {/* Real clock times under the axis. "+31 h" is not a thing anybody can
          act on, and the whole point of a 48-hour chart is reading WHEN a dip
          falls. Five evenly spaced marks, which line up with `space-between`
          because the horizons are hourly and the axis is linear in them. */}
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
