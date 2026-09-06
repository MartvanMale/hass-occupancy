import type { CSSProperties } from 'react'
import type { HorizonMetrics } from '../types'
import { bandPath, linePath, type Pt } from './geometry'
import { ChartTip, TipRow } from './ChartTip'
import { useChartPointer } from './useChartPointer'

/**
 * Every horizon's Brier against the baseline it had to beat, as two lines.
 *
 * The 48 rows of this were only ever a table, which answers "what did +24 h
 * score" well and "where do the two cross" not at all -- and where they cross is
 * the interesting fact, because it is where knowing the recent past stops being
 * worth more than knowing the household's habits.
 *
 * ZERO-BASED, deliberately. Brier is a squared error on a probability: zero
 * means every forecast was right, and the distance between two lines is a
 * quantity, not a rank. Cropping the axis to the data would turn a 0.005 gap
 * into a chasm, which is exactly the impression this chart must not give. The
 * fold ribbon supplies the spread that a cropped axis would otherwise be
 * smuggling in.
 *
 * `preserveAspectRatio="none"`, as everywhere else here, so nothing inside may
 * be a shape whose proportions matter -- no circles, no `<text>`. The one
 * exception a marker needs is a VERTICAL line, which is the single primitive
 * whose meaning survives an anisotropic transform; its label is HTML,
 * positioned over the box.
 */

const W = 1000
const H = 180

/** `explore.metrics_summary` skips a horizon with no metrics and writes null for
 *  a field missing from `metrics.json`, while the type says `number`. Trust the
 *  value, not the declaration. */
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

export function ScoreByHorizon({ horizons, current, onPick }: {
  horizons: HorizonMetrics[]
  current: number
  onPick: (h: number) => void
}) {
  const scored = horizons.filter((h) => finite(h.horizon_h))
  // Band mode: the marks are equal-width cells, so the cell under the pointer is
  // the one a click should take. Declared before the early return, as a hook
  // must be, which is why the count is guarded rather than derived below.
  const pointer = useChartPointer(
    scored.length === 0
      ? 1
      : Math.max(...scored.map((h) => h.horizon_h))
        - Math.min(...scored.map((h) => h.horizon_h)) + 1,
    'band',
  )

  if (scored.length === 0) {
    return <p className="empty">No horizon has been scored yet.</p>
  }

  // Keyed by horizon, never by index: a skipped horizon would otherwise shift
  // every later one one place to the left and the chart would be quietly wrong
  // rather than visibly incomplete.
  const byHour = new Map(scored.map((h) => [h.horizon_h, h]))
  const lo = Math.min(...byHour.keys())
  const hi = Math.max(...byHour.keys())
  const n = hi - lo + 1
  const hours = Array.from({ length: n }, (_, i) => lo + i)

  const values = scored.flatMap((h) => [h.brier, h.best_baseline_brier, h.brier_fold_max])
                       .filter(finite)
  // A little headroom, so the worst point is not welded to the top edge.
  const top = Math.max(...values, 0.01) * 1.08

  // Band centres, so a click lands in a band cleanly and the cells of the strip
  // below line up with the points above them.
  const X = (h: number) => ((h - lo) + 0.5) / n * W
  const Y = (v: number) => H - (v / top) * H
  const at = (h: number, pick: (m: HorizonMetrics) => number | null): Pt | null => {
    const m = byHour.get(h)
    if (!m) return null
    const v = pick(m)
    return finite(v) ? { x: X(h), y: Y(v) } : null
  }

  const model = hours.map((h) => at(h, (m) => m.brier))
  const base = hours.map((h) => at(h, (m) => m.best_baseline_brier))
  const foldLo = hours.map((h) => at(h, (m) => m.brier_fold_min))
  const foldHi = hours.map((h) => at(h, (m) => m.brier_fold_max))

  const ahead = scored.filter((h) => finite(h.brier) && finite(h.best_baseline_brier)
                                     && h.brier < h.best_baseline_brier).length
  const here = byHour.get(current) ?? null
  // The long form goes to `aria-label` and `<title>`; only `caption` is printed.
  // A sighted reader has the axes, the band and the legend and wants the two
  // numbers at the horizon they are on. A screen reader has none of that.
  const summary =
    `Brier by horizon, from +${lo} h to +${hi} h, against the best baseline at each. `
    + `Lower is better. The model is ahead at ${ahead} of ${scored.length} horizons`
    + (here && finite(here.brier) && finite(here.best_baseline_brier)
      ? `. At the horizon shown, +${current} h, it scores ${here.brier.toFixed(3)} against `
        + `${here.best_baseline.replace(/_/g, ' ')} at ${here.best_baseline_brier.toFixed(3)}`
      : '')
    + '. The shaded band is the spread across folds.'
  const caption =
    (here && finite(here.brier) && finite(here.best_baseline_brier)
      ? `At +${current} h: ${here.brier.toFixed(3)} against `
        + `${here.best_baseline.replace(/_/g, ' ')} at ${here.best_baseline_brier.toFixed(3)}. `
      : '')
    + `Ahead at ${ahead} of ${scored.length}.`

  const markerAt = ((current - lo) + 0.5) / n * 100

  const hovered = pointer.index === null ? null : byHour.get(lo + pointer.index) ?? null

  return (
    <>
      <div className="chart score" ref={pointer.ref} {...pointer.handlers}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
             aria-label={summary}>
          <title>{summary}</title>

          {[0.25, 0.5, 0.75].map((g) => (
            <line key={g} className="grid" x1={0} x2={W} y1={H * g} y2={H * g}
                  vectorEffect="non-scaling-stroke" />
          ))}

          <path className="ribbon" d={bandPath(foldLo, foldHi)} />

          {/* Dashed, and that is not decoration: under `forced-colors` both
              strokes collapse to CanvasText, and the dash is then the only thing
              telling the two series apart. */}
          <path className="line base" d={linePath(base)}
                vectorEffect="non-scaling-stroke" />
          <path className="line model" d={linePath(model)}
                vectorEffect="non-scaling-stroke" />

          <line className="marker" x1={X(current)} x2={X(current)} y1={0} y2={H}
                vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="chart-max num">{top.toFixed(2)}</span>
        <span className="chart-min num">0.00</span>
        <span className="marker-label num" style={{ left: `${markerAt}%` }}>
          +{current} h
        </span>

        {/* Click targets, laid over the chart rather than after it: a `<rect>`
            cannot be a button, and a focusable 48-stop widget inside the SVG
            would be a worse version of the range in step four. `aria-hidden`
            and `tabIndex={-1}` say the same thing -- this is a shortcut for a
            pointer, and the slider is the control. */}
        <div className="hitrow" aria-hidden="true">
          {hours.map((h) => (
            <button key={h} type="button" className="hit" tabIndex={-1}
                    onClick={() => onPick(h)} />
          ))}
        </div>

        {hovered && (
          <ChartTip fraction={pointer.fraction}>
            <div className="tiphead">horizon +{hovered.horizon_h} h</div>
            <TipRow accent="aqua" label="model"
                    value={finite(hovered.brier) ? hovered.brier.toFixed(3) : '—'} />
            <TipRow accent="orange" label={hovered.best_baseline.replace(/_/g, ' ')}
                    value={finite(hovered.best_baseline_brier)
                      ? hovered.best_baseline_brier.toFixed(3) : '—'} />
            <TipRow label="serves"
                    value={hovered.ships ? `model (${hovered.kind ?? '?'})`
                      : 'nothing — the baseline won'} />
          </ChartTip>
        )}
      </div>

      {/* Which horizons are published, per horizon. It belongs under the chart
          rather than as a third line in it: it is a category, and the y axis is
          a quantity. The dashed baseline LINE above stays -- it is still the
          bar the model had to clear, and this whole card is that comparison.
          Only the strip changes: a horizon the baseline won is not served by
          it, it is not served at all. */}
      <div className="strip" role="img"
           aria-label={`What is published, from +${lo} h to +${hi} h: `
             + `${scored.filter((h) => h.ships).length} horizons by a model, `
             + `${scored.filter((h) => !h.ships).length} not published.`}>
        {hours.map((h) => {
          const m = byHour.get(h)
          return (
            <i
              key={h}
              className={m?.ships ? undefined : 'off'}
              style={m?.ships ? { '--c': 'var(--rgb-aqua)' } as CSSProperties : undefined}
              title={m
                ? (m.ships
                    ? `+${h} h — the ${m.kind ?? 'model'} model serves`
                    : `+${h} h — nothing published; ${m.best_baseline.replace(/_/g, ' ')} beat the model`)
                : `+${h} h — not scored`}
            />
          )
        })}
      </div>

      <div className="scale">
        <span className="num">+{lo} h</span>
        <span className="legend" aria-hidden="true">
          <span className="key"><i style={{ '--c': 'var(--rgb-aqua)' } as CSSProperties} /><span>model</span></span>
          <span className="key"><i className="dashkey" /><span>best baseline</span></span>
          <span className="key"><i className="offkey" /><span>not published</span></span>
        </span>
        <span className="num">+{hi} h</span>
      </div>

      <p className="secondary chart-summary">{caption}</p>
    </>
  )
}
