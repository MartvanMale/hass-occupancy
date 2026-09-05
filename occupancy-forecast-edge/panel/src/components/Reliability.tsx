import type { CSSProperties } from 'react'
import type { ReliabilityBin } from '../types'

/**
 * The calibration curve, straight off `Metrics.reliability`.
 *
 * Calibration is the property this add-on actually needs. A model can rank
 * perfectly -- excellent AUC -- and still say 0.9 when it means 0.6, and for a
 * "pre-heat if they will be home" rule that is the difference between a warm
 * house and a wasted hour of gas. `evaluate.reliability` has been computing
 * these ten bins on every train since the beginning and nothing has ever drawn
 * them.
 *
 * A square box, so `preserveAspectRatio` is left at its default: unlike the
 * time series, the diagonal here only means "perfectly calibrated" if the axes
 * are on the same scale. That is also why the dots may be circles.
 */

const SIZE = 100
/** Beyond this the bin's stated probability is not what happened. Ten points is
 *  about where the heating decision would actually change. */
const TOLERANCE = 0.1

export function Reliability({ bins }: { bins: ReliabilityBin[] }) {
  if (bins.length === 0) {
    return <p className="empty">No calibration curve was recorded for this horizon.</p>
  }

  const worst = bins.reduce((a, b) =>
    Math.abs(b.observed - b.predicted) > Math.abs(a.observed - a.predicted) ? b : a)
  const gap = Math.abs(worst.observed - worst.predicted)
  const total = bins.reduce((n, b) => n + b.n, 0)

  // Long form for `aria-label`, short one for print. See TimeSeriesProps.caption
  // for why the two are not the same string.
  const summary =
    `Across ${bins.length} bins and ${total.toLocaleString()} scored slots, the widest ` +
    `gap between what the model said and what happened is at ` +
    `${worst.bin_low.toFixed(1)}–${worst.bin_high.toFixed(1)}, where it said ` +
    `${worst.predicted.toFixed(2)} and ${worst.observed.toFixed(2)} actually happened` +
    `${gap < TOLERANCE ? ' — within ten points everywhere, which is calibrated enough to act on.'
                       : `, a gap of ${(gap * 100).toFixed(0)} points.`}`
  const caption =
    `Widest gap at ${worst.bin_low.toFixed(1)}–${worst.bin_high.toFixed(1)}: said ` +
    `${worst.predicted.toFixed(2)}, ${worst.observed.toFixed(2)} happened` +
    `${gap < TOLERANCE ? '. Within ten points everywhere.'
                       : ` — ${(gap * 100).toFixed(0)} points.`}`

  // sqrt, so a bin with 4,000 samples does not read forty times louder than one
  // with 100; area rather than radius is what the eye compares.
  const counts = bins.map((b) => b.n)
  const biggest = Math.max(...counts, 1)
  const radius = (n: number) => 2 + 5 * Math.sqrt(n / biggest)

  return (
    <>
      <div className="calib">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label={summary}>
          <title>{summary}</title>
          {/* Perfect calibration. Dashed, because it is a reference and not
              data -- the same reason a target line is never solid. */}
          <line className="ref" x1="0" y1={SIZE} x2={SIZE} y2="0" />
          {bins.map((b) => {
            const off = Math.abs(b.observed - b.predicted) >= TOLERANCE
            return (
              <circle
                key={b.bin_low}
                className="bin"
                style={{ '--c': `var(--rgb-${off ? 'orange' : 'aqua'})` } as CSSProperties}
                cx={b.predicted * SIZE}
                cy={SIZE - b.observed * SIZE}
                r={radius(b.n)}
              >
                <title>
                  {`${b.bin_low.toFixed(1)}–${b.bin_high.toFixed(1)}: said ${
                    b.predicted.toFixed(2)}, observed ${b.observed.toFixed(2)}, ${
                    b.n.toLocaleString()} slots`}
                </title>
              </circle>
            )
          })}
        </svg>
      </div>

      <div className="scale">
        <span>says 0</span>
        <span className="legend" aria-hidden="true">
          <i style={{ '--c': 'var(--rgb-aqua)' } as CSSProperties} />
          <span>within 10 pts</span>
          <i style={{ '--c': 'var(--rgb-orange)' } as CSSProperties} />
          <span>off</span>
        </span>
        <span>says 1</span>
      </div>

      {/* The numbers in text as well as in position. A dot near the diagonal is
          a claim, and this is where a reader checks it. */}
      <table className="bins">
        <thead>
          <tr><th>bin</th><th>said</th><th>happened</th><th>slots</th></tr>
        </thead>
        <tbody>
          {bins.map((b) => (
            <tr key={b.bin_low}
                className={Math.abs(b.observed - b.predicted) >= TOLERANCE ? 'off' : ''}>
              <td>{b.bin_low.toFixed(1)}–{b.bin_high.toFixed(1)}</td>
              <td className="num">{b.predicted.toFixed(2)}</td>
              <td className="num">{b.observed.toFixed(2)}</td>
              <td className="num">{b.n.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="secondary chart-summary">{caption}</p>
    </>
  )
}
