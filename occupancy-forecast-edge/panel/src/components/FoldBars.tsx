import type { CSSProperties } from 'react'
import type { FoldScore } from '../types'

/**
 * Per-fold Brier, as bars. `Metrics.per_fold` has always been written and never
 * shown.
 *
 * It is the difference between two claims that look identical in a pooled
 * number: a model that beats its baseline in eleven weeks out of fifteen, and
 * one that beats it on the strength of a single lucky week. The ship gate
 * already tests for this -- `folds_beating_best_baseline` and a sign test -- and
 * this is what that gate is looking at.
 *
 * Built on the horizon strip's CSS rather than as a new chart: it is the same
 * shape, a row of cells read left to right, with a height that varies.
 * Lower is better, so a SHORT bar is a good week.
 */
export function FoldBars({ folds, baseline }: {
  folds: FoldScore[]
  /** The baseline this horizon had to beat. A fold above the line is a week the
   *  model lost. */
  baseline: number | null
}) {
  // A fold with no test rows for this horizon is padding, not a zero: the list
  // is emitted for every fold INDEX so that `ships` can walk it positionally
  // against the baseline ladder's. Those entries carry a null Brier, and they
  // are dropped here rather than drawn -- a bar of height zero would read as a
  // perfect week, which is the opposite of what an empty fold means.
  const scored = folds.filter((f): f is FoldScore & { brier: number } =>
    typeof f.brier === 'number')
  const empty = folds.length - scored.length

  if (scored.length === 0) {
    return <p className="empty">No per-fold scores were recorded for this horizon.</p>
  }

  const scores = scored.map((f) => f.brier)
  const worst = Math.max(...scores, baseline ?? 0)
  const beat = baseline === null ? null : scores.filter((s) => s < baseline).length

  // Long form for `aria-label`, short one for print. See TimeSeriesProps.caption
  // for why the two are not the same string.
  const summary =
    `${scored.length} folds, scored one week at a time. Brier runs from ` +
    `${Math.min(...scores).toFixed(3)} to ${Math.max(...scores).toFixed(3)}` +
    (baseline === null
      ? '. Lower is better.'
      : `, against a baseline of ${baseline.toFixed(3)}. The model was better in ${beat} of ` +
        `them — lower is better, so a shorter bar is a better week.`) +
    (empty ? ` ${empty} more had no rows to score at this horizon.` : '')
  const caption =
    `${scored.length} folds, Brier ${Math.min(...scores).toFixed(3)}–${
      Math.max(...scores).toFixed(3)}` +
    (baseline === null ? '.' : ` against ${baseline.toFixed(3)}. Better in ${beat}.`) +
    (empty ? ` ${empty} unscored.` : '')

  return (
    <>
      <div className="strip bars" role="img" aria-label={summary}>
        {scored.map((f, i) => {
          const lost = baseline !== null && f.brier >= baseline
          return (
            <i
              key={i}
              style={{
                '--c': `var(--rgb-${lost ? 'orange' : 'aqua'})`,
                height: `${Math.max(4, (f.brier / (worst || 1)) * 44)}px`,
              } as CSSProperties}
              title={`fold ${i + 1}: Brier ${f.brier.toFixed(3)} over ${
                f.n.toLocaleString()} slots${lost ? ' — the baseline won this week' : ''}`}
            />
          )
        })}
      </div>

      <div className="scale">
        <span>first week</span>
        <span className="legend" aria-hidden="true">
          <i style={{ '--c': 'var(--rgb-aqua)' } as CSSProperties} />
          <span>model won</span>
          <i style={{ '--c': 'var(--rgb-orange)' } as CSSProperties} />
          <span>baseline won</span>
        </span>
        <span>last week</span>
      </div>

      <p className="secondary chart-summary">{caption}</p>
    </>
  )
}
