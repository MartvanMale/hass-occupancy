import type { CSSProperties } from 'react'
import type { FoldScore } from '../types'

/**
 * Per-fold Brier, as bars -- the difference between beating the baseline in
 * eleven weeks of fifteen and beating it on one lucky week, which a pooled
 * number hides. Built on the horizon strip's CSS: the same shape, with a
 * varying height. Lower is better, so a SHORT bar is a good week.
 */
export function FoldBars({ folds, baseline }: {
  folds: FoldScore[]
  /** The baseline this horizon had to beat. A fold above the line is a week the
   *  model lost. */
  baseline: number | null
}) {
  // A fold with no test rows is padding, not a zero -- the list is emitted for
  // every fold INDEX. A bar of height zero would read as a perfect week.
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
