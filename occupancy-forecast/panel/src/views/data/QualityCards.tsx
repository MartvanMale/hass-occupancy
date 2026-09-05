import type { HorizonMetrics, MetricsDetail, MetricsSummary } from '../../types'
import { Chip } from '../../components/Chip'
import { Row } from '../../components/Row'
import { FoldBars } from '../../components/FoldBars'
import { Reliability } from '../../components/Reliability'
import { ScoreByHorizon } from '../../components/ScoreByHorizon'
import { relativeTime } from '../../format'

/**
 * How well the models actually score.
 *
 * Every number here has been written to `/data/models/metrics.json` on every
 * train since the beginning and none of it has ever been rendered: the panel
 * could say that a horizon ships, but not by how much, not how much the folds
 * disagreed, and not whether the probability it states is the probability that
 * happens. Nothing new is computed to draw any of it.
 */

const pct = (n: number) => `${n >= 0 ? '+' : ''}${n.toFixed(1)}%`

/**
 * How the shipping horizons divide between the two families.
 *
 * Written as a sentence rather than a chip because the interesting thing is not
 * the counts but that there IS a split -- one family wins the near horizons and
 * the other the far ones, and where they cross is measured on this household's
 * own history rather than chosen. Empty string when nothing ships or when only
 * one family produced a candidate, so the card reads normally on day one.
 */
function familySplit(horizons: HorizonMetrics[]): string {
  const dedicated = horizons.filter((h) => h.kind === 'dedicated').length
  const pooled = horizons.filter((h) => h.kind === 'pooled').length
  if (dedicated === 0 || pooled === 0) return ''
  return ` ${dedicated} fitted per horizon, ${pooled} pooled.`
}

export function QualityCard({ metrics, current, onPick }: {
  metrics: MetricsSummary | null
  /** The horizon the rest of the step is showing, marked on the chart. */
  current: number
  onPick: (h: number) => void
}) {
  if (!metrics) return <p className="empty">Loading…</p>
  if (!metrics.available) return <p className="empty">{metrics.reason}</p>

  const failed = Object.keys(metrics.failed)
  return (
    <>
      <Row
        icon="model"
        accent={metrics.shipping > 0 ? 'aqua' : 'grey'}
        primary={`${metrics.shipping} of ${metrics.horizons.length} horizons ship a model`}
        secondary={`Scored ${metrics.evaluation ?? 'by rolling origin'}, trained ${
          relativeTime(metrics.trained_at)}.${familySplit(metrics.horizons)}`}
        trailing={
          <Chip label={`${metrics.shipping}/${metrics.horizons.length}`}
                icon="model" accent={metrics.shipping > 0 ? 'aqua' : 'grey'} />
        }
      />
      {failed.length > 0 && (
        <Row
          icon="alert"
          accent="red"
          primary={`${failed.length} ${failed.length === 1 ? 'horizon' : 'horizons'} failed to train`}
          secondary={failed.map((h) => `+${h} h: ${metrics.failed[h]}`).join(' · ')}
        />
      )}

      <ScoreByHorizon horizons={metrics.horizons} current={current} onPick={onPick} />

      {/* The 48 rows are still here, because six of the columns appear nowhere
          else -- but folded away, because the chart above answers the question
          people arrive with and the table answers the one they arrive with
          second. Lower Brier is better, so it is read as: how far below the
          baseline column did the model get. */}
      <details className="more">
        <summary>All {metrics.horizons.length} horizons, as numbers</summary>
        <div className="scroller">
        <table className="metrics">
          <thead>
            <tr>
              <th>horizon</th><th>Brier</th><th>baseline</th>
              <th>skill</th><th>folds won</th><th>serves</th><th>other family</th>
            </tr>
          </thead>
          <tbody>
            {metrics.horizons.map((h) => (
              <tr key={h.horizon_h} className={h.ships ? '' : 'off'}>
                <td className="num">+{h.horizon_h} h</td>
                <td className="num">{h.brier.toFixed(3)}</td>
                <td className="num" title={h.best_baseline}>
                  {h.best_baseline_brier.toFixed(3)}
                </td>
                <td className="num">{pct(h.skill_vs_best_baseline_pct)}</td>
                <td className="num">
                  {h.folds_beating_best_baseline}/{h.n_folds}
                </td>
                {/* The baseline that won is two columns left; repeating its
                    name here would say it serves, and nothing does. */}
                <td title={h.ships ? undefined : h.best_baseline}>
                  {h.ships ? `model (${h.kind ?? '?'})` : '—'}
                </td>
                <td className="num" title={h.rival_kind ?? undefined}>
                  {h.rival_brier === null ? '—' : h.rival_brier.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      </details>
      {/* This one stays, shortened. It is a table legend, not a caption: seven
          columns of unlabelled decimals are unreadable without it, and the
          table is behind a <details> so it is only on screen when asked for.
          What the two model families ARE, and why where they cross matters, is
          in DOCS.md under "Which horizons publish, and why some do not". */}
      <p className="secondary chart-summary">
        Brier is a squared error on a probability — lower is better, and the baseline
        column is the number to beat. Skill is the percentage below it; the last column
        is whichever model family lost here.
      </p>
    </>
  )
}

/** The losing family's number, when there was one. Both families are fitted at
 *  every horizon, so saying only the winner's Brier hides the whole comparison
 *  the gate actually made. */
function rivalClause(detail: { rival_kind: string | null; rival_brier: number | null }): string {
  if (detail.rival_kind === null || detail.rival_brier === null) return ''
  return ` The ${detail.rival_kind} fit scored ${detail.rival_brier.toFixed(3)} here.`
}

export function HorizonQualityCard({ detail }: { detail: MetricsDetail | null }) {
  if (!detail) return <p className="empty">Loading…</p>
  if (!detail.available) return <p className="empty">{detail.reason}</p>

  const ladder = Object.entries(detail.baselines)
    .filter((entry): entry is [string, number] => typeof entry[1] === 'number')
    .sort((a, b) => a[1] - b[1])

  return (
    <>
      <Row
        icon={detail.ships ? 'model' : 'minus'}
        accent={detail.ships ? 'aqua' : 'grey'}
        primary={detail.ships
          ? `The ${detail.kind ?? ''} model serves +${detail.horizon_h} h`.replace('  ', ' ')
          : `Nothing is published for +${detail.horizon_h} h`}
        secondary={detail.ships
          ? `Brier ${detail.brier.toFixed(3)} against ${detail.best_baseline} at ${
              detail.best_baseline_brier.toFixed(3)} — ${pct(detail.skill_vs_best_baseline_pct)},
              winning ${detail.folds_beating_best_baseline} of ${detail.n_folds} folds
              (sign test p=${detail.sign_test_p.toFixed(3)}).${rivalClause(detail)}`
          : `It scored ${detail.brier.toFixed(3)} and ${detail.best_baseline} scored ${
              detail.best_baseline_brier.toFixed(3)}, so nothing is published for it.${
              rivalClause(detail)}`}
        trailing={
          <Chip label={detail.ships ? 'model' : 'not served'}
                icon={detail.ships ? 'model' : 'minus'}
                accent={detail.ships ? 'aqua' : 'grey'} />
        }
      />

      <div className="cols">
        <div>
          <p className="subhead">Week by week</p>
          <FoldBars folds={detail.per_fold} baseline={detail.best_baseline_brier} />
        </div>
        <div>
          <p className="subhead">Does it mean what it says?</p>
          <Reliability bins={detail.reliability} />
        </div>
      </div>

      <p className="subhead">The whole baseline ladder</p>
      <div className="scroller">
        <table className="metrics">
          <thead><tr><th>baseline</th><th>Brier</th></tr></thead>
          <tbody>
            {ladder.map(([name, brier]) => (
              <tr key={name} className={name === detail.best_baseline ? '' : 'off'}>
                <td>{name.replace(/_/g, ' ')}</td>
                <td className="num">{brier.toFixed(3)}</td>
              </tr>
            ))}
            <tr>
              <td>the model</td>
              <td className="num">{detail.brier.toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  )
}
