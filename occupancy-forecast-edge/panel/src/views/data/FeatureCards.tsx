import type {
  FeatureFamily, FeatureInventory, FeatureSeries, HorizonLag, HorizonRecipe,
} from '../../types'
import { Chip } from '../../components/Chip'
import { Row } from '../../components/Row'
import { TimeSeries, type Point } from '../../components/TimeSeries'
import { absoluteTime, bytes, count, relativeTime, share } from '../../format'

/**
 * The feature table, in the only three ways it can usefully be looked at -- not
 * as a table, at a thousand-odd columns wide. Every count on screen is derived
 * from the inventory: a literal here goes quietly wrong on somebody else's commit.
 */

export function FeatureTableCard({ inventory }: { inventory: FeatureInventory | null }) {
  if (!inventory) return <p className="empty">Loading…</p>
  if (!inventory.available) return <p className="empty">{inventory.reason}</p>

  const widest = Math.max(1, ...inventory.families.map((f) => f.columns))

  return (
    <>
      <Row
        icon="table"
        accent="aqua"
        primary={`${count(inventory.rows)} rows × ${count(inventory.columns)} columns`}
        secondary={`${bytes(inventory.bytes)} in ${inventory.row_groups} row groups, rebuilt ${
          relativeTime(inventory.built_at)}. One row per subject per ${
          inventory.grid_minutes}-minute slot.`}
      />
      {!inventory.statistics && (
        <Row
          icon="alert"
          accent="orange"
          primary="No column statistics in this file"
          secondary="Read from the schema; missing-value percentages need a full
            table read."
        />
      )}
      <div className="cols">
        {columnsOf(inventory.families).map((column, i) => (
          <div key={i}>
            {column.map((f) => (
              <FamilyRow key={f.family} family={f} max={widest} />
            ))}
          </div>
        ))}
      </div>
    </>
  )
}

/** One family, with a bar to read it against the others. `sqrt`, not linear:
 *  the families span three orders of magnitude, and linearly all but the largest
 *  few are invisible. The count is printed above the bar because the NUMBER is
 *  what you read for a value; the bar is for comparison. */
function FamilyRow({ family: f, max }: { family: FeatureFamily; max: number }) {
  const none = f.columns === 0
  const width = max > 0 ? Math.sqrt(f.columns / max) * 100 : 0
  return (
    <div className={`famrow${none ? ' off' : ''}`}>
      <div>
        <b>{f.family.replace(/_/g, ' ')}</b>
        <span>{f.words}. {share(f.null_frac)} missing.</span>
      </div>
      <div className="amt">
        <em className="num">{count(f.columns)}</em>
        <div className="track">
          <i style={{ width: `${width.toFixed(1)}%` }} />
        </div>
      </div>
    </div>
  )
}

/** Two balanced columns in declared order, down the first and then the second;
 *  `Math.ceil` puts the odd one at the foot of the left column. */
function columnsOf<T>(items: T[]): [T[], T[]] {
  const half = Math.ceil(items.length / 2)
  return [items.slice(0, half), items.slice(half)]
}

/** The four daily lags, and which of them this horizon may read. */
function LagRows({ lags }: { lags: HorizonLag[] }) {
  return (
    <>
      {lags.map((lag) => (
        <Row
          key={lag.days}
          icon={lag.safe ? 'check' : 'eye-off'}
          accent={lag.safe ? 'aqua' : 'grey'}
          muted={!lag.safe}
          primary={`${lag.days} ${lag.days === 1 ? 'day' : 'days'} before the target slot`}
          secondary={lag.why ?? `${lag.column} — known by the time the forecast is made.`}
          trailing={
            <Chip label={lag.safe ? 'used' : 'not allowed'}
                  icon={lag.safe ? 'check' : 'eye-off'}
                  accent={lag.safe ? 'aqua' : 'grey'} />
          }
        />
      ))}
    </>
  )
}

/** The line under the horizon readout: what this one costs, and how far the
 *  folds are held apart. Here, not in the slider, which knows nothing of recipes. */
export function horizonSummary(
  recipe: HorizonRecipe | null,
  totalColumns: number | null,
  pending: boolean,
): string {
  // While the slider moves, the recipe on screen belongs to the horizon you have just left.
  if (pending) return 'reading what this horizon may use…'
  if (!recipe) return 'Loading…'
  if (!recipe.available) return recipe.reason
  const of = totalColumns === null ? "the table's" : `the table's ${count(totalColumns)}`
  return `${count(recipe.n_features)} of ${of} columns · folds held `
    + `${recipe.embargo_hours} h apart`
}

export function HorizonCard({ recipe }: { recipe: HorizonRecipe | null }) {
  if (!recipe) return <p className="empty">Loading…</p>
  if (!recipe.available) return <p className="empty">{recipe.reason}</p>

  const blocked = recipe.daily_lags.filter((l) => !l.safe).length
  return (
    <>
      <Row
        icon="target"
        accent="aqua"
        primary={`Predicting ${recipe.target} as a change from ${recipe.residual_base}`}
        secondary={`Reads ${recipe.columns_read} columns; folds held ${
          recipe.embargo_hours} h apart.`}
        trailing={
          // Driven off `ships`, not `served_by`'s truthiness: "none" is a truthy
          // string, which reads "trained and lost" as shipping.
          recipe.ships === null ? (
            <Chip label="untrained" icon="minus" accent="grey" />
          ) : recipe.ships ? (
            <Chip label="model" icon="model" accent="aqua" />
          ) : (
            <Chip label="not served" icon="minus" accent="grey" />
          )
        }
      />

      {/* Side by side, because the point of this step is the CONTRAST: what it
          may read now, against what it may not read from earlier days. */}
      <div className="cols">
        <div>
          <p className="subhead">Read at the moment the forecast is made</p>
          {recipe.families.map((f) => (
            <Row
              key={f.family}
              icon="check"
              accent="blue"
              primary={f.family.replace(/_/g, ' ')}
              secondary={f.words}
              trailing={<Chip label={`${f.columns}`} icon="table" accent="blue" />}
            />
          ))}
        </div>
        <div>
          <p className="subhead">
            The same slot on earlier days
            {blocked > 0
              && ` — ${blocked} of these ${blocked === 1 ? 'is' : 'are'} off limits here`}
          </p>
          <LagRows lags={recipe.daily_lags} />
        </div>
      </div>
    </>
  )
}

export function ColumnCard({ series }: { series: FeatureSeries | null }) {
  if (!series) return <p className="empty">Loading…</p>
  if (!series.available) return <p className="empty">{series.reason}</p>

  const { summary, safe_for: safety } = series
  const points: Point[] = series.points.map((p) => ({ t: p.t, v: p.v }))
  const unsafe = safety !== null && !safety.safe

  const range = summary.min === null
    ? 'It has no values in this window.'
    : `It ranges from ${summary.min} to ${summary.max}, averaging ${summary.mean}.`
  // Long form for the chart's `aria-label`, short one for the line under it;
  // why a missing value reads as "unknown" is in DOCS.md under "The Data tab".
  const gaps = summary.nulls === 0
    ? 'No slot is missing it.'
    : `${count(summary.nulls)} of ${count(summary.n)} slots are missing it — which the model
       reads as "unknown" rather than as a zero.`
  const shortGaps = summary.nulls === 0
    ? 'No slot is missing it.'
    : `${count(summary.nulls)} of ${count(summary.n)} slots are missing it.`
  const prose = `${series.column} for ${series.subject}. ${range} ${gaps}${
    series.thinned ? ' The window was sampled down to fit the chart.' : ''}`
  const caption = `${range} ${shortGaps}${series.thinned ? ' Sampled down to fit.' : ''}`

  return (
    <>
      {/* An unsafe lag is charted, but never without this. The column is really
          in the table; what it is not is something the model may read. */}
      {unsafe && (
        <Row
          icon="eye-off"
          accent="orange"
          primary={`Horizon +${safety.horizon_h} h is not allowed to use this column`}
          secondary={safety.why}
        />
      )}
      <Row
        icon="chart"
        accent={unsafe ? 'orange' : 'aqua'}
        primary={series.column}
        secondary={`${series.words}. Subject: ${series.subject}.`}
        trailing={<Chip label={series.family.replace(/_/g, ' ')} icon="table" accent="blue" />}
      />
      <TimeSeries
        points={points}
        accent={unsafe ? 'orange' : 'aqua'}
        label={series.column}
        startLabel={absoluteTime(series.start)}
        endLabel={absoluteTime(series.stop)}
        summary={prose}
        caption={caption}
      />
    </>
  )
}
