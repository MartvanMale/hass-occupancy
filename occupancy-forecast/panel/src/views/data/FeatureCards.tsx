import type {
  FeatureFamily, FeatureInventory, FeatureSeries, HorizonLag, HorizonRecipe,
} from '../../types'
import { Chip } from '../../components/Chip'
import { Row } from '../../components/Row'
import { TimeSeries, type Point } from '../../components/TimeSeries'
import { absoluteTime, relativeTime } from '../../format'

/**
 * The feature table, in the only three ways it can usefully be looked at.
 *
 * Not as a table: it is a thousand-odd columns wide, and a list of that many
 * column names is no more browsable than the table is. What a reader actually
 * wants is how much of each *kind* of thing there is, what one column looks like
 * over time, and which columns a given horizon is allowed to read.
 *
 * Every count on screen is derived from the inventory, never written down here.
 * The width moves whenever a feature family is added -- it was 988 before three
 * candidate families landed and 1,230 after -- and a literal in this file is a
 * number that goes quietly wrong on somebody else's commit.
 */

const count = (n: number) => n.toLocaleString()

const bytes = (n: number) => {
  if (n < 1024) return `${n} B`
  const units = ['kB', 'MB', 'GB']
  let value = n / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1 }
  return `${value.toFixed(1)} ${units[i]}`
}

const percent = (frac: number | null) =>
  frac === null ? 'unknown' : `${(frac * 100).toFixed(frac > 0 && frac < 0.01 ? 2 : 0)}%`

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

/** One family: what it is, how much of it there is, and a bar to read the
 *  second against the others by.
 *
 *  `sqrt`, not a linear share. The families span three orders of magnitude --
 *  `key` has 2 columns and `target calendar` has 384 -- and linearly every
 *  family but the largest three is a bar too short to see, which is a chart
 *  that answers one question and hides eleven. The square root is an area
 *  encoding on a one-dimensional mark, which is a compromise, and it is why the
 *  count is printed above the bar rather than left to it: the NUMBER is what
 *  you read for a value, the bar is what you read for a comparison. */
function FamilyRow({ family: f, max }: { family: FeatureFamily; max: number }) {
  const none = f.columns === 0
  const width = max > 0 ? Math.sqrt(f.columns / max) * 100 : 0
  return (
    <div className={`famrow${none ? ' off' : ''}`}>
      <div>
        <b>{f.family.replace(/_/g, ' ')}</b>
        <span>{f.words}. {percent(f.null_frac)} missing.</span>
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

/** Split into two balanced columns, in order down the first and then the second.
 *  The families are declared in the order the table is built, which is an order
 *  worth keeping; `Math.ceil` puts the odd one at the foot of the left column
 *  rather than leaving a short first column and a long second. */
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
 *  folds are held apart. Lives here rather than in the slider because it is a
 *  fact about the recipe, and the slider knows nothing about recipes. */
export function horizonSummary(
  recipe: HorizonRecipe | null,
  totalColumns: number | null,
  pending: boolean,
): string {
  // While the slider is still moving the recipe on screen belongs to the horizon
  // you have just left. Saying nothing is better than attributing the old
  // horizon's column count to the new one for a quarter of a second.
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
          // Driven off `ships`, not off `served_by`'s truthiness: "none" is a
          // truthy string, so the old test read "trained and lost" as "trained
          // and shipping". `ships` is the field that is already three-valued.
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
          may read now, against what it may not read from earlier days. Stacked,
          the second list read as a continuation of the first. */}
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
  // Long form for the chart's `aria-label`, short one for the line under it.
  // That a missing value is read as "unknown" rather than as a zero is in
  // DOCS.md under "The Data tab"; it does not need restating on every column.
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
