import type { CSSProperties } from 'react'
import { Row } from './Row'
import { Chip } from './Chip'
import type { Accent } from './Icon'
import { integerRuns } from './geometry'
import type { ModelKind, ServedBy } from '../types'

/**
 * What is serving each horizon, as one row and a 22px strip -- 48 of them, and
 * a wall of near-identical rows was almost none of it information. Three
 * states: a dedicated model, the pooled one, or nothing. Green and BLUE, not
 * two greens, which cannot separate inside the dark-mode lightness band.
 * This strip is about the MODELS, not about the last forecast; they can differ.
 */

type Range = [number, number]

/** Contiguous runs, so 1..9 is one range and not nine. */
const runs = (nums: number[]): Range[] => integerRuns(nums)

const rangeLabel = ([a, b]: Range) => (a === b ? `+${a} h` : `+${a} h to +${b} h`)

function joinList(parts: string[]): string {
  if (parts.length <= 1) return parts[0] ?? ''
  return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
}

/** Capped at three: past that the shape of the split has stopped being readable
 *  as prose, and the strip below is the better way to see it. */
function summariseRuns(rs: Range[]): string {
  if (rs.length <= 3) return joinList(rs.map(rangeLabel))
  return `${rs.slice(0, 3).map(rangeLabel).join(', ')} and ${rs.length - 3} more`
}

type State = 'dedicated' | 'pooled' | 'none'

const ACCENT: Record<State, Accent> = {
  dedicated: 'aqua',
  pooled: 'blue',
  none: 'grey',
}

const WORDS: Record<State, string> = {
  dedicated: 'model, fitted for this horizon',
  pooled: 'model, pooled over all horizons',
  none: 'not served — nothing is published',
}

export function Horizons({ served, kinds = {}, beatenBy = {} }: {
  served: Record<string, ServedBy>
  kinds?: Record<string, ModelKind>
  /** Per horizon, the baseline that beat the model. Absent where no model was
   *  ever trained -- both publish nothing, only one had a bake-off. */
  beatenBy?: Record<string, string>
}) {
  const horizons = Object.keys(served)
    .map(Number)
    .sort((a, b) => a - b)

  const isModel = (h: number) => served[String(h)] === 'model'
  // A missing `model_kind` entry means "a model serves this, family unknown" --
  // the dedicated colour, not a fourth state.
  const stateOf = (h: number): State =>
    !isModel(h) ? 'none' : kinds[String(h)] === 'pooled' ? 'pooled' : 'dedicated'
  const modelled = horizons.filter(isModel)

  const dedicated = horizons.filter((h) => stateOf(h) === 'dedicated')
  const pooled = horizons.filter((h) => stateOf(h) === 'pooled')
  const unserved = horizons.length - modelled.length
  // `best_baseline` is present for exactly the horizons that lost a bake-off,
  // which is the convention the tooltip reads.
  const beaten = horizons.filter((h) => stateOf(h) === 'none' && beatenBy[String(h)])
  const untrained = unserved - beaten.length

  const primary =
    modelled.length === 0
      ? beaten.length
        ? 'No horizon has beaten its baseline yet, so nothing is published'
        : 'No model has trained yet, so nothing is published'
      : modelled.length === horizons.length
        ? `Every horizon is served by the model`
        : `Model serves ${modelled.length} of ${horizons.length} horizons`

  // Three counted facts, joined. The longer explanation is in DOCS.md.
  const secondary = [
    modelled.length ? summariseRuns(runs(modelled)) : '',
    dedicated.length && pooled.length
      ? `${dedicated.length} per horizon, ${pooled.length} pooled`
      : '',
    unserved ? `${unserved} not served` : '',
  ].filter(Boolean).join(' · ')

  // The strip's accessible description and the only place the long form
  // survives: a screen reader has this sentence and nothing else.
  const split = dedicated.length && pooled.length
    ? `${dedicated.length} of those are fitted for one horizon each (${summariseRuns(runs(dedicated))}) `
      + `and ${pooled.length} come from the one pooled model (${summariseRuns(runs(pooled))}). `
    : ''
  const where = modelled.length ? `${summariseRuns(runs(modelled))}. ` : ''
  const why = beaten.length && untrained
    ? `the model lost to its own baseline at ${beaten.length} of them and has not `
      + `trained or passed its publication checks at the other ${untrained}`
    : beaten.length
      ? 'the model did not beat its own baseline'
      : 'the model has not trained or has not passed its publication checks'
  const rest = unserved
    ? `${modelled.length ? 'Elsewhere nothing' : 'Nothing'} is published: ${why}, `
      + 'so the sensor reads unknown and the forecast chart has a gap.'
    : ''
  const described = `${where}${split}${rest}`

  const anyModel = modelled.length > 0

  return (
    <>
      <Row
        icon={anyModel ? 'model' : 'minus'}
        accent={anyModel ? 'aqua' : 'grey'}
        primary={primary}
        secondary={secondary}
        trailing={
          <Chip
            label={`${modelled.length}/${horizons.length}`}
            icon={anyModel ? 'model' : 'minus'}
            accent={anyModel ? 'aqua' : 'grey'}
          />
        }
      />

      <div
        className="strip"
        role="img"
        aria-label={`${primary}. ${described} Read left to right from +${horizons[0]} h to +${horizons[horizons.length - 1]} h.`}
      >
        {horizons.map((h) => {
          const state = stateOf(h)
          const beat = beatenBy[String(h)]
          const what = state !== 'none'
            ? WORDS[state]
            : beat
              ? `not served; the ${beat} baseline beat the model`
              : 'not served; no model trained for it yet'
          return (
            <i
              key={h}
              className={state === 'none' ? 'off' : 'model'}
              style={state === 'none' ? undefined
                : { '--c': `var(--rgb-${ACCENT[state]})` } as CSSProperties}
              title={`+${h} h — ${what}`}
            />
          )
        })}
      </div>

      <div className="scale">
        <span>+{horizons[0]} h</span>
        {/* Names only the states actually on the strip: a key to a colour
            that is not there is a question rather than an answer. */}
        <span className="legend" aria-hidden="true">
          {([
            ['dedicated', 'per horizon', dedicated.length],
            ['pooled', 'pooled', pooled.length],
            ['none', 'not served', unserved],
          ] as [State, string, number][])
            .filter(([, , n]) => n > 0)
            .map(([state, label]) => (
              <span key={state} className="key">
                <i className={state === 'none' ? 'offkey' : undefined}
                   style={state === 'none' ? undefined
                     : { '--c': `var(--rgb-${ACCENT[state]})` } as CSSProperties} />
                <span>{label}</span>
              </span>
            ))}
        </span>
        <span>+{horizons[horizons.length - 1]} h</span>
      </div>
    </>
  )
}
