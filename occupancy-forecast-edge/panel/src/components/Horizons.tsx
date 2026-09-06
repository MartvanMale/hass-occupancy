import type { CSSProperties } from 'react'
import { Row } from './Row'
import { Chip } from './Chip'
import type { Accent } from './Icon'
import { integerRuns } from './geometry'
import type { ModelKind, ServedBy } from '../types'

/**
 * What is serving each horizon, as one row and a strip.
 *
 * There are 48 horizons (`config.HORIZONS_H`), and the page this replaced drew
 * one full Mushroom row for each of them: a wall of near-identical text in a
 * card sitting next to three-row cards. Almost none of it was information. What
 * a reader actually wants is where the boundary falls and what is on each side
 * of it, and that fits in a sentence and a 22px strip.
 *
 * Three states, not two: a horizon is served by a model fitted for that horizon
 * alone, by the one pooled model fitted over every horizon, or by nothing at
 * all. Where the first two hand over is the measured crossover, and it is the
 * most interesting thing on the strip, so it is worth a colour.
 *
 * **The two model families are green and BLUE, not two greens.** Two shades of
 * one hue were tried and measured: to sit inside the dark-mode lightness band a
 * second green has to be close enough to the first that normal-vision Delta E
 * falls to 3.6-5.9 against a floor of 15, and the only green that separates
 * properly is too light for the dark surface and drops to 2:1 on the light one.
 * Green/blue passes every check in both modes, and blue is already a token.
 *
 * **The third state is hollow grey, not a third colour.** It was orange, back
 * when a horizon the model had not earned was served its baseline instead.
 * Nothing is published there now, and a solid third hue would say "a third kind
 * of thing serves this" when the fact is that nothing does. It would also put a
 * third saturated fill beside a pair whose separation was measured. An empty
 * cell is the honest shape for an empty horizon.
 *
 * **This strip is about the MODELS, not about the last forecast.** A horizon
 * that ships can still come out empty on the chart, if a feature the model
 * wanted was missing from that particular row. The two will occasionally
 * disagree, and that disagreement is the signal -- `predict_rows` logs a
 * warning when it happens. It is not a bug to be reconciled away.
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
  // `model_kind` is keyed only by the horizons a model serves, and it is absent
  // altogether on a build older than 0.4.0 -- so a missing entry means "a model
  // serves this, family unknown", which is the dedicated colour rather than a
  // fourth state nobody wants to see.
  const stateOf = (h: number): State =>
    !isModel(h) ? 'none' : kinds[String(h)] === 'pooled' ? 'pooled' : 'dedicated'
  const modelled = horizons.filter(isModel)

  const primary =
    modelled.length === 0
      ? 'No horizon has beaten its baseline yet, so nothing is published'
      : modelled.length === horizons.length
        ? `Every horizon is served by the model`
        : `Model serves ${modelled.length} of ${horizons.length} horizons`

  const dedicated = horizons.filter((h) => stateOf(h) === 'dedicated')
  const pooled = horizons.filter((h) => stateOf(h) === 'pooled')
  const unserved = horizons.length - modelled.length

  // The visible line: three counted facts, joined. What it USED to say -- that
  // an unserved horizon means the model did not beat its baseline, so the
  // sensor reads unknown and the chart has a gap -- is in DOCS.md under
  // "Which horizons publish, and why some do not". It is the same sentence
  // every time the page loads, and after the first read it is furniture.
  const secondary = [
    modelled.length ? summariseRuns(runs(modelled)) : '',
    dedicated.length && pooled.length
      ? `${dedicated.length} per horizon, ${pooled.length} pooled`
      : '',
    unserved ? `${unserved} not served` : '',
  ].filter(Boolean).join(' · ')

  // The strip's accessible description, and the ONLY place the long form
  // survives on screen. A sighted reader has the strip itself and the colour
  // key under it; a screen reader has this sentence and nothing else, so
  // shortening it here would not be a trim, it would be a removal.
  const split = dedicated.length && pooled.length
    ? `${dedicated.length} of those are fitted for one horizon each (${summariseRuns(runs(dedicated))}) `
      + `and ${pooled.length} come from the one pooled model (${summariseRuns(runs(pooled))}). `
    : ''
  const where = modelled.length ? `${summariseRuns(runs(modelled))}. ` : ''
  const rest = unserved
    ? `${modelled.length ? 'Elsewhere nothing' : 'Nothing'} is published: the `
      + 'model did not beat its own baseline, so the sensor reads unknown and '
      + 'the forecast chart has a gap.'
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
