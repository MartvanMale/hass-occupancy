import { useEffect, useState, type ReactNode } from 'react'
import { getForecast } from '../api'
import type { Forecast, NextChange, Status, SubjectForecast } from '../types'
import { Card } from '../components/Card'
import { Chip } from '../components/Chip'
import { Row } from '../components/Row'
import { Curves, type Curve } from '../components/Curves'
import { Horizons } from '../components/Horizons'
import { Training } from '../components/Training'
import { VerificationCard } from '../components/VerificationCard'
import { percent, pretty, relativeTime } from '../format'

/**
 * Overview: who is home, who is coming home, and how much of it to believe.
 *
 * The forecast is read from one endpoint returning what was last PUBLISHED, so
 * this view and the Home Assistant entities cannot drift apart. Two pairs and
 * then the chart: what the forecast says, what is behind it, then 48 hours.
 */

const CURVE_ACCENT = ['blue', 'orange', 'aqua', 'red'] as const

function clock(at: Date): string {
  return at.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

/**
 * The one sentence a row shows. The model decides WHETHER and which hour; the
 * routine may only sharpen it -- see `predict._next_change`. The date is named
 * when it is not today.
 */
function changeSentence(change: NextChange): string {
  // Not "no change expected", a claim about the house: the curve is sparse, so
  // the usual reason there is no crossing is that no horizon near it publishes.
  if (change.direction === null || change.at === null) {
    return 'No arrival or departure time is predicted.'
  }
  const at = new Date(change.at)
  if (Number.isNaN(at.getTime())) return 'No arrival or departure time is predicted.'

  const sameDay = at.toDateString() === new Date().toDateString()
  const when = sameDay ? clock(at) : `${clock(at)} ${dayWord(at)}`
  return change.direction === 'leaving'
    ? `Expected to leave around ${when}.`
    : `Expected back around ${when}.`
}

/**
 * How well-supported that time is. Without it the card states a rare day as
 * confidently as a routine one, which is what made an hour the chart beside it
 * read as "home" look like a forecast.
 */
function supportClause(change: NextChange): string {
  const day = change.routine_day
  if (day === null || change.at === null) return ''
  const at = new Date(change.at)
  if (Number.isNaN(at.getTime()) || day.n_weekday < 3) return ''

  const weekday = at.toLocaleDateString(undefined, { weekday: 'long' })
  const support = `Left on ${day.n_left_weekday} of ${day.n_weekday} ${weekday}s`
  // The spread only where it describes the time actually shown: on a refused
  // routine hour it is the spread around a moment the row does not name.
  const sd = change.direction === 'leaving' ? day.departure_sd : day.return_sd
  if (change.at_from !== 'routine' || sd === null || !(sd > 0)) return `${support}.`
  const ms = sd * 3_600_000
  return `${support}, usually ${clock(new Date(at.getTime() - ms))}`
    + `–${clock(new Date(at.getTime() + ms))}.`
}

/** "tomorrow", or a weekday name past that. */
function dayWord(at: Date): string {
  const days = Math.round(
    (new Date(at.toDateString()).getTime() - new Date(new Date().toDateString()).getTime())
    / 86_400_000)
  return days === 1 ? 'tomorrow'
    : at.toLocaleDateString(undefined, { weekday: 'long' })
}

/** "in 3 h", "within the hour", from a moment rather than the raw crossing --
 *  so the chip and the sentence cannot disagree about the same event. */
function untilLabel(at: string | null): string {
  // Both of these are the ABSENT case, not a settled one: "steady" would be a
  // positive claim about a house the add-on has nothing to say about.
  if (at === null) return 'no estimate'
  const hours = (new Date(at).getTime() - Date.now()) / 3_600_000
  if (!Number.isFinite(hours)) return 'no estimate'
  if (hours <= 1) return 'within the hour'
  return `in ${Math.round(hours)} h`
}


function SubjectRows({ subjects, house }: { subjects: SubjectForecast[]; house: string }) {
  return (
    <>
      {subjects.map((s) => {
        const homeNow = s.current >= 0.5
        const soon = s.curve['1']
        return (
          <Row
            key={s.subject}
            icon={homeNow ? 'model' : 'baseline'}
            accent={homeNow ? 'aqua' : 'orange'}
            primary={s.subject === house ? 'The house' : pretty(s.subject)}
            secondary={
              `${homeNow ? 'Home now' : 'Away now'}` +
              (soon === undefined
                ? ' · +1 h not served'
                : ` · ${percent(soon)} in an hour`)
            }
            trailing={
              <Chip
                label={soon === undefined ? '—' : percent(soon)}
                icon={homeNow ? 'model' : 'baseline'}
                accent={homeNow ? 'aqua' : 'orange'}
              />
            }
          />
        )
      })}
    </>
  )
}

function ChangeRows({ subjects, house }: { subjects: SubjectForecast[]; house: string }) {
  return (
    <>
      {subjects.map((s) => {
        const change: NextChange = s.next_change
          ?? { direction: null, in_hours: null, at: null, at_from: null,
               routine_at: null, routine_day: null }
        const leaving = change.direction === 'leaving'
        const arriving = change.direction === 'arriving'
        const name = s.subject === house ? 'The house' : pretty(s.subject)
        // `eta_minutes` is null unless they are demonstrably travelling, so the
        // sentence needs no "if already on the way" hedge.
        const secondary = [
          changeSentence(change),
          supportClause(change),
          arriving && s.eta_minutes !== null
            ? `On the way, ${Math.round(s.eta_minutes)} min out.` : '',
        ].filter(Boolean).join(' ')
        return (
          <Row
            key={s.subject}
            icon={leaving || arriving ? 'clock' : 'model'}
            accent={leaving ? 'orange' : arriving ? 'blue' : 'aqua'}
            primary={name}
            secondary={secondary}
            trailing={
              <Chip
                label={untilLabel(change.at)}
                icon={leaving || arriving ? 'clock' : 'model'}
                accent={leaving ? 'orange' : arriving ? 'blue' : 'aqua'}
              />
            }
          />
        )
      })}
    </>
  )
}

export function OverviewView({ status, refreshStatus }: {
  status: Status | null
  refreshStatus: () => Promise<void>
}) {
  const [forecast, setForecast] = useState<Forecast | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    const load = () =>
      getForecast()
        .then((f) => { if (live) { setForecast(f); setError(null) } })
        .catch((e) => { if (live) setError(String(e)) })
    load()
    // The forecast is republished every cycle; a minute is well inside that and
    // far outside anything that would make this tab a load.
    const timer = setInterval(load, 60_000)
    return () => { live = false; clearInterval(timer) }
  }, [])

  // One sentence standing in for whichever forecast cards cannot be drawn.
  // Deliberately NOT an early return: it would hide the two cards that explain why.
  const missing: ReactNode | null = error
    ? <p className="empty error">{error}</p>
    : !forecast
      ? <p className="empty">Loading…</p>
      : !forecast.available
        ? (
          <p className="empty">
            Nothing published yet — the first forecast lands a few minutes after start.
          </p>
        )
        : null

  const ready = forecast?.available ? forecast : null
  const house = ready?.house ?? ''
  const ordered = ready
    ? [...ready.subjects.filter((s) => s.subject !== house),
       ...ready.subjects.filter((s) => s.subject === house)]
    : []

  // `?? null`, never `?? 0`: `curve` is sparse, and a zero is the strongest claim
  // the chart can make -- certainly away, for hours the add-on said nothing about.
  const curves: Curve[] = ordered.map((s, i) => ({
    key: s.subject,
    label: s.subject === house ? 'House' : pretty(s.subject),
    accent: CURVE_ACCENT[i % CURVE_ACCENT.length]!,
    values: (ready?.horizons ?? []).map((h) => s.curve[String(h)] ?? null),
  }))

  // A cycle ran, but no horizon has a model -- unlike `missing`, "no cycle has
  // run". The two cards on the right explain this one, so only the chart goes.
  const nothingServed = ready !== null && curves.length > 0
    && curves.every((c) => c.values.every((v) => v === null))

  return (
    /* A grid, not bare siblings: `.card` carries no margin of its own, so
       full-width cards outside `.cards` sit flush. */
    <div className="stack">
      {/* Two pairs in ROW order, so the cards of a row share a height: what the
          forecast SAYS on the left, what is BEHIND it on the right. */}
      {/* No explanatory subtitles: this page is left open, and DOCS.md explains
          the numbers under "Reading the panel". */}
      <div className="cards pair">
        <Card
          title="Right now"
          subtitle={ready ? `Published ${relativeTime(ready.predicted_at)}.` : undefined}
        >
          {missing ?? <SubjectRows subjects={ordered} house={house} />}
        </Card>

        <Card title="What is serving each horizon">
          {status
            ? <Horizons served={status.served_by} kinds={status.model_kind}
                        beatenBy={status.best_baseline} />
            : <p className="empty">Loading…</p>}
        </Card>

        <Card title="Next expected change">
          {missing ?? <ChangeRows subjects={ordered} house={house} />}
        </Card>

        <Card title="Training">
          {status
            ? <Training status={status} refresh={refreshStatus} />
            : <p className="empty">Loading…</p>}
        </Card>
      </div>

      {/* Full width, alone: the one thing on this tab that spends every pixel. */}
      <Card title="The next 48 hours">
        {missing ?? (nothingServed ? (
          <p className="empty">
            No horizon has a trained model yet, so there is nothing to draw —
            see “What is serving each horizon” above.
          </p>
        ) : (
          <Curves
            curves={curves}
            hours={ready?.horizons ?? []}
            night={ready?.night ?? []}
            // The slot the horizons are measured from. Every subject comes off
            // one row of one feature table, so the first is the household's.
            at={ordered[0]?.observed_at}
            label={`Probability of being home over the next ${
              ready?.horizons[ready.horizons.length - 1]} hours, `
              + `for ${curves.map((c) => c.label).join(', ')}.`}
          />
        ))}
      </Card>

      {/* Under the forecast: what it says, then what happened last time. Its own
          slider, because the horizon worth checking is rarely the one just read. */}
      <VerificationCard status={status} />
    </div>
  )
}
