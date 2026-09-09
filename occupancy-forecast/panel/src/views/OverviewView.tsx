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
 * This is the dashboard people actually kept open, rebuilt inside the add-on.
 * It was a hand-built Lovelace view before, which meant every installation had
 * to write out its own entity ids -- and those ids embed the add-on's slug, so
 * the stable and edge builds needed two copies of the same YAML kept in step by
 * hand. Here the panel already knows which add-on it is.
 *
 * The forecast is read from one endpoint that returns what was last PUBLISHED,
 * so this view and the Home Assistant entities cannot drift apart.
 *
 * The right-hand column -- which horizons a model is serving, and when the
 * models were last rebuilt -- was on Setup, and was the only thing there that
 * changed on its own. Setup is a page you fill in once; this is the page that is
 * left open, and "how much of this forecast exists at all" is a fact about the
 * forecast beside it, not a setting.
 *
 * The layout is two pairs and then the chart: what the forecast SAYS on the
 * left, what is BEHIND it on the right, and the 48 hours across the foot,
 * because that one spends every pixel it is given and the other four do not.
 */

const CURVE_ACCENT = ['blue', 'orange', 'aqua', 'red'] as const

/**
 * The one sentence a row shows: what is coming, and when.
 *
 * The model decides WHETHER and the routine decides WHEN -- see
 * `predict._next_change`. Two lines here used to read "Expected back in 3 h."
 * over "Expected back around 18:00.", which is one event with two answers 2.4 h
 * apart and reads as a fault however it is worded.
 *
 * The date is named when it is not today: a bare "08:00" on a row about a change
 * fifteen hours away is worse than the relative form it replaced.
 */
function changeSentence(change: NextChange): string {
  if (change.direction === null || change.at === null) {
    return 'No change expected in the next 48 hours.'
  }
  const at = new Date(change.at)
  if (Number.isNaN(at.getTime())) return 'No change expected in the next 48 hours.'

  const clock = at.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  const sameDay = at.toDateString() === new Date().toDateString()
  const when = sameDay ? clock : `${clock} ${dayWord(at)}`
  return change.direction === 'leaving'
    ? `Expected to leave around ${when}.`
    : `Expected back around ${when}.`
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
  if (at === null) return 'steady'
  const hours = (new Date(at).getTime() - Date.now()) / 3_600_000
  if (!Number.isFinite(hours)) return 'steady'
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
          ?? { direction: null, in_hours: null, at: null, at_from: null }
        const leaving = change.direction === 'leaving'
        const arriving = change.direction === 'arriving'
        const name = s.subject === house ? 'The house' : pretty(s.subject)
        // The ETA earns its place only while they are demonstrably travelling:
        // `eta_minutes` is null otherwise, which is why the old "if already on
        // the way" hedge could be dropped.
        const secondary = changeSentence(change) +
          (arriving && s.eta_minutes !== null
            ? ` On the way, ${Math.round(s.eta_minutes)} min out.`
            : '')
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

  // One sentence standing in for whichever forecast cards cannot be drawn, and
  // null once there is a forecast. Deliberately NOT an early return: on day one
  // "nothing published yet" and "not trained yet" are the same story told from
  // both ends, and returning here would show only the first half of it while
  // hiding the two cards that explain why.
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

  // `?? null`, never `?? 0`. `curve` is sparse -- a horizon no model serves is
  // simply not in it -- and a zero here is not a missing value, it is the
  // strongest claim the chart can make: certainly away, for exactly the hours
  // the add-on has nothing to say about.
  const curves: Curve[] = ordered.map((s, i) => ({
    key: s.subject,
    label: s.subject === house ? 'House' : pretty(s.subject),
    accent: CURVE_ACCENT[i % CURVE_ACCENT.length]!,
    values: (ready?.horizons ?? []).map((h) => s.curve[String(h)] ?? null),
  }))

  // A cycle ran, but no horizon has a model. Distinct from `missing` above,
  // which is "no cycle has run": the two cards on the right explain this one,
  // so only the chart needs replacing.
  const nothingServed = ready !== null && curves.length > 0
    && curves.every((c) => c.values.every((v) => v === null))

  return (
    /* A grid, not bare siblings. `.card` carries no margin of its own -- all the
       spacing on this tab came from `.cards`' own gap, so the full-width cards
       after that grid sat flush against each other. That exact bug was fixed
       once before on this tab ("The cards on Now were touching") and came back
       the moment a fourth full-width card was added, which is the argument for
       a container rather than a margin: `.steps` on the Data tab is the same
       shape for the same reason. */
    <div className="stack">
      {/* A two-by-two grid in ROW order, so the two cards of a row share a
          height and their edges line up across the page. Not the auto-fit grid
          `.cards` gives by default: that would put all four in one row on a wide
          panel and say they are four equal things, when they are two pairs --
          what the forecast SAYS on the left, what is BEHIND it on the right.

          The cards stretch, which is the point and also the cost: the shorter
          card of a row carries whitespace under its content. That trade is
          deliberate. The version before this let each column pack
          independently, which wasted no space and left a ragged seam down the
          middle of the page, which was worse. */}
      {/* The only subtitle left on this tab is "Right now"'s, and that one is a
          timestamp. Everything else here states numbers; DOCS.md explains them,
          under "Reading the panel" -- the ship gate and the two model families,
          how a crossing time is arrived at and why it is hourly, what the
          training cadence means. This is a page somebody leaves open, and a
          paragraph you have already read is furniture every time after. */}
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

      {/* Full width, alone: 48 hours across three curves is the one thing on
          this tab that spends every pixel it is given. Its subtitle said that a
          gap is an hour no model has earned and that the grey is the household
          asleep, both of which are already on screen: the legend carries an
          "asleep" key, and the card above counts the gap. */}
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
            // The slot the horizons are measured from, so the axis can show
            // clock times. Every subject shares it -- they come off one row of
            // one feature table -- so the first is the household's.
            at={ordered[0]?.observed_at}
            label={`Probability of being home over the next ${
              ready?.horizons[ready.horizons.length - 1]} hours, `
              + `for ${curves.map((c) => c.label).join(', ')}.`}
          />
        ))}
      </Card>

      {/* Under the forecast, and that order is the argument: what it says, then
          what happened last time it said it. Its own slider, because the
          horizon worth checking is rarely the one you were just looking at. */}
      <VerificationCard status={status} />
    </div>
  )
}
