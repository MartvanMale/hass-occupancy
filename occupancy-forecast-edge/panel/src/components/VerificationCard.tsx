import { useEffect, useState } from 'react'
import { getVerification } from '../api'
import type { Status, Verification as VerificationData } from '../types'
import { Card } from './Card'
import { HorizonSlider } from './HorizonSlider'
import { Row } from './Row'
import { Select } from './Select'
import { Verification } from './Verification'
import { useDebounced } from '../hooks'
import { absoluteTime, DAY_OPTIONS_RECENT, pretty } from '../format'

/**
 * "Was it right?" -- the same card on both tabs, each owning its own controls.
 *
 * The live Brier here and the backtest Brier on the Judge step are DIFFERENT
 * quantities: the backtest cannot see the nowcast pin, a tracker that went
 * quiet, or the ship gate, and is computed from rows never actually served.
 */

/** The window the log actually keeps, which is not the window being charted.
 *  0 is the "keep everything" setting, not a zero-length window. */
function keptFor(days: number | null): string {
  if (days === null) return 'Kept for as long as the Setup tab says.'
  return days === 0 ? 'Kept indefinitely.' : `Kept for ${days} days.`
}

export function VerificationCard({ status, defaultHorizon = 6 }: {
  status: Status | null
  /** 6 h by default because that is the window the Lovelace card watches and
   *  the one a heating decision rests on. */
  defaultHorizon?: number
}) {
  const [subject, setSubject] = useState<string>('')
  // Debounced: dragging end to end is otherwise 48 requests to an endpoint that
  // is deliberately not cached.
  const [horizonInput, setHorizonInput] = useState<number>(defaultHorizon)
  const horizon = useDebounced(horizonInput, 250)
  const [days, setDays] = useState<string>('7')
  const [data, setData] = useState<VerificationData | null>(null)
  const [error, setError] = useState<string | null>(null)

  const subjects = ['house', ...(status?.people ?? [])]
  const chosen = subject || subjects[0] || ''

  useEffect(() => {
    if (!chosen) return
    let live = true
    setData(null)
    // Cleared per fetch, or one transient failure while dragging the slider
    // pinned this card on its message until the page was reloaded.
    setError(null)
    getVerification(chosen, horizon, Number(days))
      .then((v) => { if (live) setData(v) })
      .catch((e: Error) => { if (live) setError(e.message) })
    return () => { live = false }
  }, [chosen, horizon, days])

  // What the slider's own line says. It is about THIS horizon's record rather
  // than the chart's contents, so it stays useful while the fetch is in flight.
  const sliderSummary = (() => {
    if (horizonInput !== horizon) return 'reading…'
    const serving = status?.served_by?.[String(horizonInput)]
    if (serving === 'none') {
      const beaten = status?.best_baseline?.[String(horizonInput)]
      return beaten
        ? `not published — ${beaten.replace(/_/g, ' ')} beat the model here`
        : 'not published — no model has been trained for this horizon yet'
    }
    if (!data) return 'reading…'
    if (!data.available) return data.reason
    return `${data.scored} of ${data.slots} slots could be scored`
  })()

  return (
    <Card title="Was it right?">
      {/* `Row control`, not bare selects: a `Select` is wide enough to need a
          line of its own. */}
      <Row
        icon="people"
        control
        accent="blue"
        primary="Subject"
        trailing={
          <Select
            label="Subject"
            value={chosen}
            onChange={setSubject}
            options={subjects.length
              ? subjects.map((s) => ({ value: s, label: s === 'house' ? 'House' : pretty(s) }))
              : [{ value: '', label: '— no subjects configured —' }]}
          />
        }
      />
      <Row
        icon="clock"
        control
        accent="blue"
        primary="Window"
        secondary={keptFor(data?.available ? data.retention_days : null)}
        trailing={
          <Select label="Time window" value={days} onChange={setDays}
                  options={DAY_OPTIONS_RECENT} />
        }
      />

      <HorizonSlider value={horizonInput} onChange={setHorizonInput}
                     summary={sliderSummary} />

      {error ? <p className="empty">{error}</p>
        : !data ? <p className="empty">Loading…</p>
          : !data.available ? <p className="empty">{data.reason}</p>
            : (
              <Verification
                points={data.points}
                horizon={data.horizon_h}
                summary={data.summary}
                startLabel={absoluteTime(data.start)}
                endLabel={absoluteTime(data.stop)}
              />
            )}
    </Card>
  )
}
