import { useEffect, useState } from 'react'
import { getVerification } from '../api'
import type { Status, Verification as VerificationData } from '../types'
import { Card } from './Card'
import { HorizonSlider } from './HorizonSlider'
import { Row } from './Row'
import { Select } from './Select'
import { Verification } from './Verification'
import { useDebounced } from '../hooks'
import { absoluteTime } from '../format'

/**
 * "Was it right?" -- the same card on both tabs, each owning its own controls.
 *
 * It answers the one question nothing else here can. Every score on the Judge
 * step is rolling-origin cross-validation computed at TRAINING time over the
 * feature table: "how would a model fitted on folds [0,k) have scored on fold
 * k". That is a real number and it is not this one. It cannot see the nowcast
 * pin, a tracker that went quiet at 07:00, or the ship gate deciding a horizon
 * is not worth publishing -- and it is computed from rows the deployed add-on
 * may never have served.
 *
 * So the live Brier here and the backtest Brier there are different quantities,
 * and a gap between them is a finding about the deployment rather than a bug in
 * either. Saying so beside the backtest number is half of why this card exists.
 *
 * Rendered on both Overview and Data, with its own slider on each: on Overview
 * it sits under the 48-hour forecast, which is the natural order -- what the
 * forecast says, then what happened last time it said it. On Data it closes
 * step five, whose subtitle already asks the question.
 */

const DAY_OPTIONS = [
  { value: '1', label: 'last 24 hours' },
  { value: '7', label: 'last 7 days' },
  { value: '30', label: 'last 30 days' },
]

const pretty = (slug: string) => slug.charAt(0).toUpperCase() + slug.slice(1)

export function VerificationCard({ status, defaultHorizon = 6 }: {
  status: Status | null
  /** 6 h by default because that is the window the Lovelace card watches and
   *  the one a heating decision rests on. */
  defaultHorizon?: number
}) {
  const [subject, setSubject] = useState<string>('')
  // Debounced for the same reason the Data tab's slider is: dragging end to end
  // is otherwise 48 requests, each of which reads the archive and rebuilds a
  // grid, and this endpoint is deliberately not cached.
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
    /* No subtitle. That this is the only score measured on the serving path
       rather than at training time is the whole reason the card exists, and it
       is written out at length in DOCS.md under "Was it right?" -- which is
       where a reader who has not met the distinction should meet it, not on a
       card they see every time they open the panel. */
    <Card title="Was it right?">
      {/* The same control rows the Data tab's cards use, rather than a bare
          pair of selects: a `Select` is wide enough to need a line of its own,
          which is exactly what `Row control` is for. */}
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
        secondary={`Kept for ${data?.available ? data.retention_days : 30} days.`}
        trailing={
          <Select label="Time window" value={days} onChange={setDays}
                  options={DAY_OPTIONS} />
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
