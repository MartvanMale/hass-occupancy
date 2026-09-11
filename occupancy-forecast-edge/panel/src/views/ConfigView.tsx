import type { FormEvent } from 'react'
import type { Candidates, FeatureDetail, Status } from '../types'
import { Card } from '../components/Card'
import { Chip } from '../components/Chip'
import { Row } from '../components/Row'
import { Icon } from '../components/Icon'
import { relativeTime } from '../format'
import { Select } from '../components/Select'

/**
 * Setup: what this installation is, and what it is doing about it. The state
 * lives in `App`, which owns the status poll -- see there.
 */

const round = (n: number) => Math.round(n).toString()
const count = (n: number) => n.toLocaleString()

/** `feature_groups[*].detail` is polymorphic -- a list, a sentence, or the
 *  person->zone mapping an older config.json can still be showing. */
function formatDetail(detail: FeatureDetail): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.join(', ')
  return Object.entries(detail)
    .map(([person, zone]) => `${person} → ${zone}`)
    .join(', ')
}

function StatusRows({ status }: { status: Status }) {
  const { history, mqtt, listener } = status
  const days = history.days ?? 0
  const remaining = status.days_until_training ?? 0
  return (
    <>
      {remaining > 0 ? (
        <Row
          icon="collecting"
          accent="orange"
          primary={`Collecting history — ${round(days)} days so far`}
          secondary={`A model can first be validated in about ${round(remaining)} more days.
            Nothing is published until then.`}
        />
      ) : (
        <Row
          icon="collected"
          accent="aqua"
          primary={`${round(days)} days of history`}
          secondary={`${count(history.rows ?? 0)} state changes.`}
        />
      )}

      {mqtt.connected ? (
        <Row
          icon="mqtt-on"
          accent="aqua"
          primary="MQTT connected"
          secondary="Forecasts are being published as entities."
        />
      ) : (
        <Row
          icon="mqtt-off"
          accent="red"
          primary="MQTT is not connected"
          secondary={`Entities will not appear until it is. ${mqtt.error ?? ''}`}
        />
      )}

      {/* Red, and shown only when true: a stalled worker hides behind every other
          green light on this page, because a blocked thread is not a raising one. */}
      {status.worker?.stalled && (
        <Row
          icon="alert"
          accent="red"
          primary={`The worker has been stuck in "${status.worker.stalled_in}" since ${
            relativeTime(status.worker.stalled_since)}`}
          secondary={`Forecasts are not being updated. Thread stacks are in the add-on
            log; restarting the add-on clears it. Stalls since start: ${
            status.worker.stalls}.`}
        />
      )}

      {/* Orange, not red: with the trigger subscription dead the five-minute poll
          carries on, so this is slower rather than broken. */}
      {listener.connected ? (
        <Row
          icon="listening"
          accent="aqua"
          primary={`Listening to ${listener.entities ?? 0} entities`}
          secondary={`${count(listener.fired ?? 0)} of ${count(listener.events ?? 0)} events were
            worth re-predicting. Last: ${listener.last_event ?? 'none yet'}.`}
        />
      ) : (
        <Row
          icon="deaf"
          accent="orange"
          primary="Not subscribed to Home Assistant triggers"
          secondary={`Forecasts will still publish every five minutes, just not the moment
            somebody comes or goes. ${listener.last_error ?? ''}`}
        />
      )}
    </>
  )
}

function FeatureRows({ groups }: { groups: Status['feature_groups'] }) {
  const names = Object.keys(groups)
  if (names.length === 0) return <p className="empty">Nothing configured yet.</p>
  return (
    <>
      {names.map((name) => {
        const active = groups[name]!.active
        return (
          <Row
            key={name}
            icon={active ? 'check' : 'minus'}
            accent={active ? 'aqua' : 'grey'}
            primary={name.replace(/_/g, ' ')}
            secondary={formatDetail(groups[name]!.detail)}
            muted={!active}
            trailing={
              <Chip
                label={active ? 'active' : 'not available'}
                icon={active ? 'check' : 'minus'}
                accent={active ? 'aqua' : 'grey'}
              />
            }
          />
        )
      })}
    </>
  )
}

export interface ConfigViewProps {
  status: Status | null
  candidates: Candidates | null
  people: string[]
  zones: string[]
  house: string
  holiday: string
  daySchedule: string
  setDaySchedule: (v: string) => void
  departure: string
  arrival: string
  minHours: string
  retention: string
  loaded: boolean
  saving: boolean
  saved: boolean
  error: string | null
  onSubmit: (e: FormEvent) => void
  togglePerson: (entity: string, on: boolean) => void
  toggleZone: (entity: string, on: boolean) => void
  setHouse: (value: string) => void
  setHoliday: (value: string) => void
  setDeparture: (value: string) => void
  setArrival: (value: string) => void
  setMinHours: (value: string) => void
  setRetention: (value: string) => void
}

/** Discrete options rather than a number input: 0.42 is unreachable, which is
 *  fine on a 48-point hourly curve. The server still validates. */
const CUTS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7].map((v) => ({
  value: v.toFixed(2),
  label: `${Math.round(v * 100)} %`,
}))

const RUNS = [1, 2, 3, 4, 6].map((h) => ({
  value: String(h),
  label: h === 1 ? '1 hour (any single hour)' : `${h} hours`,
}))


export function ConfigView({
  status, candidates,
  people, zones, house, holiday, daySchedule, departure, arrival, minHours,
  retention, loaded, saving, saved, error,
  onSubmit, togglePerson, toggleZone, setHouse, setHoliday, setDaySchedule,
  setDeparture, setArrival, setMinHours, setRetention,
}: ConfigViewProps) {
  // Typed, so it can be mid-edit. An empty field must not reach the server as
  // Number('') === 0, which is the one value that means something else.
  const retentionOk = /^\d+$/.test(retention.trim())
  const complaint = error ?? (retentionOk
    ? null
    : 'Days to keep must be a whole number of days, 0 or more.')
  return (
    <>
      <div className="cards">
        <Card title="Status">
          {status ? <StatusRows status={status} /> : <p className="empty">Loading…</p>}
        </Card>

        <Card
          title="What this installation has"
          subtitle="A missing signal is not an error — the forecast is just less sharp."
        >
          {status ? <FeatureRows groups={status.feature_groups} /> : <p className="empty">Loading…</p>}
        </Card>
      </div>

      <form onSubmit={onSubmit}>
        <div className="cards">
          <Card title="People" subtitle="Occupancy is the one thing this cannot run without.">
            {!candidates ? (
              <p className="empty">Loading…</p>
            ) : candidates.people.length === 0 ? (
              <p className="empty">None found on this Home Assistant.</p>
            ) : (
              candidates.people.map((person) => (
                <Row
                  key={person.entity_id}
                  as="label"
                  icon="people"
                  accent="blue"
                  primary={person.name}
                  secondary={person.entity_id}
                  trailing={
                    <input
                      type="checkbox"
                      checked={people.includes(person.entity_id)}
                      onChange={(e) => togglePerson(person.entity_id, e.target.checked)}
                    />
                  }
                />
              ))
            )}
          </Card>

          <Card
            title="Zones"
            optional
            subtitle="Anywhere worth knowing about — work, school, the supermarket.
              Home is excluded."
          >
            {!candidates ? (
              <p className="empty">Loading…</p>
            ) : candidates.zones.length === 0 ? (
              <p className="empty">No zones in Home Assistant yet.</p>
            ) : (
              candidates.zones.map((zone) => (
                <Row
                  key={zone.entity_id}
                  as="label"
                  icon="marker"
                  accent="blue"
                  primary={zone.name}
                  secondary={zone.entity_id}
                  trailing={
                    <input
                      type="checkbox"
                      checked={zones.includes(zone.entity_id)}
                      onChange={(e) => toggleZone(zone.entity_id, e.target.checked)}
                    />
                  }
                />
              ))
            )}
          </Card>

          <Card title="The house" optional subtitle="A person group, if you have one.">
            <Row
              icon="house"
              control
              accent="blue"
              primary="Person group"
              secondary="Left unset, the house counts as occupied whenever anyone is home."
              trailing={
                <Select
                  label="Person group"
                  value={house}
                  onChange={setHouse}
                  options={[
                    { value: '', label: '— derive it from the people —' },
                    ...(candidates?.groups ?? []).map((g) => ({ value: g.entity_id, label: g.name })),
                  ]}
                />
              }
            />
          </Card>

          <Card
            title="Holiday calendar"
            optional
            subtitle="Which public holidays this household keeps — not necessarily the
              country you live in."
          >
            {candidates && candidates.countries.length === 0 ? (
              <p className="empty">
                The holidays package is unavailable, so no calendar can be picked.
                is_holiday will be 0 everywhere.
              </p>
            ) : (
              <Row
                icon="calendar"
                control
                accent="blue"
                primary="Public holidays"
                secondary="Takes effect at the next training run."
                trailing={
                  <Select
                    label="Holiday calendar"
                    searchable
                    value={holiday}
                    onChange={setHoliday}
                    options={[
                      { value: '', label: '— none —' },
                      ...(candidates?.countries ?? []).map((c) => ({ value: c.code, label: c.name })),
                    ]}
                  />
                }
              />
            )}
          </Card>

          <Card
            title="Night shading"
            optional
            subtitle="Greys out the hours outside a schedule you already keep.
              Display only — no feature, no model, no entity."
          >
            {candidates && candidates.schedules.length === 0 ? (
              <p className="empty">
                No schedule entities exist here, so there is nothing to shade by. The
                chart works without it.
              </p>
            ) : (
              <Row
                icon="clock"
                control
                accent="blue"
                primary="Waking hours"
                secondary="Read from the schedule's own last week."
                trailing={
                  <Select
                    label="Day schedule"
                    searchable
                    value={daySchedule}
                    onChange={setDaySchedule}
                    options={[
                      { value: '', label: '— none —' },
                      ...(candidates?.schedules ?? []).map(
                        (e) => ({ value: e.entity_id, label: e.name })),
                    ]}
                  />
                }
              />
            )}
          </Card>

          <Card
            title="When a crossing counts"
            subtitle="How far the curve has to move, and for how long, before the
              countdown changes. No retrain needed."
          >
            <Row
              icon="target"
              control
              accent="blue"
              primary="Away when the chance of being home falls below"
              trailing={
                <Select label="Away cut" value={departure} onChange={setDeparture}
                        options={CUTS} />
              }
            />
            <Row
              icon="target"
              control
              accent="blue"
              primary="Home when it reaches"
              secondary="Keep this at or above the away cut."
              trailing={
                <Select label="Home cut" value={arrival} onChange={setArrival}
                        options={CUTS} />
              }
            />
            <Row
              icon="clock"
              control
              accent="blue"
              primary="and stays there for"
              secondary="A single hour on the wrong side is a wobble, not a departure."
              trailing={
                <Select label="Minimum run" value={minHours} onChange={setMinHours}
                        options={RUNS} />
              }
            />
          </Card>

          <Card
            title="Forecast record"
            subtitle="How long to keep what was published, for the “Was it right?”
              chart. Nothing is trained on it and no entity reads it."
          >
            <Row
              icon="database"
              control
              accent="blue"
              primary="Keep each forecast for"
              secondary="0 keeps everything. Shortening this deletes the older
                rows on the next cycle, and they cannot be rebuilt."
              trailing={
                <span className="days">
                  <input
                    type="number"
                    min={0}
                    step={1}
                    inputMode="numeric"
                    aria-label="Days of forecasts to keep"
                    value={retention}
                    onChange={(e) => setRetention(e.target.value)}
                  />
                  days
                </span>
              }
            />
          </Card>
        </div>

        <div className="actions">
          <button type="submit" disabled={!loaded || saving || !retentionOk}>
            <Icon name="save" />
            {saving ? 'Saving…' : 'Save'}
          </button>
          <span className={saved ? 'saved on' : 'saved'}>Saved</span>
          {complaint && <span className="error">{complaint}</span>}
        </div>
      </form>
    </>
  )
}
