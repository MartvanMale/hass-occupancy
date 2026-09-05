import type { FormEvent } from 'react'
import type { Candidates, FeatureDetail, Status } from '../types'
import { Card } from '../components/Card'
import { Chip } from '../components/Chip'
import { Row } from '../components/Row'
import { Icon } from '../components/Icon'
import { relativeTime } from '../format'
import { Select } from '../components/Select'

/**
 * Setup: what this installation is, and what it is doing about it.
 *
 * Everything here was `App.tsx` before there was a second view. The split is
 * along the question being asked -- this one answers "is it configured right",
 * the Data tab answers "what is it actually eating" -- and the state still lives
 * in `App`, because the status poll drives the header too and a view that owned
 * it would restart it on every tab switch.
 */

const round = (n: number) => Math.round(n).toString()
const count = (n: number) => n.toLocaleString()

/**
 * `feature_groups[*].detail` is polymorphic -- a list of entity ids or a
 * sentence, plus the person->zone mapping an older config.json can still be
 * showing before its first save. The page this replaced ran Python's `str()`
 * over it and rendered `['person.alice']` on screen, brackets, quotes and all.
 */
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

      {/* Red, and shown only when it is true, because a stalled worker is the
          one failure that hides behind every other green light on this page:
          MQTT stays connected, the listener stays subscribed, and last_error
          stays null, because a blocked thread is not a raising one. It went
          unnoticed for 11.5 hours once. */}
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

      {/* Orange, not red, and the distinction is the reason the row exists: a
          dead trigger subscription means the five-minute poll carries on, so
          this is slower rather than broken. It would otherwise be invisible. */}
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
}

/**
 * Discrete options rather than a number input.
 *
 * The panel has no styled numeric control, a free-text one needs a
 * partial-input parse dance ("0." is not a number), and impossible values then
 * cannot be typed at all. The cost is that 0.42 is unreachable, which is fine:
 * a 1% distinction on a 48-point hourly curve is not a real one. The server
 * still validates, because the API is reachable without this page.
 */
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
  loaded, saving, saved, error,
  onSubmit, togglePerson, toggleZone, setHouse, setHoliday, setDaySchedule,
  setDeparture, setArrival, setMinHours,
}: ConfigViewProps) {
  return (
    <>
      <div className="cards">
        <Card title="Status">
          {status ? <StatusRows status={status} /> : <p className="empty">Loading…</p>}
        </Card>

        {/* "What is serving each horizon" and "Training" were here and are now
            on Now. Setup is a page you fill in once and leave; those two change
            on their own and are read against the forecast, not against the
            settings below. Status stays, because what it reports on is whether
            the configuration on this page is actually working. */}
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
        </div>

        <div className="actions">
          <button type="submit" disabled={!loaded || saving}>
            <Icon name="save" />
            {saving ? 'Saving…' : 'Save'}
          </button>
          <span className={saved ? 'saved on' : 'saved'}>Saved</span>
          {error && <span className="error">{error}</span>}
        </div>
      </form>
    </>
  )
}
