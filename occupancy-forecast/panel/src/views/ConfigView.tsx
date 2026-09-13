import type { Candidates, FeatureDetail, Status } from '../types'
import { Card } from '../components/Card'
import { Chip } from '../components/Chip'
import { Row } from '../components/Row'
import { Icon } from '../components/Icon'
import { relativeTime } from '../format'
import { Field } from '../components/Field'
import { Select } from '../components/Select'

/**
 * Setup: what this installation is, and what it is doing about it. The state
 * lives in `App`, which owns the status poll -- see there.
 */

const round = (n: number) => Math.round(n).toString()

/** `feature_groups[*].detail` is polymorphic -- a list, a sentence, or the
 *  person->zone mapping an older config.json can still be showing. */
function formatDetail(detail: FeatureDetail): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.join(', ')
  return Object.entries(detail)
    .map(([person, zone]) => `${person} → ${zone}`)
    .join(', ')
}

/**
 * The exceptional states only, each as a banner and each only while it is true.
 * A green tick nobody needs to read cost a whole card; these interrupt instead.
 * Their steady-state twins live beside the thing they report on, on Connections.
 */
function Notices({ status }: { status: Status }) {
  const remaining = status.days_until_training ?? 0
  return (
    <>
      {remaining > 0 && (
        <div className="banner warn">
          <Icon name="collecting" />
          <div className="info">
            <div className="primary">
              Collecting history — {round(status.history.days ?? 0)} days so far
            </div>
            <div className="secondary">
              A model can first be validated in about {round(remaining)} more days.
              Nothing is published until then.
            </div>
          </div>
        </div>
      )}

      {/* A blocked thread never raises, so this hides behind every other green
          light on the page unless it is said out loud. */}
      {status.worker?.stalled && (
        <div className="banner">
          <Icon name="alert" />
          <div className="info">
            <div className="primary">
              The worker has been stuck in “{status.worker.stalled_in}” since{' '}
              {relativeTime(status.worker.stalled_since)}
            </div>
            <div className="secondary">
              Forecasts are not being updated. Thread stacks are in the add-on log;
              restarting the add-on clears it. Stalls since start: {status.worker.stalls}.
            </div>
          </div>
        </div>
      )}

      {/* The one case where a connection changes what ANOTHER tab means: what you
          set here is stored but is not reaching Home Assistant. */}
      {!status.mqtt.connected && (
        <div className="banner">
          <Icon name="mqtt-off" />
          <div className="info">
            <div className="primary">Not publishing — the MQTT broker is not connected.</div>
            <div className="secondary">
              Anything you save here is stored, but no entities are being updated.
              Fix it on the Connections tab.
            </div>
          </div>
        </div>
      )}
    </>
  )
}


function FeatureRows({ groups }: { groups: Status['feature_groups'] }) {
  const names = Object.keys(groups)
  if (names.length === 0) return <p className="empty">Nothing configured yet.</p>
  return (
    // Two columns on a card wide enough for them, one on a phone. Keyed off the
    // CARD's width, so the same rule works here and in the Data walkthrough.
    <div className="cols">
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
    </div>
  )
}

/** The form's connection half. One object rather than ten props and ten
 *  setters, which is also what keeps `App`'s wiring readable. */
export interface Connection {
  source: 'store' | 'influx'
  influx_url: string
  influx_org: string
  influx_bucket: string
  /** Typed now, never loaded: the server does not serve a stored secret back. */
  influx_token: string
  influx_token_set: boolean
  mqtt_host: string
  mqtt_port: string
  mqtt_user: string
  mqtt_password: string
  mqtt_password_set: boolean
  mqtt_ssl: boolean
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
  retention,
  togglePerson, toggleZone, setHouse, setHoliday, setDaySchedule,
  setDeparture, setArrival, setMinHours, setRetention,
}: ConfigViewProps) {
  return (
    <>
      {status && <Notices status={status} />}

      {/* Read-only, and quiet ground so it does not read as a peer of the form
          below it -- it is what the form ADDS UP TO. */}
      <Card
        quiet
        title="What the model will train on"
        subtitle="A consequence of the choices below. Read-only."
      >
        {status ? <FeatureRows groups={status.feature_groups} />
                : <p className="empty">Loading…</p>}
      </Card>

      <div className="cards wide">
        {/* One column: two lists of the same kind of thing, read down rather than
            across, so the three grid items are lists / signals / numbers. */}
        <div className="stack">
          <Card title="People" subtitle="Required. At least one.">
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
            subtitle="Work, school, the supermarket. Home is excluded."
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
        </div>

        {/* Three cards wrapping one select each was 150px of card per 42px of
            control, and they are all the same kind of thing: an optional Home
            Assistant entity the model may borrow. */}
        <Card
          title="Optional signals"
          optional
          subtitle="Entities the model may borrow. Skip any you do not have."
        >
          <Field
            icon="house"
            label="Person group"
            hint="Unset: occupied whenever anyone is home."
            control={
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
          {candidates && candidates.countries.length === 0 ? (
            <p className="empty">
              The holidays package is unavailable, so no calendar can be picked.
              is_holiday will be 0 everywhere.
            </p>
          ) : (
            <Field
              icon="calendar"
              label="Holiday calendar"
              hint="Applies at the next training run."
              control={
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
          {candidates && candidates.schedules.length === 0 ? (
            <p className="empty">
              No schedule entities exist here, so there is nothing to shade by. The
              chart works without it.
            </p>
          ) : (
            <Field
              icon="clock"
              label="Night shading"
              hint="Display only. Greys out hours outside it."
              control={
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

        {/* Numbers you tune after the fact, and none of them needs a retrain --
            which is what separates them from the two cards above. */}
        <Card
          title="Tuning"
          subtitle="Takes effect immediately. Nothing here needs a retrain."
        >
          <p className="subhead">When a crossing counts</p>
          <Field
            icon="target"
            label="Away below"
            hint="Chance of being home."
            control={
              <Select label="Away cut" value={departure} onChange={setDeparture}
                      options={CUTS} />
            }
          />
          <Field
            icon="target"
            label="Home at or above"
            hint="Keep at or above the away cut."
            control={
              <Select label="Home cut" value={arrival} onChange={setArrival}
                      options={CUTS} />
            }
          />
          <Field
            icon="clock"
            label="and stays there for"
            control={
              <Select label="Minimum run" value={minHours} onChange={setMinHours}
                      options={RUNS} />
            }
          />

          <p className="subhead">Forecast record</p>
          <Field
            icon="database"
            label="Keep each forecast for"
            hint="0 keeps everything. Shortening it deletes older rows permanently."
            control={
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

    </>
  )
}
