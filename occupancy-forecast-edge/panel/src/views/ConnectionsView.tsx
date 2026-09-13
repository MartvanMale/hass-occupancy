import type { BrokerCheck, InfluxCheck, Status } from '../types'
import { Card } from '../components/Card'
import { Chip } from '../components/Chip'
import { Field } from '../components/Field'
import { Icon } from '../components/Icon'
import { Row } from '../components/Row'
import { Select } from '../components/Select'
import { Text } from '../components/Text'
import type { Connection } from './ConfigView'
import { count, relativeTime } from '../format'

/**
 * Where the add-on reaches the rest of the house: history in, entities out,
 * events back. One card per connection, each holding its own fields, its own
 * Check button and its own verdict -- a status box elsewhere on the page made
 * you match four green ticks to four subsystems by yourself.
 */

/* Short: the row is already titled "History source" and hinted with what each
   one means, so the option only has to name the branch. */
const SOURCES = [
  { value: 'store', label: 'Own archive' },
  { value: 'influx', label: 'InfluxDB' },
]

/** The staged Influx report. A stage is present only once it was reached, so
 *  the list itself shows how far it got. */
function InfluxReport({ check, checking }: { check: InfluxCheck | null; checking: boolean }) {
  if (checking) return <p className="empty">Checking…</p>
  if (!check) return null
  return (
    <>
      <ul className="stages">
        {check.stages.map((stage) => (
          <li key={stage.name} className={stage.ok ? 'pass' : 'fail'}>
            <Icon name={stage.ok ? 'check' : 'alert'} />
            <span>{stage.detail}</span>
          </li>
        ))}
      </ul>
      {check.hints.length > 0 && (
        <div className="hints">
          {check.hints.map((hint) => <span key={hint}>{hint}</span>)}
        </div>
      )}
    </>
  )
}

export interface ConnectionsViewProps {
  status: Status | null
  connection: Connection
  setConnection: (patch: Partial<Connection>) => void
  influxCheck: InfluxCheck | null
  influxChecking: boolean
  onCheckInflux: () => void
  brokerCheck: BrokerCheck | null
  brokerChecking: boolean
  onCheckBroker: () => void
}

export function ConnectionsView({
  status, connection, setConnection,
  influxCheck, influxChecking, onCheckInflux,
  brokerCheck, brokerChecking, onCheckBroker,
}: ConnectionsViewProps) {
  const onInflux = connection.source === 'influx'
  const listener = status?.listener
  const mqtt = status?.mqtt

  // On Influx there is no local archive to measure, so `status.history` carries
  // only a note -- the row count comes from Check connection instead.
  const archived = status && !onInflux && status.history.days != null
    ? <Chip icon="database" accent="aqua"
            label={`${Math.round(status.history.days)} days archived`} />
    : status?.influx_version
      ? <Chip icon="database" accent="aqua" label={status.influx_version} />
      : undefined

  return (
    <div className="cards wide">
      <Card
        title="Where history comes from"
        subtitle="Moved here from the add-on options in 0.4.0."
        badge={archived}
      >
        <Field
          icon="database"
          label="History source"
          hint="The archive builds up from today. An InfluxDB you already have
                brings months with it."
          control={
            <Select label="History source" value={connection.source}
                    onChange={(v) => setConnection({ source: v as 'store' | 'influx' })}
                    options={SOURCES} />
          }
        />
        {onInflux && (
          <>
            <Field
              label="URL" hint="Where the server answers, including the port."
              control={
                <Text label="InfluxDB URL" value={connection.influx_url}
                      placeholder="http://192.0.2.10:8086"
                      onChange={(v) => setConnection({ influx_url: v })} />
              }
            />
            <Field
              label="Organisation"
              hint="Required by the endpoint; ignored by a 1.x server."
              control={
                <Text label="InfluxDB organisation" value={connection.influx_org}
                      onChange={(v) => setConnection({ influx_org: v })} />
              }
            />
            <Field
              label="Bucket"
              hint="On 1.x this is database/retention-policy, such as
                    homeassistant/autogen."
              control={
                <Text label="InfluxDB bucket" value={connection.influx_bucket}
                      onChange={(v) => setConnection({ influx_bucket: v })} />
              }
            />
            <Field
              label="Token"
              hint={connection.influx_token_set
                ? 'A token is stored. Type to replace it; leave blank to keep it.'
                : 'On 1.x this is username:password, not an API token.'}
              control={
                <Text type="password" label="InfluxDB token"
                      value={connection.influx_token}
                      placeholder={connection.influx_token_set ? 'unchanged' : ''}
                      onChange={(v) => setConnection({ influx_token: v })} />
              }
            />
            <div className="actions">
              {/* type="button": this sits inside the page's form, and the
                  default would submit it. */}
              <button type="button" className="secondary" onClick={onCheckInflux}
                      disabled={influxChecking || !connection.influx_url}>
                <Icon name="refresh" />
                {influxChecking ? 'Checking…' : 'Check connection'}
              </button>
            </div>
            <InfluxReport check={influxCheck} checking={influxChecking} />
          </>
        )}
      </Card>

      <Card
        title="MQTT broker"
        subtitle="Where forecasts go out as entities."
        badge={mqtt && (
          <Chip icon={mqtt.connected ? 'mqtt-on' : 'mqtt-off'}
                accent={mqtt.connected ? 'aqua' : 'red'}
                label={mqtt.connected ? 'Connected' : 'Not connected'} />
        )}
      >
        <Field
          icon={connection.mqtt_host ? 'mqtt-on' : 'mqtt-off'}
          accent={connection.mqtt_host ? 'aqua' : 'grey'}
          label="Host"
          hint="Empty uses the broker Supervisor already knows about, which is the
                usual answer. Give a host and the port, credentials and TLS follow."
          control={
            <Text label="Broker host" value={connection.mqtt_host}
                  placeholder="Supervisor’s own broker"
                  onChange={(v) => setConnection({ mqtt_host: v })} />
          }
        />
        {connection.mqtt_host && (
          <>
            <Field
              label="Port"
              control={
                <Text label="Broker port" value={connection.mqtt_port}
                      onChange={(v) => setConnection({ mqtt_port: v })} />
              }
            />
            <Field
              label="Username"
              control={
                <Text label="Broker username" value={connection.mqtt_user}
                      onChange={(v) => setConnection({ mqtt_user: v })} />
              }
            />
            <Field
              label="Password"
              hint={connection.mqtt_password_set
                ? 'A password is stored. Type to replace it; leave blank to keep it.'
                : undefined}
              control={
                <Text type="password" label="Broker password"
                      value={connection.mqtt_password}
                      placeholder={connection.mqtt_password_set ? 'unchanged' : ''}
                      onChange={(v) => setConnection({ mqtt_password: v })} />
              }
            />
            <Field
              label="Use TLS"
              hint="A TLS-only broker reports the add-on as unavailable without this."
              control={
                <input
                  type="checkbox"
                  aria-label="Use TLS"
                  checked={connection.mqtt_ssl}
                  onChange={(e) => setConnection({ mqtt_ssl: e.target.checked })}
                />
              }
            />
          </>
        )}
        <div className="actions">
          <button type="button" className="secondary" onClick={onCheckBroker}
                  disabled={brokerChecking}>
            <Icon name="refresh" />
            {brokerChecking ? 'Checking…' : 'Check broker'}
          </button>
        </div>
        {brokerChecking ? (
          <p className="empty">Checking…</p>
        ) : brokerCheck && (
          <ul className="stages">
            <li className={brokerCheck.ok ? 'pass' : 'fail'}>
              <Icon name={brokerCheck.ok ? 'check' : 'alert'} />
              <span>{brokerCheck.detail}</span>
            </li>
          </ul>
        )}
      </Card>

      <Card
        title="Home Assistant events"
        subtitle="Nothing to configure. Re-predicts the moment somebody comes or goes."
        badge={listener && (
          <Chip icon={listener.connected ? 'listening' : 'deaf'}
                accent={listener.connected ? 'aqua' : 'orange'}
                label={listener.connected ? 'Subscribed' : 'Not subscribed'} />
        )}
      >
        {!listener ? (
          <p className="empty">Loading…</p>
        ) : (
          <>
            <Row
              icon={listener.connected ? 'listening' : 'deaf'}
              accent={listener.connected ? 'aqua' : 'orange'}
              primary={`Listening to ${count(listener.entities ?? 0)} entities`}
              secondary={listener.events
                ? `${count(listener.fired ?? 0)} of ${count(listener.events)} events
                   were worth re-predicting.${listener.last_event
                     ? ` Last ${relativeTime(listener.last_event)}.` : ''}`
                : 'No events seen yet.'}
            />
            <Row
              icon="clock"
              accent="grey"
              primary="Five-minute fallback poll"
              secondary="Runs whether or not the subscription is alive, so a dead
                one is slower, not broken."
            />
          </>
        )}
      </Card>
    </div>
  )
}
