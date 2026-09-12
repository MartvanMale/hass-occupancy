import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { checkBroker, checkInflux, getCandidates, getSettings, getStatus, saveConfig } from './api'
import type { BrokerCheck, Candidates, InfluxCheck, Settings, Status } from './types'
import { Icon, Shape } from './components/Icon'
import { SaveDock } from './components/SaveDock'
import { ConfigView, type Connection } from './views/ConfigView'
import { ConnectionsView } from './views/ConnectionsView'
import { DataView } from './views/DataView'
import { OverviewView } from './views/OverviewView'

/** Status is polled; the configuration is not, because only the form on this
 *  page changes it. Ten seconds is fast enough to watch MQTT reconnect. */
const POLL_MS = 10_000

/** A running train has an elapsed time that should tick and a finish worth
 *  noticing promptly; it lasts minutes, so the faster poll costs nothing. */
const POLL_MS_TRAINING = 3_000

/**
 * The panel's only navigation. Deliberately not persisted: a hash pushes entries
 * into the *top-level* history from inside an iframe, and `localStorage` in an
 * Ingress iframe is Home Assistant's own, shared by the stable and edge add-ons.
 */
const VIEWS = [
  // Overview first, and the default: it is the one people keep open. Setup is
  // a thing you do once.
  { id: 'overview', label: 'Overview', icon: 'house' },
  { id: 'config', label: 'Setup', icon: 'tune' },
  // After Setup, not before it: this is the tab you open twice -- the day you
  // install and the day something breaks -- so it must not be the landing page.
  { id: 'connections', label: 'Connections', icon: 'mqtt-on' },
  { id: 'data', label: 'Data', icon: 'chart' },
] as const

type View = (typeof VIEWS)[number]['id']

/** Only ever on screen before the first GET answers, so the defaults just have
 *  to be harmless -- `applySettings` replaces the whole object. */
const EMPTY_CONNECTION: Connection = {
  source: 'store',
  influx_url: '', influx_org: '', influx_bucket: 'homeassistant',
  influx_token: '', influx_token_set: false,
  mqtt_host: '', mqtt_port: '1883', mqtt_user: '',
  mqtt_password: '', mqtt_password_set: false, mqtt_ssl: false,
}

export function App() {
  const [view, setView] = useState<View>('overview')
  const [status, setStatus] = useState<Status | null>(null)
  const [candidates, setCandidates] = useState<Candidates | null>(null)

  // The form's own state, seeded once from GET /api/config and kept apart from
  // `status`: the poll below must never overwrite a half-finished edit.
  const [people, setPeople] = useState<string[]>([])
  const [zones, setZones] = useState<string[]>([])
  const [house, setHouse] = useState<string>('')
  const [holiday, setHoliday] = useState<string>('')
  const [daySchedule, setDaySchedule] = useState<string>('')
  // Strings, because `Select` speaks strings. Coerced back on submit.
  const [departure, setDeparture] = useState<string>('0.5')
  const [arrival, setArrival] = useState<string>('0.5')
  const [minHours, setMinHours] = useState<string>('2')
  const [retention, setRetention] = useState<string>('30')
  const [loaded, setLoaded] = useState(false)

  // The connection half, grouped: ten more `useState` pairs threaded through
  // ConfigView's props would be the larger half of this file.
  const [connection, setWholeConnection] = useState<Connection>(EMPTY_CONNECTION)
  const setConnection = useCallback(
    (patch: Partial<Connection>) => setWholeConnection((c) => ({ ...c, ...patch })),
    [],
  )
  const [influxCheck, setInfluxCheck] = useState<InfluxCheck | null>(null)
  const [influxChecking, setInfluxChecking] = useState(false)
  const [brokerCheck, setBrokerCheck] = useState<BrokerCheck | null>(null)
  const [brokerChecking, setBrokerChecking] = useState(false)
  // What the server last gave us, kept so the form can tell whether it differs.
  // One form backs Setup and Connections, so a save from either writes both --
  // which is only honest if the button says there is something to write.
  const [loadedSettings, setLoadedSettings] = useState<Settings | null>(null)

  // Typed, so they can be mid-edit. An empty field must not reach the server as
  // Number('') === 0, which is the one value that means something else.
  const retentionOk = /^\d+$/.test(retention.trim())
  const portOk = /^\d+$/.test(connection.mqtt_port.trim())
    && Number(connection.mqtt_port) >= 1 && Number(connection.mqtt_port) <= 65535

  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const applySettings = useCallback((s: Settings) => {
    setPeople(s.people)
    setZones(s.zones)
    setHouse(s.house_entity ?? '')
    setHoliday(s.holiday_country ?? '')
    setDaySchedule(s.day_schedule ?? '')
    setDeparture(s.departure_threshold.toFixed(2))
    setArrival(s.arrival_threshold.toFixed(2))
    setMinHours(String(s.crossing_min_hours))
    setRetention(String(s.forecast_retention_days))
    // The two secrets are deliberately not seeded -- the server never sends
    // them. An empty box means "keep what is stored"; the `_set` flags say so.
    setWholeConnection({
      source: s.source,
      influx_url: s.influx_url,
      influx_org: s.influx_org,
      influx_bucket: s.influx_bucket,
      influx_token: '',
      influx_token_set: s.influx_token_set,
      mqtt_host: s.mqtt_host,
      mqtt_port: String(s.mqtt_port),
      mqtt_user: s.mqtt_user,
      mqtt_password: '',
      mqtt_password_set: s.mqtt_password_set,
      mqtt_ssl: s.mqtt_ssl,
    })
    setLoadedSettings(s)
    setLoaded(true)
  }, [])

  // Compared field by field against what was loaded. A typed secret always
  // counts: the form cannot compare it with something it was never given.
  const dirty = loadedSettings != null && (
    connection.influx_token !== '' || connection.mqtt_password !== ''
    || people.join() !== loadedSettings.people.join()
    || zones.join() !== loadedSettings.zones.join()
    || house !== (loadedSettings.house_entity ?? '')
    || holiday !== (loadedSettings.holiday_country ?? '')
    || daySchedule !== (loadedSettings.day_schedule ?? '')
    || Number(departure) !== loadedSettings.departure_threshold
    || Number(arrival) !== loadedSettings.arrival_threshold
    || Number(minHours) !== loadedSettings.crossing_min_hours
    || Number(retention) !== loadedSettings.forecast_retention_days
    || connection.source !== loadedSettings.source
    || connection.influx_url !== loadedSettings.influx_url
    || connection.influx_org !== loadedSettings.influx_org
    || connection.influx_bucket !== loadedSettings.influx_bucket
    || connection.mqtt_host !== loadedSettings.mqtt_host
    || Number(connection.mqtt_port) !== loadedSettings.mqtt_port
    || connection.mqtt_user !== loadedSettings.mqtt_user
    || connection.mqtt_ssl !== loadedSettings.mqtt_ssl
  )

  useEffect(() => {
    let live = true
    getCandidates()
      .then((c) => { if (live) setCandidates(c) })
      .catch((e: Error) => { if (live) setError(e.message) })
    getSettings()
      .then((s) => { if (live) applySettings(s) })
      .catch((e: Error) => { if (live) setError(e.message) })
    return () => { live = false }
  }, [applySettings])

  // The "saved" tick's timer, so two saves inside two seconds do not race and
  // the timer is cleared with the page.
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => { if (savedTimer.current) clearTimeout(savedTimer.current) }, [])

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await getStatus())
    } catch {
      /* a poll that fails leaves the last good status on screen */
    }
  }, [])

  // The poll stays here rather than in ConfigView: the header reads `status`
  // too, and a view that owned it would restart it on every tab switch.
  const training = status?.training_in_progress ?? false
  useEffect(() => {
    let live = true
    const tick = () => {
      getStatus()
        .then((s) => { if (live) setStatus(s) })
        .catch(() => {})
    }
    tick()
    const id = setInterval(tick, training ? POLL_MS_TRAINING : POLL_MS)
    return () => { live = false; clearInterval(id) }
  }, [training])

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setSaving(true)
    setError(null)
    try {
      await saveConfig({
        people,
        zones,
        house_entity: house || null,
        // Omitted when the holidays package could not be loaded, which leaves
        // the stored calendar alone rather than clearing it.
        ...(candidates?.countries.length ? { holiday_country: holiday } : {}),
        // Empty means "no shading", a real choice: sent as null rather than
        // omitted, or the setting could never be cleared once set.
        day_schedule: daySchedule || null,
        departure_threshold: Number(departure),
        arrival_threshold: Number(arrival),
        crossing_min_hours: Number(minHours),
        forecast_retention_days: Number(retention),
        source: connection.source,
        influx_url: connection.influx_url,
        influx_org: connection.influx_org,
        influx_bucket: connection.influx_bucket,
        mqtt_host: connection.mqtt_host,
        mqtt_port: Number(connection.mqtt_port),
        mqtt_user: connection.mqtt_user,
        mqtt_ssl: connection.mqtt_ssl,
        // Sent only when typed into: the form was never given the stored value,
        // so sending an empty box would blank the password it cannot see.
        ...(connection.influx_token ? { influx_token: connection.influx_token } : {}),
        ...(connection.mqtt_password ? { mqtt_password: connection.mqtt_password } : {}),
      })
      setSaved(true)
      if (savedTimer.current) clearTimeout(savedTimer.current)
      savedTimer.current = setTimeout(() => setSaved(false), 2000)
      // Saving rebuilds the runtime from the new settings, so both of these are
      // now stale -- refetch rather than reload the page.
      getSettings().then(applySettings).catch(() => {})
      getStatus().then(setStatus).catch(() => {})
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  async function onCheckInflux() {
    setInfluxChecking(true)
    setInfluxCheck(null)
    try {
      setInfluxCheck(await checkInflux({
        influx_url: connection.influx_url,
        influx_org: connection.influx_org,
        influx_bucket: connection.influx_bucket,
        // Blank means "the token already stored", so a saved one is testable.
        influx_token: connection.influx_token,
      }))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setInfluxChecking(false)
    }
  }

  async function onCheckBroker() {
    setBrokerChecking(true)
    setBrokerCheck(null)
    try {
      setBrokerCheck(await checkBroker({
        mqtt_host: connection.mqtt_host,
        mqtt_port: Number(connection.mqtt_port) || 1883,
        mqtt_user: connection.mqtt_user,
        mqtt_ssl: connection.mqtt_ssl,
        mqtt_password: connection.mqtt_password,
      }))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBrokerChecking(false)
    }
  }

  const togglePerson = (entity: string, on: boolean) =>
    setPeople((current) =>
      on ? [...current, entity] : current.filter((p) => p !== entity),
    )

  const toggleZone = (entity: string, on: boolean) =>
    setZones((current) =>
      on ? [...current, entity] : current.filter((z) => z !== entity),
    )

  return (
    <div className="wrap">
      <header className="head">
        <Shape name="logo" accent="aqua" />
        <div>
          {/* No fallback string: this name is all that separates two add-ons'
              identical-looking panels, so it comes from the server or not at all. */}
          <h1>{status?.display_name ?? ' '}</h1>
          <p className="sub">Who is home, and who is coming home.</p>
        </div>

        {/* Three buttons, so no roving tabindex: they are already keyboard
            reachable in order, and the arrow-key pattern would be a regression. */}
        <nav className="tabs" role="tablist" aria-label="Sections">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              type="button"
              role="tab"
              id={`tab-${v.id}`}
              className="tab"
              aria-selected={view === v.id}
              aria-controls={`panel-${v.id}`}
              onClick={() => setView(v.id)}
            >
              <Icon name={v.icon} />
              {v.label}
            </button>
          ))}
        </nav>
      </header>

      <div role="tabpanel" id={`panel-${view}`} aria-labelledby={`tab-${view}`}>
        {view === 'config' || view === 'connections' ? (
          /* ONE form over both tabs, so there is one Save and one dirty state:
             two forms would each quietly write the other tab's fields. */
          <form onSubmit={onSubmit}>
            {view === 'config' ? (
              <ConfigView
                status={status}
                candidates={candidates}
                people={people}
                zones={zones}
                house={house}
                holiday={holiday}
                daySchedule={daySchedule}
                setDaySchedule={setDaySchedule}
                departure={departure}
                arrival={arrival}
                minHours={minHours}
                retention={retention}
                togglePerson={togglePerson}
                toggleZone={toggleZone}
                setHouse={setHouse}
                setHoliday={setHoliday}
                setDeparture={setDeparture}
                setArrival={setArrival}
                setMinHours={setMinHours}
                setRetention={setRetention}
              />
            ) : (
              <ConnectionsView
                status={status}
                connection={connection}
                setConnection={setConnection}
                influxCheck={influxCheck}
                influxChecking={influxChecking}
                onCheckInflux={onCheckInflux}
                brokerCheck={brokerCheck}
                brokerChecking={brokerChecking}
                onCheckBroker={onCheckBroker}
              />
            )}
            <SaveDock
              loaded={loaded}
              saving={saving}
              saved={saved}
              dirty={dirty}
              complaint={error ?? (retentionOk
                ? portOk ? null : 'The broker port must be a whole number between 1 and 65535.'
                : 'Days to keep must be a whole number of days, 0 or more.')}
            />
          </form>
        ) : view === 'data' ? (
          <DataView status={status} />
        ) : (
          <OverviewView status={status} refreshStatus={refreshStatus} />
        )}
      </div>
    </div>
  )
}
