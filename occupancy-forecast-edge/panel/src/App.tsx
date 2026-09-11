import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { getCandidates, getSettings, getStatus, saveConfig } from './api'
import type { Candidates, Settings, Status } from './types'
import { Icon, Shape } from './components/Icon'
import { ConfigView } from './views/ConfigView'
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
  { id: 'data', label: 'Data', icon: 'chart' },
] as const

type View = (typeof VIEWS)[number]['id']

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
    setLoaded(true)
  }, [])

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
            loaded={loaded}
            saving={saving}
            saved={saved}
            error={error}
            onSubmit={onSubmit}
            togglePerson={togglePerson}
            toggleZone={toggleZone}
            setHouse={setHouse}
            setHoliday={setHoliday}
            setDeparture={setDeparture}
            setArrival={setArrival}
            setMinHours={setMinHours}
            setRetention={setRetention}
          />
        ) : view === 'data' ? (
          <DataView status={status} />
        ) : (
          <OverviewView status={status} refreshStatus={refreshStatus} />
        )}
      </div>
    </div>
  )
}
