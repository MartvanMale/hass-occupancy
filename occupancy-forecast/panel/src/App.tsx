import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { getCandidates, getSettings, getStatus, saveConfig } from './api'
import type { Candidates, Settings, Status } from './types'
import { Icon, Shape } from './components/Icon'
import { ConfigView } from './views/ConfigView'
import { DataView } from './views/DataView'
import { OverviewView } from './views/OverviewView'

/** Home Assistant is polled for status; the configuration is not, because the
 *  only thing that changes it is the form on this page. Ten seconds is fast
 *  enough to watch MQTT reconnect and slow enough to be invisible. */
const POLL_MS = 10_000

/** While a train is running there is an elapsed time on screen that should tick
 *  and a finish worth noticing promptly. It is minutes, not hours, so a faster
 *  poll for its duration costs nothing. */
const POLL_MS_TRAINING = 3_000

/**
 * The two questions the panel answers, and the only navigation it has.
 *
 * Deliberately not persisted. A hash would push entries into the *top-level*
 * history from inside an iframe, so Back would step through tab changes instead
 * of leaving the Home Assistant page -- and the Ingress URL carries a rotating
 * token, so the deep link a hash would buy cannot be shared anyway.
 * `localStorage` in an Ingress iframe is Home Assistant's own storage, shared by
 * the stable and edge add-ons, so an unnamespaced key there is two add-ons
 * quietly fighting over one slot. Landing on Setup after a reload is fine.
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
  // Overview is the landing tab. Setup is a thing you do once; this is the
  // thing people come back to.
  const [view, setView] = useState<View>('overview')
  const [status, setStatus] = useState<Status | null>(null)
  const [candidates, setCandidates] = useState<Candidates | null>(null)

  // The form's own state, seeded from GET /api/config once and owned by the
  // page thereafter. Kept apart from `status` on purpose: the poll below must
  // never overwrite a half-finished edit.
  const [people, setPeople] = useState<string[]>([])
  const [zones, setZones] = useState<string[]>([])
  const [house, setHouse] = useState<string>('')
  const [holiday, setHoliday] = useState<string>('')
  const [daySchedule, setDaySchedule] = useState<string>('')
  // Strings, because `Select` speaks strings. Coerced back on submit.
  const [departure, setDeparture] = useState<string>('0.5')
  const [arrival, setArrival] = useState<string>('0.5')
  const [minHours, setMinHours] = useState<string>('2')
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
        // Empty means "no shading", which is a real choice, so it is sent as
        // null rather than omitted -- omitting would make the setting
        // impossible to clear once set.
        day_schedule: daySchedule || null,
        departure_threshold: Number(departure),
        arrival_threshold: Number(arrival),
        crossing_min_hours: Number(minHours),
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
          {/* No fallback string: two add-ons serve identical-looking panels and
              only this name separates them, so it comes from the server or not
              at all. */}
          <h1>{status?.display_name ?? ' '}</h1>
          <p className="sub">Who is home, and who is coming home.</p>
        </div>

        {/* Two buttons, so no roving tabindex: they are already keyboard
            reachable in order, and the arrow-key pattern would be a regression
            over that rather than an improvement. */}
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
