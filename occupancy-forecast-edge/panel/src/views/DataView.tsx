import { useCallback, useEffect, useState } from 'react'
import {
  getArchive, getEntitySeries, getFeatureInventory, getFeatureSeries, getHorizon,
  getMetrics, getMetricsDetail,
} from '../api'
import type {
  Archive, EntitySeries, FeatureInventory, FeatureSeries,
  HorizonRecipe, MetricsDetail, MetricsSummary, Status,
} from '../types'
import { Card } from '../components/Card'
import { Step } from '../components/Step'
import { Chip } from '../components/Chip'
import { Row } from '../components/Row'
import { Select } from '../components/Select'
import { HorizonSlider } from '../components/HorizonSlider'
import { VerificationCard } from '../components/VerificationCard'
import { TimeSeries, type Point } from '../components/TimeSeries'
import { absoluteTime, count, DAY_OPTIONS } from '../format'
import { useDebounced } from '../hooks'
import { ArchiveCard } from './data/ArchiveStep'
import {
  ColumnCard, FeatureTableCard, HorizonCard, horizonSummary,
} from './data/FeatureCards'
import { HorizonQualityCard, QualityCard } from './data/QualityCards'

/**
 * Data: the path a state change takes to become a forecast, in five steps.
 * One column, read top to bottom -- it is one argument in order, and a grid that
 * reflows by width shuffles the argument. Mounted only while its tab is showing,
 * and nothing here polls: fetches happen on mount and when a control changes.
 */

function EntityCard({ series }: { series: EntitySeries | null }) {
  if (!series) return <p className="empty">Loading…</p>
  if (!series.available) return <p className="empty">{series.reason}</p>

  const points: Point[] = series.gridded.map((g) => ({ t: g.t, v: g.v }))
  const { summary, unit } = series
  const presence = series.kind !== 'numeric'

  const observed = summary.n - summary.nulls
  const range = summary.min === null
    ? ''
    : presence
      ? `It ranges from ${summary.min} to ${summary.max}, averaging ${summary.mean}.`
      : `It ranges from ${summary.min}${unit ? ` ${unit}` : ''} to ${summary.max}${
          unit ? ` ${unit}` : ''}, averaging ${summary.mean}${unit ? ` ${unit}` : ''}.`

  // Two forms of the same fact: the long one is the chart's `aria-label`, the
  // short one is printed under it, where the breaks already show a missing slot.
  const gaps = summary.nulls === 0
    ? 'Every slot in the window was observed.'
    : `${count(summary.nulls)} of ${count(summary.n)} slots were not observed — either nothing ` +
      `reported, or less than ${Math.round(series.min_coverage * 100)}% of the slot was covered, ` +
      `which is not enough to call. Those are drawn as breaks, not as zero.`
  const shortGaps = summary.nulls === 0
    ? 'Every slot observed.'
    : `${count(summary.nulls)} of ${count(summary.n)} slots not observed.`

  const prose = `${series.gridded_label}. ${count(observed)} observed slots. ${range} ${gaps}`
  const caption = `${count(observed)} observed slots. ${range} ${shortGaps}`

  return (
    <>
      <Row
        icon="chart"
        accent="aqua"
        primary={series.entity_id}
        secondary={
          series.truncated
            ? `${count(series.raw_rows)} raw transitions in this window; the ${count(
                series.events.length)} most recent are charted below.`
            : `${count(series.raw_rows)} raw transitions in this window.`
        }
        trailing={
          <Chip label={series.kind} icon={presence ? 'people' : 'marker'}
                accent="blue" />
        }
      />
      <TimeSeries
        points={points}
        label={presence ? 'home_frac' : `value${unit ? ` (${unit})` : ''}`}
        unit={presence ? null : unit}
        {...(presence ? { domain: [0, 1] as [number, number] } : {})}
        startLabel={absoluteTime(series.start)}
        endLabel={absoluteTime(series.stop)}
        summary={prose}
        caption={caption}
      />
    </>
  )
}

export function DataView({ status }: { status: Status | null }) {
  const [archive, setArchive] = useState<Archive | null>(null)
  const [picked, setPicked] = useState<string>('')
  const [days, setDays] = useState<string>('7')
  const [series, setSeries] = useState<EntitySeries | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [inventory, setInventory] = useState<FeatureInventory | null>(null)
  // Two, not one: `horizonInput` drives the readout so dragging feels attached,
  // `horizon` keys the fetches and settles only when the drag stops.
  const [horizonInput, setHorizonInput] = useState<number>(24)
  const horizon = useDebounced(horizonInput, 250)
  const [recipe, setRecipe] = useState<HorizonRecipe | null>(null)
  const [subject, setSubject] = useState<string>('')
  const [column, setColumn] = useState<string>('home_frac')
  const [featureDays, setFeatureDays] = useState<string>('30')
  const [featureSeries, setFeatureSeries] = useState<FeatureSeries | null>(null)

  const [metrics, setMetrics] = useState<MetricsSummary | null>(null)
  const [detail, setDetail] = useState<MetricsDetail | null>(null)

  useEffect(() => {
    getArchive()
      .then((a) => {
        setArchive(a)
        // Open on something worth looking at rather than on an empty picker:
        // the first entity that is both read and has rows.
        if (a.available && !picked) {
          const first = a.entities.find((e) => e.tracked && e.rows > 0)
          if (first) setPicked(first.entity_id)
        }
      })
      .catch((e: Error) => setError(e.message))
    // Once, on mount: nothing here is polled, and listing `picked` would refetch
    // the whole inventory on every change of entity.
  }, [])

  useEffect(() => {
    if (!picked) return
    let live = true
    setSeries(null)
    // The one fetch that re-runs on a control, so the one that clears a stale
    // message -- after the `picked` guard, so an archive load failure stays.
    setError(null)
    getEntitySeries(picked, Number(days))
      .then((s) => { if (live) setSeries(s) })
      .catch((e: Error) => { if (live) setError(e.message) })
    return () => { live = false }
  }, [picked, days])

  useEffect(() => {
    getFeatureInventory().then(setInventory).catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => {
    let live = true
    setRecipe(null)
    getHorizon(horizon)
      .then((r) => { if (live) setRecipe(r) })
      .catch(() => {})
    return () => { live = false }
  }, [horizon])

  // The house is always a subject; the people come from the status poll, which
  // is where the slugs are already derived.
  const subjects = ['house', ...(status?.people ?? [])]
  const chosenSubject = subject || subjects[0] || ''

  useEffect(() => {
    if (!chosenSubject || !column) return
    let live = true
    setFeatureSeries(null)
    getFeatureSeries(chosenSubject, column, Number(featureDays))
      .then((s) => { if (live) setFeatureSeries(s) })
      .catch(() => {})
    return () => { live = false }
  }, [chosenSubject, column, featureDays])

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => {})
  }, [])

  // Keyed off the same horizon picker as the recipe, so the two cards are
  // always talking about the same model.
  useEffect(() => {
    let live = true
    setDetail(null)
    getMetricsDetail(horizon)
      .then((d) => { if (live) setDetail(d) })
      .catch(() => {})
    return () => { live = false }
  }, [horizon])

  // Picking sets the entity and nothing else: a page that moves under you takes
  // the decision about where to look out of your hands.
  const pick = useCallback((entityId: string) => setPicked(entityId), [])

  const options = archive?.available
    ? archive.entities.filter((e) => e.rows > 0)
        .map((e) => ({ value: e.entity_id, label: e.entity_id }))
    : []

  const columnOptions = inventory?.available
    ? inventory.browsable.map((c) => ({
        value: c.name,
        label: `${c.family.replace(/_/g, ' ')} · ${c.name}`,
      }))
    : []

  // Every count in the prose below is derived: a literal is a number that goes
  // quietly wrong on somebody else's commit.
  const width = inventory?.available ? inventory.columns : null
  // Not "per-horizon columns": the two key columns are neither browsable nor
  // per-horizon, so this is honestly "the ones not offered in the picker".
  const unlisted = inventory?.available
    ? inventory.columns - inventory.browsable.length
    : null
  // `served_by` carries one entry per horizon, so it is also the horizon count
  // without a second field saying the same thing.
  const nHorizons = status ? Object.keys(status.served_by).length : null

  return (
    <div className="steps">
      {/* One clause per step, saying what the step SHOWS; the justification is
          in DOCS.md under "The Data tab". */}
      <p className="lede">
        Every number the forecast rests on, in the order it is made.
      </p>

      <Step
        n={1}
        id="archive"
        eyebrow="Input"
        title="The archive"
        intro="Every state change the add-on has collected, in its own database."
      >
        <Card>
          {error ? <p className="empty error">{error}</p>
                 : <ArchiveCard archive={archive} picked={picked} onPick={pick} />}
        </Card>
      </Step>

      <Step
        n={2}
        id="entity"
        eyebrow="Read"
        title="One entity, as the model reads it"
        intro="Above the line is what Home Assistant reported; the chart is what the
          feature builder makes of it."
      >
        <Card>
          <Row
            icon="database"
            control
            accent="blue"
            primary="Entity"
            secondary="Only entities with rows in the archive can be charted."
            trailing={
              <Select
                label="Entity to inspect"
                searchable
                value={picked}
                onChange={setPicked}
                options={options.length ? options : [{ value: '', label: '— nothing collected yet —' }]}
              />
            }
          />
          <Row
            icon="clock"
            control
            accent="blue"
            primary="Window"
            secondary="How far back to read."
            trailing={
              <Select label="Time window" value={days} onChange={setDays}
                      options={DAY_OPTIONS} />
            }
          />
          {picked ? <EntityCard series={series} /> : null}
        </Card>
      </Step>

      <Step
        n={3}
        id="features"
        eyebrow="Build"
        title="The feature table"
        intro={`What the archive becomes on the modelling grid${
          width === null ? '' : `: ${count(width)} columns`}, described by family.`}
      >
        <Card>
          <FeatureTableCard inventory={inventory} />
        </Card>

        <Card
          title="A column over time"
          subtitle="Straight out of the feature table, not a recomputation."
        >
          <Row
            icon="people"
            control
            accent="blue"
            primary="Subject"
            secondary="One per person, plus the house."
            trailing={
              <Select
                label="Subject"
                value={chosenSubject}
                onChange={setSubject}
                options={subjects.length
                  ? subjects.map((s) => ({ value: s, label: s }))
                  : [{ value: '', label: '— nobody configured —' }]}
              />
            }
          />
          <Row
            icon="table"
            control
            accent="blue"
            primary="Column"
            secondary={unlisted === null
              ? 'The origin features.'
              : `The origin features. The other ${count(unlisted)} columns are summarised
                 above by family rather than listed here.`}
            trailing={
              <Select
                label="Column"
                searchable
                value={column}
                onChange={setColumn}
                options={columnOptions.length
                  ? columnOptions
                  : [{ value: '', label: '— no feature table yet —' }]}
              />
            }
          />
          <Row
            icon="clock"
            control
            accent="blue"
            primary="Window"
            secondary="How far back to read."
            trailing={
              <Select label="Feature window" value={featureDays} onChange={setFeatureDays}
                      options={[...DAY_OPTIONS, { value: '400', label: 'everything' }]} />
            }
          />
          {columnOptions.length ? <ColumnCard series={featureSeries} /> : null}
        </Card>
      </Step>

      <Step
        n={4}
        id="horizon"
        eyebrow="Withhold"
        title="What one horizon is allowed to use"
        intro={`Each horizon${nHorizons ? ` — there are ${nHorizons} of them` : ''} gets its
          own model and its own slice of the table.`}
      >
        <Card>
          <HorizonSlider
            value={horizonInput}
            onChange={setHorizonInput}
            summary={horizonSummary(
              recipe,
              inventory?.available ? inventory.columns : null,
              horizonInput !== horizon,
            )}
          />
          <HorizonCard recipe={recipe} />
        </Card>
      </Step>

      <Step
        n={5}
        id="scores"
        eyebrow="Judge"
        title="Whether any of it worked"
        intro="Every horizon against the baseline it had to beat."
      >
        <Card>
          <QualityCard metrics={metrics} current={horizonInput}
                       onPick={setHorizonInput} />
        </Card>

        {/* No subtitle: the two subheads inside already label both charts, and
            the title names the horizon. */}
        <Card title={`Horizon +${horizon} h in detail`}>
          <HorizonQualityCard detail={detail} />
        </Card>

        {/* The only entry in this step that is not a backtest: what the add-on
            actually published and what happened next. A gap from the two above is the finding. */}
        <VerificationCard status={status} />
      </Step>
    </div>
  )
}
