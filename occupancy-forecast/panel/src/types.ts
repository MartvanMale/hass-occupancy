/**
 * The shape of what `server.py` serves -- one half of a contract whose other
 * half is `occupancy_forecast/tests/test_api_contract.py`. Nothing checks the
 * two against each other: if you add or rename a field, do it in both places.
 * Only what the panel reads is typed.
 */

export interface History {
  days?: number
  rows?: number
  note?: string
}

export interface Mqtt {
  connected: boolean
  error: string | null
}

export interface Listener {
  connected: boolean
  entities?: number
  fired?: number
  events?: number
  last_event?: string | null
  last_error?: string | null
}

/** Polymorphic: a list of entity ids for `presence`, a sentence for the rest, and
 *  a `Record` an older config.json can still be showing. See `formatDetail`. */
export type FeatureDetail = string | string[] | Record<string, string>

export interface FeatureGroup {
  active: boolean
  detail: FeatureDetail
}

/** What serves a horizon. "model" where a trained model ships and beat its
 *  baseline; "none" everywhere else -- nothing is published, the sensor reads
 *  unknown. The baseline's name travels in `Status.best_baseline`. */
export type ServedBy = 'model' | 'none'

/** Dedicated = fitted for one horizon; pooled = fitted once over all of them.
 *  Which serves is measured per horizon. */
export type ModelKind = 'dedicated' | 'pooled'

export interface WorkerHealth {
  /** Which step it is in: collect, predict, notify, waiting. */
  phase: string
  cycles: number
  seconds_since_phase: number
  stalled: boolean
  stalled_since: string | null
  stalled_in: string | null
  /** Stall episodes since the add-on started, not since it was installed. */
  stalls: number
}

export interface Status {
  display_name: string
  history: History
  days_until_training: number
  /** Presence actually observed -- lower than `history.days`, which an unused
   *  tracker inflates. Null on Influx. */
  usable_presence_days: number | null
  /** Subject slugs, one per configured person; `config.HOUSE_SLUG` is the other
   *  subject and is always present. */
  people: string[]
  feature_groups: Record<string, FeatureGroup>
  /** Keyed by every horizon in the grid, even on a fresh install, so the strip's
   *  denominator cannot shrink when an artifact fails to load. */
  served_by: Record<string, ServedBy>
  /** WHICH family served, keyed only by shipping horizons -- a lookup for an
   *  unserved one is `undefined`, not null. */
  model_kind: Record<string, ModelKind>
  /** For unpublished horizons, the baseline that beat the model. ABSENT where no
   *  model was ever trained; the absence is how the strip tells them apart. */
  best_baseline: Record<string, string>
  mqtt: Mqtt
  listener: Listener
  last_train: string | null
  /** How long the last train took, end to end. Null until one has been timed --
   *  models trained by an older build have a timestamp but no duration. */
  last_train_seconds: number | null
  /** When the worker will next retrain, local time with its offset. Null while
   *  there is too little history for a train to be possible at all. */
  next_train: string | null
  /** Which schedule `next_train` came off, so the panel need not infer it. */
  train_cadence: 'daily' | 'weekly'
  training_in_progress: boolean
  /** When the run currently in progress began. Stale once it finishes; only
   *  read while `training_in_progress`. */
  training_started_at: string | null
  /** The worker's own health: everything else stays green while it is hung, since
   *  a blocked thread never raises. `seconds_since_phase` is the one that ages. */
  worker: WorkerHealth
  last_collect: string | null
  last_predict: string | null
  last_error: string | null
}

export interface Entity {
  entity_id: string
  name: string
}

export interface Country {
  code: string
  name: string
}

export interface Candidates {
  people: Entity[]
  zones: Entity[]
  groups: Entity[]
  /** `schedule.*` entities, for the optional night shading. */
  schedules: Entity[]
  countries: Country[]
  has_proximity: boolean
}

/** `config.Settings`, as `asdict` renders it. */
export interface Settings {
  /** A `schedule.*` entity for the household's waking hours, or null.
   *  Shades the forecast chart and nothing else. */
  day_schedule: string | null
  people: string[]
  zones: string[]
  house_entity: string | null
  holiday_country: string | null
  departure_threshold: number
  arrival_threshold: number
  crossing_min_hours: number
  /** Days of published forecasts kept for the "Was it right?" chart.
   *  0 means never pruned. */
  forecast_retention_days: number
}

/** What POST /api/config accepts. A key left out is a setting left alone. */
export interface ConfigPatch {
  /** A `schedule.*` entity for the household's waking hours, or null.
   *  Shades the forecast chart and nothing else. */
  day_schedule: string | null
  people: string[]
  zones: string[]
  house_entity: string | null
  holiday_country?: string
  // Required, not optional: an optional field is how a form quietly stops
  // sending a value the user set.
  departure_threshold: number
  arrival_threshold: number
  crossing_min_hours: number
  /** Days of published forecasts kept for the "Was it right?" chart.
   *  0 means never pruned. */
  forecast_retention_days: number
}

// --- the Overview tab -----------------------------------------------------

/** One subject's published forecast, exactly as it went to MQTT. */
export interface SubjectForecast {
  subject: string
  /** Fraction of the last five minutes spent at home -- an OBSERVATION, not a
   *  prediction. Shown beside the forecast because the comparison is the point. */
  current: number
  /** The slot the horizons are measured from -- NOT `predicted_at`, which can be
   *  half an hour later. Clock times on the forecast axis must anchor here. */
  observed_at: string
  /** Horizon in hours (string keys; JSON has no integer ones) to P(home).
   *  SPARSE -- a missing key is a hole, never a zero: `?? 0` draws "certainly
   *  away" over the hours nothing was said about. */
  curve: Record<string, number>
  next_departure_h: number | null
  next_arrival_h: number | null
  /** Minutes until home from the proximity trace, or null. Null unless actually
   *  closing faster than `eta.MIN_CLOSING_KMH` -- the model is trained only within
   *  three hours of an arrival, so asked about somebody stationary it answers
   *  near the top of its range. */
  eta_minutes: number | null
  /** Null for the house, and for anyone without enough history yet. */
  out: OutRoutine | null
  next_change: NextChange | null
}

/** The model's verdict that a change is coming, optionally sharpened by that
 *  person's routine. `at_from` says which quality of answer it is: `routine` is
 *  a measured hour for that weekday, `crossing` is the model's own rounded one.
 *  The routine may only move the crossing a few hours -- further than that and
 *  the two are naming different events, so the crossing is kept. */
export interface NextChange {
  direction: 'leaving' | 'arriving' | null
  /** The model's own crossing, in whole hours ahead. Kept for reference. */
  in_hours: number | null
  at: string | null
  at_from: 'routine' | 'crossing' | null
  /** What the routine offered, whether or not it was used -- null when that day
   *  had no hour to give. Shown so a refusal is visible rather than silent. */
  routine_at: string | null
  /** That person's routine for the day the change FALLS ON, not for today. Null
   *  for the house before it has enough history, and on a fresh install. */
  routine_day: DepartureRoutine | null
}

/** What this person's own history says about a given weekday -- NOT a model
 *  forecast. Fitted on "left the house at all", so a short errand counts; its
 *  twin `OutRoutine` counts only days that reached a configured zone. */
export interface DepartureRoutine {
  probability: number
  weekday: number
  n_weekday: number
  n_left_weekday: number
  departure_hour: number | null
  departure_sd: number | null
  /** 'weekday' is measured on that weekday and is the only one allowed to move
   *  the crossing. 'overall' is a median off the OTHER weekdays -- worth
   *  showing, never worth acting on. 'never' means that weekday has been seen
   *  often enough with no departures at all, and the hours are null. */
  departure_from: 'weekday' | 'overall' | 'never'
  return_hour: number | null
  return_sd: number | null
  return_from: 'weekday' | 'overall' | 'never'
  fitted_at: string | null
}

/** What this person's own history says about today -- NOT a model forecast.
 *  Every number arrives with what it was built from: `n_out_weekday` and `*_from`
 *  are what separate a median off four Fridays from one off thirty. */
export interface OutRoutine {
  probability: number
  weekday: number
  n_weekday: number
  n_out_weekday: number
  departure_hour: number | null
  departure_sd: number | null
  /** 'never' when this weekday has been seen often enough with no days out at
   *  all -- an answer rather than a gap, and the hours are null. */
  departure_from: 'weekday' | 'overall' | 'never'
  return_hour: number | null
  return_sd: number | null
  return_from: 'weekday' | 'overall' | 'never'
  fitted_at: string | null
}

/** One asleep run on the forecast chart, in hours ahead of now -- the chart's
 *  own axis, not clock time. Empty unless a day schedule is configured. */
export interface NightBand {
  from: number
  to: number
}

export type Forecast = Explorable<{
  predicted_at: string | null
  house: string
  horizons: number[]
  night: NightBand[]
  subjects: SubjectForecast[]
}>

// --- the Data tab ---------------------------------------------------------

/** Every explorer endpoint answers "not yet" rather than 404ing. A union, not an
 *  optional field, so `tsc` refuses a view that reads the payload without
 *  narrowing on `available` first. */
export type Unavailable = { available: false; reason: string }
export type Explorable<T> = Unavailable | ({ available: true } & T)

/** `span()` from the store: what the whole archive covers. */
export interface ArchiveSpan {
  first: string | null
  last: string | null
  rows: number
  days: number
  bytes: number
}

export interface ArchiveEntity {
  entity_id: string
  rows: number
  first: string | null
  last: string | null
  /** Read off the values, never off the entity id -- naming is the user's. */
  kind: 'presence' | 'numeric' | 'heartbeat' | 'other'
  /** What the add-on uses it for, from the settings. */
  role: string
  /** Whether anything actually reads it. A false here on a row with a healthy
   *  count, or a true on a row with none, is the whole point of the card. */
  tracked: boolean
}

export type Archive = Explorable<{
  span: ArchiveSpan
  entities: ArchiveEntity[]
}>

/** A raw transition, exactly as it sits in the archive. */
export interface RawEvent {
  t: string
  v: string
}

/** One slot of the modelling grid. `v` is null where nothing was observed --
 *  never zero, which would draw as "away" and read as a fact. */
export interface GridPoint {
  t: string
  v: number | null
  coverage: number | null
}

export interface SeriesSummary {
  n: number
  nulls: number
  min: number | null
  max: number | null
  mean: number | null
  last: number | null
}

export type EntitySeries = Explorable<{
  entity_id: string
  kind: ArchiveEntity['kind']
  role: string
  start: string
  stop: string
  unit: string | null
  raw_rows: number
  truncated: boolean
  events: RawEvent[]
  grid_minutes: number
  gridded: GridPoint[]
  gridded_label: string
  min_coverage: number
  summary: SeriesSummary
}>

/** One slot of the verification chart. BOTH fields are nullable and neither may
 *  be coerced: `actual` null is a slot the trackers did not see, `forecast` null
 *  is a slot nothing was published for. A zero draws as "certainly away". */
export interface VerificationPoint {
  t: string
  actual: number | null
  forecast: number | null
}

/** What the add-on SAID against what happened -- the only number here about the
 *  serving path rather than a backtest. */
export type Verification = Explorable<{
  subject: string
  horizon_h: number
  grid_minutes: number
  start: string
  stop: string
  points: VerificationPoint[]
  /** Slots in the window, and how many of them carried a forecast. */
  slots: number
  served: number
  /** Slots where both a forecast and an observation exist -- what was scored. */
  scored: number
  brier: number | null
  mae: number | null
  /** How long the forecast log keeps a row. 0 means never pruned. */
  retention_days: number
  summary: string
}>

/** One family of feature columns -- the table is a thousand columns wide, so it
 *  is described as a list of families, never of columns. */
export interface FeatureFamily {
  family: string
  words: string
  columns: number
  /** Null when the parquet footer carried no statistics. */
  null_frac: number | null
}

/** A column offered as a chart. Only the origin families; the 672 per-horizon
 *  columns are summarised by family and never listed. */
export interface ColumnStat {
  name: string
  family: string
  null_frac: number | null
  min: number | null
  max: number | null
}

export type FeatureInventory = Explorable<{
  path: string
  built_at: string
  bytes: number
  rows: number
  columns: number
  row_groups: number
  /** Length of one slot, so the panel does not keep its own copy of a server
   *  constant to describe what a row is. */
  grid_minutes: number
  /** False when pyarrow wrote no column statistics. The families are read off
   *  the schema and stay right; only the null fractions and ranges go missing. */
  statistics: boolean
  families: FeatureFamily[]
  browsable: ColumnStat[]
}>

/** The leakage gate from either end. `tgt{h}h_lag{k}d` is valid only where
 *  `24k >= h`; two shapes because a column asks "which horizon may not use me"
 *  and a horizon asks "which lags am I allowed". */
export interface LagSafety {
  horizon_h: number
  days: number
  safe: boolean
  why: string | null
}

export interface HorizonLag {
  days: number
  column: string
  safe: boolean
  why: string | null
}

export interface SeriesPoint {
  t: string
  v: number | null
}

export type FeatureSeries = Explorable<{
  subject: string
  column: string
  family: string
  words: string
  grid_minutes: number
  points: SeriesPoint[]
  /** True when the window was sampled down rather than cut short. */
  thinned: boolean
  safe_for: LagSafety | null
  start: string
  stop: string
  summary: SeriesSummary
}>

export interface RecipeFamily {
  family: string
  words: string
  columns: number
}

export type HorizonRecipe = Explorable<{
  horizon_h: number
  target: string
  residual_base: string
  n_features: number
  features: string[]
  families: RecipeFamily[]
  daily_lags: HorizonLag[]
  climatology: string
  columns_read: number
  embargo_hours: number
  /** "none" = trained here and lost; null = never trained. Both publish nothing;
   *  only one has a bake-off to show. */
  served_by: ServedBy | null
  ships: boolean | null
  /** Which family's recipe this is. The two read different feature lists, so a
   *  card that did not say would be describing the wrong model half the time. */
  kind: ModelKind | null
}>

/** One week of the rolling-origin evaluation. */
export interface FoldScore {
  n: number
  /** Null on a fold this horizon had no test rows for. The list is padded for
   *  EVERY fold index, because `ships` walks it positionally against the
   *  ladder's -- do not type these as plain numbers. */
  base_rate: number | null
  brier: number | null
  log_loss: number | null
  auc: number | null
  mae_frac: number | null
}

/** One bin of the calibration curve, from `evaluate.reliability`. */
export interface ReliabilityBin {
  bin_low: number
  bin_high: number
  n: number
  predicted: number
  observed: number
}

/** The scalars, for the list; anything bulky arrives with the detail. Every
 *  float is nullable -- the server writes NaN where a score is undefined and
 *  `explore._json_safe` nulls it at the boundary. */
export interface HorizonMetrics {
  horizon_h: number
  brier: number | null
  log_loss: number | null
  auc: number | null
  mae_frac: number | null
  base_rate: number | null
  n_folds: number
  n_scored: number
  n_train_final: number
  best_baseline: string
  best_baseline_brier: number | null
  skill_vs_best_baseline_pct: number | null
  folds_beating_best_baseline: number
  sign_test_p: number | null
  ships: boolean
  brier_fold_min: number | null
  brier_fold_max: number | null
  /** Which family won this horizon, or null when a baseline did. */
  kind: ModelKind | null
  /** The losing family's Brier and name, so the crossover is readable off the
   *  table. Null when only one family produced a candidate. */
  rival_brier: number | null
  rival_kind: ModelKind | null
}

export type MetricsSummary = Explorable<{
  trained_at: string | null
  model_version: string | null
  evaluation: string | null
  duration_s: number | null
  shipping: number
  horizons: HorizonMetrics[]
  /** Horizons whose training raised, keyed by horizon, valued by the error. */
  failed: Record<string, string>
}>

/** One horizon in full, including the two series `metrics.json` has always carried. */
export type MetricsDetail = Explorable<HorizonMetrics & {
  per_fold: FoldScore[]
  reliability: ReliabilityBin[]
  /** The full ladder: base_rate, persistence, same_slot_yesterday and the three
   *  climatologies, each with the Brier it scored. */
  baselines: Record<string, number | null>
  fallback: { which?: string; column?: string; weight?: number; base?: number }
}>
