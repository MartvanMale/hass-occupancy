/**
 * The shape of what `server.py` serves.
 *
 * This file is one half of a contract; the other half is
 * `occupancy_forecast/tests/test_api_contract.py`, which asserts that every key named
 * here is actually present in a live response. Nothing checks the two against
 * each other automatically -- if you add a field, add it in both places. A
 * rename that only lands here is a panel that renders `undefined`, and a rename
 * that only lands in Python is caught by that test.
 *
 * Only what the panel reads is typed. `/api/status` carries a good deal more
 * (model_version, code fingerprints, eta_models); it is not the panel's business
 * and typing it would make the contract test complain about fields nobody uses.
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

/**
 * `detail` is genuinely polymorphic: a list of entity ids for `presence`, a
 * sentence for the rest. The old page ran `str()` over it and rendered
 * `['person.alice']` on screen; see `formatDetail` in `App.tsx` for the
 * per-shape formatting that replaces it.
 *
 * The `Record` arm is what `work_zones` used to send, back when a zone belonged
 * to a person. Kept because a config.json written by an older build can still
 * be on screen before the first save migrates it.
 */
export type FeatureDetail = string | string[] | Record<string, string>

export interface FeatureGroup {
  active: boolean
  detail: FeatureDetail
}

/** What serves a horizon. "model" where a trained model ships for it and beat
 *  its own baseline; "none" everywhere else, meaning nothing is published at
 *  all -- the sensor reads unknown and the forecast curve has a hole.
 *
 *  A union rather than `string`, and that is the point of the type: this used
 *  to be "model" or "baseline:<name>", and every reader had to split a status
 *  value on a colon to find out what it meant. Narrowing it makes `tsc` refuse
 *  any code that still tries. The baseline's name travels separately, in
 *  `Status.best_baseline`. */
export type ServedBy = 'model' | 'none'

/** The two model families. A dedicated model is fitted for one horizon and
 *  reads that horizon's own feature list; the pooled model is fitted once over
 *  every horizon with `horizon_h` among its columns. Which one serves is
 *  measured per horizon, not decided in advance. */
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
  /** Subject slugs, one per configured person. The Data tab needs them to name
   *  a subject in the feature table; `config.HOUSE_SLUG` is the other one and
   *  is always present. */
  people: string[]
  feature_groups: Record<string, FeatureGroup>
  /** Keyed by every horizon in the grid, including on a fresh install where
   *  nothing has been trained -- so the strip always has 48 cells to draw and
   *  its denominator cannot silently shrink when an artifact fails to load. */
  served_by: Record<string, ServedBy>
  /** WHICH family served, for the horizons where a model did. Additive beside
   *  `served_by`, which keeps its two values -- the horizon strip counts
   *  `=== 'model'` and should not have to learn about families. Keyed only by
   *  the shipping horizons, so a lookup for an unserved one is `undefined`
   *  rather than null. Redundant with `served_by` by construction now (a
   *  family entry exists exactly where the value is "model"); kept because
   *  `served_by` is the field the docs and `/health` name, and two fields
   *  saying one thing is better than a third field saying it again. */
  model_kind: Record<string, ModelKind>
  /** For the horizons nothing is published for, the baseline that beat the
   *  model. ABSENT where no model has been trained yet -- both cases publish
   *  nothing, and the absence is how the strip tells them apart in its
   *  tooltip. Same absence-means-something convention as `model_kind`. */
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
  /** The worker's own health. Everything else on this page can look perfect
   *  while it is hung -- `last_error` stays null when the thread is blocked
   *  rather than raising -- so `seconds_since_phase` is the one that ages. */
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
}

// --- the Overview tab -----------------------------------------------------

/** One subject's published forecast, exactly as it went to MQTT. */
export interface SubjectForecast {
  subject: string
  /** Fraction of the last five minutes spent at home -- an OBSERVATION, not a
   *  prediction. Shown beside the forecast because the comparison is the point. */
  current: number
  /** The slot the horizons are measured from — NOT `predicted_at`, which can be
   *  up to half an hour later. Anything putting clock times on the forecast axis
   *  must anchor here or be wrong by up to a slot. */
  observed_at: string
  /** Horizon in hours (as a string, JSON has no integer keys) to P(home).
   *
   *  SPARSE. Keyed only by the horizons a model actually served, so a missing
   *  key is a hole and not a zero -- reading it with `?? 0` draws a confident
   *  "certainly away" over the part of the curve the add-on has nothing to say
   *  about, which is what this used to do. */
  curve: Record<string, number>
  next_departure_h: number | null
  next_arrival_h: number | null
  /** Minutes until home from the proximity trace, or null.
   *
   *  Null unless they are actually CLOSING on home faster than
   *  `eta.MIN_CLOSING_KMH`. The model is conditional on being on a journey and
   *  is trained only within three hours of an arrival, so asked about somebody
   *  stationary it answers near the top of its range -- which it did, with 169
   *  minutes for a person at her desk six hours from home. */
  eta_minutes: number | null
  /** Null for the house, and for anyone without enough history yet. */
  out: OutRoutine | null
  next_change: NextChange | null
}

/**
 * What this person's own history says about today -- NOT a model forecast.
 *
 * Every number arrives with what it was built from. `n_out_weekday` is the
 * number of days out behind the hours, and `*_from` says whether the hour is
 * this weekday's own median or the fallback across all of them. A median off
 * four Fridays and one off thirty render identically without them.
 */
/**
 * One answer per person: the model's verdict that a change is coming, timed by
 * that person's routine for the day it falls on.
 *
 * `at_from` is the honest part. `routine` means a measured hour for that
 * weekday; `crossing` means the model's own rounded hour, used where the day
 * has no measured one. They are different qualities of answer and the card
 * should never have to guess which it holds.
 */
export interface NextChange {
  direction: 'leaving' | 'arriving' | null
  /** The model's own crossing, in whole hours ahead. Kept for reference. */
  in_hours: number | null
  at: string | null
  at_from: 'routine' | 'crossing' | null
}

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

/**
 * Every explorer endpoint answers "not yet" rather than 404ing, because on a
 * fresh install that is the truth for most of them and for the first ten days.
 *
 * Modelled as a union rather than an optional field on purpose: with `strict`
 * and `exactOptionalPropertyTypes`, `tsc --noEmit` then refuses to compile a
 * view that reads `.entities` without having narrowed on `available` first. The
 * empty state stops being something to remember.
 */
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

/** One family of feature columns. The table is a thousand columns wide, so this is how
 *  it is described -- a list of families, never a list of columns. */
/**
 * One slot of the verification chart.
 *
 * BOTH fields are nullable and neither may be coerced. `actual` is null for a
 * slot the trackers did not observe; `forecast` is null for a slot nothing was
 * published for -- either the horizon does not ship, or the add-on was not
 * running. A zero in place of either draws as "certainly away", which is the
 * reassuring lie the serving rule was changed to stop telling.
 */
export interface VerificationPoint {
  t: string
  actual: number | null
  forecast: number | null
}

/**
 * What the add-on SAID, against what happened -- the only number here that is
 * about the serving path rather than about a backtest.
 */
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
  retention_days: number
  summary: string
}>

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

/**
 * The leakage gate, seen from either end.
 *
 * `tgt{h}h_lag{k}d` is written into the table for EVERY horizon and is valid
 * only where `24k >= h`. Two shapes because the two views ask different
 * questions: a charted column asks "which horizon may not use me", and a
 * horizon asks "which of my four lags am I allowed".
 */
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
  /** Three values, and the third is not the second: "none" means a model was
   *  trained here and lost to its baseline, null means none was ever trained.
   *  Both publish nothing; only one of them has a bake-off to show. */
  served_by: ServedBy | null
  ships: boolean | null
  /** Which family's recipe this is. The two read different feature lists, so a
   *  card that did not say would be describing the wrong model half the time. */
  kind: ModelKind | null
}>

/** One week of the rolling-origin evaluation. */
export interface FoldScore {
  n: number
  /** Null on a fold this horizon had no test rows for.
   *
   *  `_scores_by_fold` emits an entry for EVERY fold index, empty ones
   *  included, because `ships` walks this list positionally against the
   *  baseline ladder's and a skipped fold would shift every later comparison
   *  by one. So the list is padded, and the padding is null -- typing these as
   *  plain numbers is what let a `.toFixed` on null reach the browser and
   *  black out the whole Data tab. */
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

/** The scalars, for the list. Everything bulky arrives with the detail. */
export interface HorizonMetrics {
  horizon_h: number
  brier: number
  log_loss: number
  auc: number
  mae_frac: number
  base_rate: number
  n_folds: number
  n_scored: number
  n_train_final: number
  best_baseline: string
  best_baseline_brier: number
  skill_vs_best_baseline_pct: number
  folds_beating_best_baseline: number
  sign_test_p: number
  ships: boolean
  brier_fold_min: number
  brier_fold_max: number
  /** Which family won this horizon, or null when a baseline did. */
  kind: ModelKind | null
  /** The losing family's Brier and name, so the crossover between the two is
   *  readable off the table rather than being something only a training log
   *  knows. Null when only one family produced a candidate. */
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

/**
 * One horizon in full. The two series here have been written to
 * `/data/models/metrics.json` on every train since the beginning and were never
 * rendered by anything.
 */
export type MetricsDetail = Explorable<HorizonMetrics & {
  per_fold: FoldScore[]
  reliability: ReliabilityBin[]
  /** The full ladder: base_rate, persistence, same_slot_yesterday and the three
   *  climatologies, each with the Brier it scored. */
  baselines: Record<string, number | null>
  fallback: { which?: string; column?: string; weight?: number; base?: number }
}>
