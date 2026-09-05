import type {
  Archive, Candidates, ConfigPatch, EntitySeries, FeatureInventory, Forecast,
  FeatureSeries, HorizonRecipe, MetricsDetail, MetricsSummary, Settings, Status,
  Verification,
} from './types'

/**
 * Every path here is relative and must stay that way.
 *
 * Ingress serves the panel from `/api/hassio_ingress/<token>/`, so `api/status`
 * resolves against that prefix while `/api/status` would escape it and hit Home
 * Assistant's own API, which answers 401. The build side of the same rule is
 * `base: './'` in vite.config.ts.
 */
async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path}: ${res.status} ${res.statusText}`)
  return (await res.json()) as T
}

export const getStatus = () => get<Status>('api/status')
export const getCandidates = () => get<Candidates>('api/candidates')
export const getSettings = () => get<Settings>('api/config')

/** The Overview tab. Served from what was last published, so this and the Home
 *  Assistant entities can never disagree. */
export const getForecast = () => get<Forecast>('api/forecast')

/**
 * The Data tab.
 *
 * `URLSearchParams` rather than interpolation: an entity id is full of dots and
 * could hold a `+` or a `#`, and a hand-built query string turns the first of
 * those into a space server-side and truncates at the second.
 */
export const getArchive = () => get<Archive>('api/explore/archive')

export const getEntitySeries = (entityId: string, days: number) =>
  get<EntitySeries>(
    `api/explore/entity?${new URLSearchParams({ entity_id: entityId, days: String(days) })}`,
  )

export const getFeatureInventory = () => get<FeatureInventory>('api/explore/features')

export const getFeatureSeries = (subject: string, column: string, days: number) =>
  get<FeatureSeries>(
    `api/explore/feature-series?${new URLSearchParams({ subject, column, days: String(days) })}`,
  )

export const getHorizon = (horizon: number) =>
  get<HorizonRecipe>(`api/explore/horizon/${horizon}`)

/** Uncached server-side, deliberately: the serve cycle rewrites both the
 *  archive and the forecast record every five minutes, and a stale answer is
 *  exactly what this card exists to catch. */
export const getVerification = (subject: string, horizon: number, days: number) =>
  get<Verification>(
    `api/explore/verification?${new URLSearchParams({
      subject, horizon: String(horizon), days: String(days),
    })}`,
  )

export const getMetrics = () => get<MetricsSummary>('api/explore/metrics')

export const getMetricsDetail = (horizon: number) =>
  get<MetricsDetail>(`api/explore/metrics/${horizon}`)

export async function saveConfig(patch: ConfigPatch): Promise<void> {
  const res = await fetch('api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!res.ok) {
    // The endpoint refuses an unknown country code with a 400 and a `detail`;
    // surfacing it is the difference between "save failed" and knowing why.
    const body = (await res.json().catch(() => ({}))) as { detail?: string }
    throw new Error(body.detail || `Save failed: ${res.status}`)
  }
}

/** The four things the Training card can set going.
 *
 *  `train` is the odd one out: it takes minutes, so it is asked to run in the
 *  background and answers 202 immediately. Progress is then read from
 *  `training_in_progress` on the status poll rather than from this promise.
 */
export type Action = 'train' | 'collect' | 'predict' | 'reload'

const PATHS: Record<Action, string> = {
  train: 'train?background=1',
  collect: 'collect',
  predict: 'predict',
  reload: 'reload',
}

export async function runAction(action: Action): Promise<Record<string, unknown>> {
  const res = await fetch(PATHS[action], { method: 'POST' })
  const body = (await res.json().catch(() => ({}))) as Record<string, unknown>
  if (!res.ok) {
    // /train answers 409 with a sentence explaining that there is not enough
    // history yet, which is the most useful thing this can say.
    throw new Error((body['detail'] as string) || `${action} failed: ${res.status}`)
  }
  return body
}
