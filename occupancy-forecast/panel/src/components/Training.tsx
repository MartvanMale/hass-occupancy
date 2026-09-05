import { useState } from 'react'
import { runAction, type Action } from '../api'
import { absoluteTime, duration, relativeTime, splitError } from '../format'
import type { Status } from '../types'
import { Icon, type IconName } from './Icon'
import { Row } from './Row'

/**
 * When the model was last rebuilt, when it will be next, and the four things
 * that can be set going by hand.
 *
 * The duration is the point of the card. Retraining is 48 models plus the ETA
 * models on a box that is also running Home Assistant, and "how long does that
 * take here" was previously unanswerable -- which made pressing the button an
 * act of faith. It is measured now and shown, so the second press is informed.
 *
 * No confirmation dialog. A train is idempotent and non-destructive: it either
 * replaces the models or fails and leaves the old ones serving. The honest
 * signals are the disabled buttons and the elapsed time, not a modal.
 */

const BUTTONS: { action: Action; label: string; icon: IconName; primary?: boolean }[] = [
  { action: 'train', label: 'Retrain now', icon: 'refresh', primary: true },
  { action: 'collect', label: 'Collect now', icon: 'collected' },
  { action: 'predict', label: 'Predict now', icon: 'model' },
  { action: 'reload', label: 'Reload models', icon: 'restore' },
]

const DONE: Record<Action, string> = {
  train: 'Training started.',
  collect: 'Collected.',
  predict: 'Published a fresh forecast.',
  reload: 'Models reloaded from disk.',
}

export function Training({
  status,
  refresh,
}: {
  status: Status
  refresh: () => Promise<void>
}) {
  const [running, setRunning] = useState<Action | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [failed, setFailed] = useState<string | null>(null)

  const training = status.training_in_progress
  // The worker fills last_error for any failed job, not only a train; the
  // wording stays generic because this is the only card with room to say it.
  const failure = splitError(status.last_error)
  // The word only. Why it is weekly rather than daily -- the history has stopped
  // changing the answer -- is a fact about the scheduler's design, not about
  // this household's next train, and it is in DOCS.md under "Training cadence".
  const cadence = status.train_cadence === 'weekly' ? 'Weekly' : 'Daily'

  async function run(action: Action) {
    setRunning(action)
    setMessage(null)
    setFailed(null)
    try {
      await runAction(action)
      setMessage(DONE[action])
      // Awaited, so the buttons stay disabled until the status that says a
      // train is running has actually arrived. Released earlier they flicker
      // back to enabled for one round trip.
      await refresh()
    } catch (err) {
      setFailed((err as Error).message)
    } finally {
      setRunning(null)
    }
  }

  return (
    <>
      {training ? (
        <Row
          icon="refresh"
          accent="aqua"
          primary="Training now"
          secondary={`Started ${relativeTime(status.training_started_at)}.`}
        />
      ) : status.last_train ? (
        <Row
          icon="model"
          accent="aqua"
          primary={`Last trained ${relativeTime(status.last_train)}`}
          secondary={
            absoluteTime(status.last_train) +
            (status.last_train_seconds != null
              ? ` · took ${duration(status.last_train_seconds)}`
              : '')
          }
        />
      ) : (
        <Row
          icon="collecting"
          accent="orange"
          primary="Not trained yet"
          secondary="Not enough history yet."
        />
      )}

      <Row
        icon="clock"
        accent="blue"
        primary={
          status.next_train
            ? `Next train ${relativeTime(status.next_train)}`
            : 'No train scheduled yet'
        }
        secondary={
          status.next_train
            ? `${absoluteTime(status.next_train)} · ${cadence}`
            : 'Not enough history yet.'
        }
      />

      {failure && (
        <Row
          icon="alert"
          accent="red"
          primary="Something failed"
          secondary={
            failure.when
              ? `${failure.message} · ${relativeTime(failure.when)}`
              : failure.message
          }
        />
      )}

      <div className="actions wrapped">
        {BUTTONS.map(({ action, label, icon, primary }) => (
          <button
            key={action}
            type="button"
            className={primary ? undefined : 'secondary'}
            // Everything is disabled during a train, not just the train button:
            // a collect or a predict competing with it for the same box is how
            // you turn four minutes into eight.
            disabled={training || running !== null}
            onClick={() => void run(action)}
          >
            <Icon name={icon} />
            {running === action ? 'Working…' : label}
          </button>
        ))}
      </div>

      {(message || failed) && (
        <p className={failed ? 'empty error' : 'empty'}>{failed ?? message}</p>
      )}
    </>
  )
}
