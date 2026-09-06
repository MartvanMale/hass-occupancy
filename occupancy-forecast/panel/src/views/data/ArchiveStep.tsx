import type { Archive, ArchiveEntity } from '../../types'
import { Chip } from '../../components/Chip'
import { Row } from '../../components/Row'
import { Shape, type Accent, type IconName } from '../../components/Icon'
import { absoluteTime, relativeTime } from '../../format'

/**
 * Step one: the archive, and the entity the rest of the page follows.
 *
 * The entity list was a stack of `Row`s each ending in a small button. That put
 * a 90px target at the far right of a 24rem row and made the other 90% of the
 * row -- the name you are actually reading -- inert. It is a grid of tiles now,
 * and the tile IS the button: the whole thing is the target, `aria-current` says
 * which one the page is following, and picking one scrolls step two into view.
 *
 * The two summary rows above the grid stay rows. They are statements, not
 * choices, and making them look pickable would be a lie.
 */

const count = (n: number) => n.toLocaleString()

const bytes = (n: number) => {
  if (n < 1024) return `${n} B`
  const units = ['kB', 'MB', 'GB']
  let value = n / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1 }
  return `${value.toFixed(1)} ${units[i]}`
}

/** The icon for an entity is what the add-on uses it FOR, which is more use
 *  than what type it is -- the type is already in the secondary line. */
const ROLE_ICONS: Record<string, IconName> = {
  person: 'people',
  house: 'house',
  'work-zone': 'marker',
  proximity: 'marker',
  direction: 'marker',
  'synthetic-distance': 'marker',
  heartbeat: 'clock',
  untracked: 'eye-off',
}

const ROLE_WORDS: Record<string, string> = {
  person: 'a person the model predicts for',
  house: 'the house group',
  'work-zone': 'a work zone',
  proximity: 'distance to home, from the Proximity integration',
  direction: 'direction of travel, from the Proximity integration',
  'synthetic-distance': 'distance to home, synthesised from GPS',
  heartbeat: "the collector's own heartbeat — how it knows history was being kept",
  untracked: 'nothing reads this',
}

function EntityTile({ entity, current, onPick }: {
  entity: ArchiveEntity
  current: boolean
  onPick: (entityId: string) => void
}) {
  const { tracked, rows, role } = entity
  // Orange, not red, and not grey: an entity configured with no rows is the one
  // genuinely wrong state this card can show, and it should not look like the
  // merely-unused ones.
  const empty = tracked && rows === 0
  const accent: Accent = empty ? 'orange' : tracked ? 'aqua' : 'grey'
  const words = ROLE_WORDS[role] ?? role

  return (
    <button
      type="button"
      className={`pick${tracked ? '' : ' off'}`}
      // Not `aria-pressed`: these are one-of-many, and `current` is the word for
      // "the one being shown" rather than "switched on".
      aria-current={current}
      // Nothing to chart, so nothing to follow it to. It still appears, because
      // "this entity is configured and has never reported" is the single most
      // useful thing this grid can tell you.
      disabled={rows === 0}
      onClick={() => onPick(entity.entity_id)}
    >
      <Shape name={empty ? 'alert' : (ROLE_ICONS[role] ?? 'database')} accent={accent} />
      <span className="info">
        <span className="primary mono">{entity.entity_id}</span>
        <span className="secondary">
          {empty
            ? `${words} — never produced a row.`
            : `${words}. ${count(rows)} rows, last ${relativeTime(entity.last)}.`}
        </span>
      </span>
      <Chip label={empty ? 'no rows' : tracked ? 'read' : 'unused'}
            icon={empty ? 'alert' : tracked ? 'check' : 'minus'}
            accent={accent} />
    </button>
  )
}

export function ArchiveCard({ archive, picked, onPick }: {
  archive: Archive | null
  picked: string
  onPick: (entityId: string) => void
}) {
  if (!archive) return <p className="empty">Loading…</p>
  if (!archive.available) return <p className="empty">{archive.reason}</p>

  const { span, entities } = archive
  const untracked = entities.filter((e) => !e.tracked).length
  const empty = entities.filter((e) => e.tracked && e.rows === 0).length

  return (
    <>
      <Row
        icon="database"
        accent="aqua"
        primary={`${count(span.rows)} state changes over ${span.days} days`}
        secondary={`${bytes(span.bytes)} on disk. ${
          span.first ? `From ${absoluteTime(span.first)} to ${absoluteTime(span.last)}.` : ''
        }`}
      />
      {(untracked > 0 || empty > 0) && (
        <Row
          icon={empty > 0 ? 'alert' : 'eye-off'}
          accent={empty > 0 ? 'orange' : 'grey'}
          muted={empty === 0}
          primary={
            empty > 0
              ? `${empty} configured ${empty === 1 ? 'entity has' : 'entities have'} no history`
              : `${untracked} ${untracked === 1 ? 'entity is' : 'entities are'} not read`
          }
          secondary={
            empty > 0
              ? 'The archive kept them, but nothing has ever reported them. Check the entity ids on Setup.'
              : 'Left over from an earlier configuration. Harmless — the feature build ignores them.'
          }
        />
      )}

      <p className="subhead">
        {entities.length} {entities.length === 1 ? 'entity' : 'entities'} · pick one to follow it
      </p>
      <div className="pickers">
        {entities.map((e) => (
          <EntityTile key={e.entity_id} entity={e}
                      current={e.entity_id === picked} onPick={onPick} />
        ))}
      </div>
    </>
  )
}
