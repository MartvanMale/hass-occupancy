import type { ReactNode } from 'react'
import { Shape, type Accent, type IconName } from './Icon'

/**
 * One Mushroom row: shape, then primary over secondary, then a trailing control.
 * The trailing slot is a node, not a variant. `as="label"` makes the whole row
 * a click target -- a 36px checkbox on a phone is not one.
 */
export function Row({
  icon,
  accent,
  primary,
  secondary,
  trailing,
  as = 'div',
  muted = false,
  control = false,
}: {
  icon: IconName
  accent: Accent
  primary: ReactNode
  secondary?: ReactNode
  trailing?: ReactNode
  as?: 'div' | 'label'
  muted?: boolean
  /** The trailing slot holds a `<select>`, the only thing wide enough to need
   *  its own line. Letting chips and checkboxes wrap was worse. */
  control?: boolean
}) {
  const Tag = as
  const className = ['row', muted && 'off', control && 'control']
    .filter(Boolean)
    .join(' ')
  return (
    <Tag className={className}>
      <Shape name={icon} accent={accent} />
      <div className="info">
        <div className="primary">{primary}</div>
        {secondary && <div className="secondary">{secondary}</div>}
      </div>
      {trailing}
    </Tag>
  )
}
