import type { ReactNode } from 'react'
import { Shape, type Accent, type IconName } from './Icon'

/**
 * One Mushroom row: shape, then primary over secondary, then a trailing control.
 *
 * The trailing slot takes whatever the row is *for* -- a checkbox, a select, a
 * chip -- which is why it is a node and not a variant. `as="label"` is what makes
 * the whole row a click target for the control inside it; a 36px checkbox on a
 * phone is not one.
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
  /** The trailing slot holds a <select>, which is the only thing wide enough to
   *  need a line of its own on a narrow card. Chips and checkboxes never do, and
   *  letting them wrap was worse than the problem it was meant to solve. */
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
