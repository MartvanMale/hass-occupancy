import type { ReactNode } from 'react'
import { Shape, type Accent, type IconName } from './Icon'

/**
 * One SETTING: label over hint, then the control, on a shared grid. `Row` is its
 * status twin and sizes a trailing input by its own value, which left a column
 * of text fields at three different widths. Omit `icon` where a column of
 * shapes would only repeat the card's title.
 */
export function Field({
  label,
  hint,
  control,
  icon,
  accent = 'blue',
}: {
  label: ReactNode
  hint?: ReactNode
  control: ReactNode
  icon?: IconName
  accent?: Accent
}) {
  return (
    <div className={icon ? 'field' : 'field plain'}>
      {icon && <Shape name={icon} accent={accent} />}
      <div className="info">
        <div className="primary">{label}</div>
        {hint && <div className="secondary">{hint}</div>}
      </div>
      {control}
    </div>
  )
}
