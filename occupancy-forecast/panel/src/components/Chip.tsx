import type { CSSProperties } from 'react'
import { Icon, type Accent, type IconName } from './Icon'

export function Chip({
  label,
  icon,
  accent,
}: {
  label: string
  icon: IconName
  accent: Accent
}) {
  return (
    <span className="chip" style={{ '--c': `var(--rgb-${accent})` } as CSSProperties}>
      <Icon name={icon} />
      {label}
    </span>
  )
}
