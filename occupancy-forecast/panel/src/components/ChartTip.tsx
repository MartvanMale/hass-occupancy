import type { ReactNode } from 'react'

/**
 * The floating readout over a chart: a three-way transform clamp rather than a
 * measurement, so there is no layout read per pointer move. `aria-hidden` is
 * not an oversight -- the chart's `aria-label` and the prose under it already
 * carry this.
 */
export function ChartTip({ fraction, children }: {
  fraction: number
  children: ReactNode
}) {
  const transform = fraction < 0.2 ? 'translateX(0)'
    : fraction > 0.8 ? 'translateX(-100%)'
      : 'translateX(-50%)'
  return (
    <div className="tip" aria-hidden="true"
         style={{ left: `${(fraction * 100).toFixed(2)}%`, transform }}>
      {children}
    </div>
  )
}

/** One line of a tip: a swatch, what it is, and the number. */
export function TipRow({ accent, label, value }: {
  accent?: string
  label: string
  value: string
}) {
  return (
    <div className="tiprow">
      <i style={accent ? { background: `rgb(var(--rgb-${accent}))` } : { background: 'transparent' }} />
      <span>{label}</span>
      <b className="num">{value}</b>
    </div>
  )
}
