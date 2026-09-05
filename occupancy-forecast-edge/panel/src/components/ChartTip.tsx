import type { ReactNode } from 'react'

/**
 * The floating readout over a chart.
 *
 * Positioned as a percentage of the wrapper, with a THREE-WAY transform clamp
 * rather than a measurement: centred in the middle of the chart, pinned to the
 * left edge in the first fifth and to the right edge in the last. Measuring
 * would mean a layout read on every pointer move to save two edge cases.
 *
 * `aria-hidden`, and that is not an oversight. The chart's own `aria-label` and
 * the prose paragraph under it already carry everything this says, in a form a
 * screen reader can actually get at; announcing a value that follows a mouse
 * would be noise on top of them.
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
