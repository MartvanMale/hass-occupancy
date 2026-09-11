import type { ReactNode } from 'react'

/**
 * Which of the 48 horizons the page is talking about. A range, not a `Select`:
 * a horizon is an ordered scalar people scrub through, and native gets a
 * forced-colors rendering for free. `aria-valuetext` says "+24 hours ahead".
 */
export function HorizonSlider({ value, onChange, max = 48, summary }: {
  value: number
  onChange: (h: number) => void
  max?: number
  /** What this horizon costs and is allowed, in one line under the readout. */
  summary: ReactNode
}) {
  return (
    <div className="hz">
      <div className="hzhead">
        <span className="readout num">+{value} h</span>
        <span className="secondary hzsum">{summary}</span>
      </div>
      <input
        type="range"
        className="range"
        min={1}
        max={max}
        step={1}
        value={value}
        aria-label="Horizon, hours ahead"
        aria-valuetext={`+${value} hours ahead`}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="ticks num" aria-hidden="true">
        <span>+1 h</span><span>+12 h</span><span>+24 h</span>
        <span>+36 h</span><span>+{max} h</span>
      </div>
    </div>
  )
}
