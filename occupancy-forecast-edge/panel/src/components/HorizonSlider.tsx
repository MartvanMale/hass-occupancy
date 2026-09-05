import type { ReactNode } from 'react'

/**
 * Which of the 48 horizons the page is talking about.
 *
 * A range, not the `Select` every other control on the tab uses, and the
 * distinction is honest rather than decorative: a `Select` is for a named set --
 * an entity, a subject, a column -- and a horizon is an ordered scalar with 48
 * evenly spaced values that people scrub through to watch the daily lags drop
 * away. Reading that off a dropdown means 48 open-pick-close cycles.
 *
 * Native, styled with `accent-color`, rather than a rebuilt
 * `::-webkit-slider-thumb`. That is already the panel's answer for the
 * checkboxes, it derives from the same token, and -- the real argument -- a
 * native control is rendered by `forced-colors` in system colours for free,
 * where a hand-drawn one would need its own fallback and would be the only
 * invented control on the page.
 *
 * `aria-valuetext` because "24" is not what the number means; a screen reader
 * should say "+24 hours ahead". The tick row is `aria-hidden` -- the value is
 * already announced, and it is a scale, not content.
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
