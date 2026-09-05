import { useCallback, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent, RefObject } from 'react'
import { clamp } from './geometry'

/**
 * Where the pointer is over a chart, as an index and a fraction.
 *
 * The charts here carried native `<title>` tooltips: a hundred and twenty
 * transparent `<rect>`s, each with a `<title>` the browser shows after its own
 * delay, in its own colours, positioned wherever it likes. That is fine for a
 * label on an icon and wrong for reading a value off a line -- it is slow, it
 * cannot show three series at once, and it cannot be styled to match the panel.
 *
 * **The mapping goes through the WRAPPER's `getBoundingClientRect`, never
 * through SVG user units.** `getScreenCTM()` would in fact be affine-correct
 * even under `preserveAspectRatio="none"`, so this is not about correctness
 * today -- it is that the moment the wrapper holds anything besides one
 * full-bleed SVG the two coordinate systems disagree, silently. The rect is the
 * wrapper's, and the wrapper is what the tooltip is absolutely positioned
 * inside, so one system does both jobs.
 *
 * Touch works by pointer capture rather than a document listener: `pan-y` in the
 * CSS leaves a vertical swipe scrolling the panel, and a horizontal drag
 * scrubs. A mouse shows on move and clears on leave.
 */
export type PointerMode = 'point' | 'band'

export interface ChartPointer {
  ref: RefObject<HTMLDivElement | null>
  /** Nearest sample, or null when the pointer is away. */
  index: number | null
  /** 0..1 across the wrapper, for positioning the crosshair and the tip. */
  fraction: number
  handlers: {
    onPointerMove: (e: ReactPointerEvent<HTMLDivElement>) => void
    onPointerDown: (e: ReactPointerEvent<HTMLDivElement>) => void
    onPointerUp: (e: ReactPointerEvent<HTMLDivElement>) => void
    onPointerCancel: () => void
    onPointerLeave: () => void
  }
}

export function useChartPointer(count: number, mode: PointerMode = 'point'): ChartPointer {
  const ref = useRef<HTMLDivElement>(null)
  const [index, setIndex] = useState<number | null>(null)
  const [fraction, setFraction] = useState(0)

  const track = useCallback((clientX: number) => {
    const el = ref.current
    if (!el || count < 1) return
    const rect = el.getBoundingClientRect()
    const f = clamp((clientX - rect.left) / (rect.width || 1), 0, 1)
    setFraction(f)
    // 'band' for a chart whose marks are cells of equal width -- the click
    // should land in the cell under the pointer. 'point' for a line, where the
    // nearest sample is the one being read.
    setIndex(mode === 'band'
      ? clamp(Math.floor(f * count), 0, count - 1)
      : clamp(Math.round(f * (count - 1)), 0, count - 1))
  }, [count, mode])

  const clear = useCallback(() => setIndex(null), [])

  return {
    ref,
    index,
    fraction,
    handlers: {
      onPointerMove: (e) => track(e.clientX),
      onPointerDown: (e) => {
        // Only for touch and pen. Capturing the mouse would swallow a click on
        // anything laid over the chart.
        if (e.pointerType !== 'mouse') e.currentTarget.setPointerCapture(e.pointerId)
        track(e.clientX)
      },
      onPointerUp: (e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) {
          e.currentTarget.releasePointerCapture(e.pointerId)
          clear()
        }
      },
      onPointerCancel: clear,
      onPointerLeave: clear,
    },
  }
}
