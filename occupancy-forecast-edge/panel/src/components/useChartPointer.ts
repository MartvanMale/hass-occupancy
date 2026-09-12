import { useCallback, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent, RefObject } from 'react'
import { clamp } from './geometry'

/**
 * Where the pointer is over a chart, as an index and a fraction.
 *
 * The mapping goes through the WRAPPER's `getBoundingClientRect`, never SVG
 * user units: the wrapper is what the tooltip is positioned inside, so one
 * coordinate system does both jobs. Touch works by pointer capture, so `pan-y`
 * leaves a vertical swipe scrolling the panel.
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
    // 'band' where the marks are equal-width cells; 'point' for a line, where
    // the nearest sample is the one being read.
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
