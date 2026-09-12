import type { ReactNode } from 'react'

/**
 * One numbered step of the Data walkthrough: the tab is one argument in order,
 * not seven independent things, so it is a column rather than an auto-fit grid.
 * The slab inside is still a `Card` -- `.card` is the query container that
 * `@container (max-width: 30rem)` measures and what the `Select` popup hangs off.
 */
export function Step({ n, eyebrow, title, id, intro, children }: {
  n: number
  eyebrow: string
  title: string
  /** Also what the heading id is built from. */
  id: string
  intro?: ReactNode
  children: ReactNode
}) {
  const headingId = `step-${id}-head`
  return (
    <section className="step" aria-labelledby={headingId}>
      <div className="step-rail" aria-hidden="true">
        <span className="step-no">{n}</span>
        <span className="wire" />
      </div>
      <div className="step-body">
        <p className="step-eyebrow">{eyebrow}</p>
        <h2 className="step-head" id={headingId}>{title}</h2>
        {intro && <p className="subtitle step-intro">{intro}</p>}
        {children}
      </div>
    </section>
  )
}
