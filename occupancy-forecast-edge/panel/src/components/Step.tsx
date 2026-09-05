import type { ReactNode } from 'react'

/**
 * One numbered step of the Data walkthrough.
 *
 * The Data tab used to be seven cards in the same auto-fit grid the other two
 * tabs use, which was the wrong shape for what it is. A grid says "here are
 * seven independent things, in whatever order they happened to fit". The tab is
 * actually one argument in order: a state change becomes a row, the rows become
 * a table, a horizon is allowed to read part of that table, and the result is
 * scored. Read as tiles, that argument does not survive. So: one column,
 * numbered, with a rail down the side.
 *
 * The slab inside a step is still a `Card`, deliberately. `.card` is the query
 * container that `@container (max-width: 30rem)` on `.row.control` measures, and
 * it is what the `Select` popup hangs off. Wrapping it rather than replacing it
 * means neither of those had to be re-derived, and nothing is inserted between
 * the card and its rows.
 *
 * The number is decorative and `aria-hidden`; the accessible name is the `<h2>`,
 * which is what `aria-labelledby` points at.
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
