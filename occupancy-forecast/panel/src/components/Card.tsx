import type { ReactNode } from 'react'

/**
 * One card in the grid: a title, an optional subtitle, and rows.
 *
 * `optional` renders the small grey "optional" after the title -- the panel uses
 * it to say which settings a household can simply not have, which matters more
 * than it looks: three of the four configuration cards are skippable and a user
 * who does not know that will go hunting for zones they never created.
 */
export function Card({
  title,
  optional = false,
  subtitle,
  children,
}: {
  /** Omitted inside a walkthrough step, where the step's own `<h2>` names the
   *  slab and a title here would say it twice. */
  title?: string
  optional?: boolean
  subtitle?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="card">
      {title && (
        <div className="title">
          {title}
          {optional && <small> optional</small>}
        </div>
      )}
      {subtitle && <p className="subtitle">{subtitle}</p>}
      {children}
    </section>
  )
}
