import type { ReactNode } from 'react'

/**
 * One card: a title, an optional subtitle, and rows. `optional` renders the
 * small grey marker -- three of the four configuration cards are skippable, and
 * a user who does not know that goes hunting for zones they never created.
 */
export function Card({
  title,
  optional = false,
  quiet = false,
  subtitle,
  badge,
  children,
}: {
  /** Omitted inside a walkthrough step, where the step's own `<h2>` names the
   *  slab and a title here would say it twice. */
  title?: string
  optional?: boolean
  /** A read-only summary rather than something you can change: the page's own
   *  ground and a hairline, so it does not read as a peer of the form. */
  quiet?: boolean
  subtitle?: ReactNode
  /** The verdict on this card's own subject -- "connected", "41 days archived".
   *  Beside the thing it reports on rather than in a status box elsewhere. */
  badge?: ReactNode
  children: ReactNode
}) {
  return (
    <section className={quiet ? 'card quiet' : 'card'}>
      {title && (
        <div className="title">
          <span>
            {title}
            {optional && <small> optional</small>}
          </span>
          {badge}
        </div>
      )}
      {subtitle && <p className="subtitle">{subtitle}</p>}
      {children}
    </section>
  )
}
