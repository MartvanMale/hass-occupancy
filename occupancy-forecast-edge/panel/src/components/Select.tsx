import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { Icon } from './Icon'

/**
 * A dropdown, because the native one could not be made to look like the panel.
 *
 * A browser draws an open `<select>` itself, with the operating system's colours
 * and no CSS reaching inside it: on a dark panel that meant a white list of 159
 * countries with no way to search it. Replacing the control is the only way to
 * style the open state, so this is the ARIA combobox pattern rather than a div
 * that happens to look like one -- it is reachable by keyboard, it announces
 * itself, and Escape closes it.
 *
 * The filter box is opt-in per picker rather than switched on by counting the
 * options. Only the holiday calendar asks for it -- it has 159 countries and is
 * unusable without one -- and the zone and person-group pickers stay plain
 * dropdowns however many entities a household turns out to have. A threshold
 * would mean the ninth zone silently changing a control the user knows.
 */

export interface Option {
  value: string
  label: string
}

/** Roughly the popup's height. Below this much room, open upwards instead --
 *  the panel is an iframe, and a popup is clipped by it where a native one
 *  would have escaped to the desktop. */
const POPUP_H = 260

export function Select({
  label,
  value,
  options,
  onChange,
  searchable = false,
}: {
  label: string
  value: string
  options: Option[]
  onChange: (value: string) => void
  /** Show a filter box. Worth it past a few dozen options, silly below that. */
  searchable?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const [flip, setFlip] = useState(false)

  const root = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const search = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const id = useId()

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? options.filter((o) => o.label.toLowerCase().includes(q)) : options
  }, [options, query])

  const selected = options.find((o) => o.value === value)

  function close() {
    setOpen(false)
    button.current?.focus()
  }

  function choose(next: string) {
    onChange(next)
    setOpen(false)
    button.current?.focus()
  }

  // Opening resets the filter and points the cursor at what is already chosen,
  // so Enter twice is a no-op rather than a silent change to the first option.
  //
  // Keyed on `open` ALONE, deliberately. Every caller builds `options` as an
  // array literal in render, so listing it here re-ran this on every parent
  // render -- and the parent re-renders on every status poll, so an open
  // picker had its search box wiped and re-focused every ten seconds while
  // the user was typing into it. The values read inside are the ones current
  // at the moment of opening, which is what "opening resets" means.
  useEffect(() => {
    if (!open) return
    setQuery('')
    const i = options.findIndex((o) => o.value === value)
    setActive(i < 0 ? 0 : i)

    const rect = root.current?.getBoundingClientRect()
    if (rect) setFlip(window.innerHeight - rect.bottom < POPUP_H && rect.top > POPUP_H)

    search.current?.focus()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // Keep the cursor visible while arrowing through 159 countries.
  useEffect(() => {
    if (!open) return
    list.current?.querySelector('.active')?.scrollIntoView({ block: 'nearest' })
  }, [open, active])

  useEffect(() => {
    if (!open) return
    const outside = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', outside)
    return () => document.removeEventListener('mousedown', outside)
  }, [open])

  function onKeyDown(e: KeyboardEvent) {
    if (!open) {
      if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        setOpen(true)
      }
      return
    }
    switch (e.key) {
      case 'Escape':
        e.preventDefault()
        close()
        break
      case 'Tab':
        setOpen(false)
        break
      case 'ArrowDown':
        e.preventDefault()
        setActive((i) => Math.min(i + 1, shown.length - 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        setActive((i) => Math.max(i - 1, 0))
        break
      case 'Home':
        e.preventDefault()
        setActive(0)
        break
      case 'End':
        e.preventDefault()
        setActive(shown.length - 1)
        break
      case 'Enter': {
        e.preventDefault()
        const option = shown[active]
        if (option) choose(option.value)
        break
      }
      default:
        break
    }
  }

  // aria-activedescendant has to sit on whatever holds focus, which is the
  // filter box when there is one and the trigger when there is not.
  const activeId = open && shown[active] ? `${id}-opt-${active}` : undefined

  return (
    <div className="select" ref={root} onKeyDown={onKeyDown}>
      <button
        type="button"
        ref={button}
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-label={label}
        aria-activedescendant={searchable ? undefined : activeId}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="value">{selected?.label ?? ''}</span>
        <Icon name="chevron" />
      </button>

      {open && (
        <div className={flip ? 'popup up' : 'popup'}>
          {searchable && (
            <input
              ref={search}
              type="text"
              className="search"
              value={query}
              placeholder="Type to filter"
              aria-label={`Filter ${label}`}
              aria-controls={`${id}-list`}
              aria-activedescendant={activeId}
              onChange={(e) => {
                setQuery(e.target.value)
                setActive(0)
              }}
            />
          )}
          <ul id={`${id}-list`} ref={list} role="listbox" aria-label={label}>
            {shown.length === 0 && (
              <li role="presentation" className="none">
                Nothing matches “{query}”
              </li>
            )}
            {shown.map((option, i) => (
              <li
                key={option.value}
                id={`${id}-opt-${i}`}
                role="option"
                aria-selected={option.value === value}
                className={i === active ? 'active' : undefined}
                onMouseEnter={() => setActive(i)}
                onClick={() => choose(option.value)}
              >
                <span>{option.label}</span>
                {/* A tick, not just aqua text. On the dark popup surface the
                    accent measures 4.16:1 against 14px text -- under AA -- and
                    "which one is chosen" is not a thing to say in colour
                    alone anyway. */}
                {option.value === value && <Icon name="check" />}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
