/**
 * A free-text field for a Row's trailing slot; the panel's only one besides the
 * retention number. `type="password"` is write-only on purpose: the server never
 * serves a stored secret back, so an empty box means "leave it alone" and the
 * placeholder has to say so.
 */
export function Text({
  value,
  onChange,
  label,
  placeholder,
  type = 'text',
  wide = false,
}: {
  value: string
  onChange: (value: string) => void
  label: string
  placeholder?: string
  type?: 'text' | 'password'
  wide?: boolean
}) {
  return (
    <span className={wide ? 'text wide' : 'text'}>
      <input
        type={type}
        value={value}
        aria-label={label}
        placeholder={placeholder}
        spellCheck={false}
        autoComplete={type === 'password' ? 'new-password' : 'off'}
        onChange={(e) => onChange(e.target.value)}
      />
    </span>
  )
}
