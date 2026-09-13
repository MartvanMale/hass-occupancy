import { Icon } from './Icon'

/**
 * The one Save for Setup and Connections together. Docked, because the control
 * you just changed can be two screens above it; shared, because one form backs
 * both tabs, so a save from either writes both -- and "Unsaved changes" is what
 * makes that honest rather than surprising.
 */
export function SaveDock({
  loaded,
  saving,
  saved,
  dirty,
  complaint,
}: {
  loaded: boolean
  saving: boolean
  saved: boolean
  dirty: boolean
  complaint: string | null
}) {
  return (
    <div className="actions dock">
      <button type="submit" disabled={!loaded || saving || complaint !== null}>
        <Icon name="save" />
        {saving ? 'Saving…' : 'Save'}
      </button>
      <span className={saved ? 'saved on' : 'saved'}>Saved</span>
      {complaint && <span className="error">{complaint}</span>}
      {dirty && !complaint && <span className="dirty">Unsaved changes</span>}
    </div>
  )
}
