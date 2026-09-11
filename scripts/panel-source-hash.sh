#!/usr/bin/env bash
# Print one sha256 over everything Vite reads to produce <tree>/panel/dist, so a
# stale committed bundle can be caught: stamped by build-panel.sh, compared by
# check-panel.sh. `sort -z` and LC_ALL=C: the digest is over the ORDER, so without
# them the check passes or fails by whose shell ran it.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."

TREE="${1:-occupancy-forecast-edge}"
PANEL="$TREE/panel"
[[ -d "$PANEL" ]] || { echo "no panel in $TREE" >&2; exit 1; }

# Everything the build reads, nothing it writes. package-lock.json included: a
# dependency bump changes the bundle without touching src/.
cd "$PANEL"
find index.html package.json package-lock.json tsconfig.json vite.config.ts src \
     -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sha256sum \
  | cut -d' ' -f1
