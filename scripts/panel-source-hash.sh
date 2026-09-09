#!/usr/bin/env bash
# Print one sha256 over everything Vite reads to produce <tree>/panel/dist. The
# bundle is committed, so it can be stale, and nothing in a minified filename
# says which source built it: hash the inputs instead, stamp it at build time
# (build-panel.sh) and compare it back (check-panel.sh).
#
# `sort -z` for reproducibility, since find returns directory order. LC_ALL=C
# because sort collates by locale and the digest is over the ORDER -- without it
# the check passes or fails depending on whose shell ran it.
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
