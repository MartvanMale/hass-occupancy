#!/usr/bin/env bash
# Print one sha256 over everything Vite reads to produce <tree>/panel/dist.
#
# This exists because the built bundle is now committed, and a committed
# artifact can be stale. Nothing about a minified `index-CAV3BYGI.js` tells you
# whether it was built from the `panel/src` sitting next to it, and the content
# hash in the filename is over the OUTPUT, so a source edit that happens to
# minify to the same bytes keeps the same name. So: hash the inputs, write the
# result into dist/ at build time (scripts/build-panel.sh), and compare it back
# (scripts/check-panel.sh).
#
# `sha256sum` hashes the path as well as the contents, so a rename or a deleted
# file moves the digest -- which a hash of concatenated contents would not. The
# `sort -z` is what makes it reproducible: `find` returns directory order, which
# differs between two checkouts of the same commit.
set -euo pipefail
cd "$(dirname "$0")/.."

TREE="${1:-occupancy-forecast-edge}"
PANEL="$TREE/panel"
[[ -d "$PANEL" ]] || { echo "no panel in $TREE" >&2; exit 1; }

# Everything the build reads, and nothing it writes. package-lock.json is in
# here because the toolchain is part of the input: `npm ci` pins it, so a
# dependency bump changes the bundle without touching a line of src/.
cd "$PANEL"
find index.html package.json package-lock.json tsconfig.json vite.config.ts src \
     -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sha256sum \
  | cut -d' ' -f1
